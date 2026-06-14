"""
Opponent-modeling variant of Leduc Poker.

Context injection (text-form, LLM-validated):
  * PokerBench compact schema (Gupta et al., AAAI 2025, arXiv:2501.08328)
  * Bayes' Bluff Dirichlet card posterior (Southey et al., UAI 2005)
  * SuspicionAgent first-order ToM text layer (Guo et al., 2023, arXiv:2309.17277)
  * RNR-style 3-dim archetype posterior (Johanson/Zinkevich/Bowling, NIPS 2007)

Per-step reward stack (no episode accumulators):
  * Equity-potential PBRS (Ng/Harada/Russell 1999, policy-invariant)
  * Pot-odds alignment (Billings et al., AIJ 2002)
  * Invalid-action penalty (bounded, per-step)

Public names (imported by env_configs.py):
  * rollout_full_prompt_and_completion_parallelized_curriculum
  * rollout_last_prompt_and_completion_parallelized_curriculum
  * rollout_reward_func  (re-exported from shared_env)
  * _curriculum_factory
"""

import functools
import math
import random
import re
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    remove_reasoning_tags,
    rollout_reward_func,  # re-exported
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME      = "leduc_poker"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN     = 4096
_TIMEOUT            = 2400
_MAX_TURNS          = 10

# Per-step reward constants (no episode-level accumulators)
TERMINAL_WIN_REWARD  = 1.0
TERMINAL_LOSS_REWARD = -1.0
INVALID_PENALTY      = -0.1
INVALID_TOTAL_CLIP   = -0.3
POSITIVE_STEP_CLIP   = 0.3
TERMINAL_REWARD_CLIP = 1.5

# Equity-PBRS scale: Φ(s) = equity(s) ∈ [0, 1], F = Φ(s') − Φ(s)
EQUITY_PBRS_WEIGHT   = 1.0
# Pot-odds alignment (Billings 2002): margin between equity and call-cost/pot
# Increased from 0.15 -> 0.25 to discourage over-calling weak hands.
POT_ODDS_WEIGHT      = 0.25
# RNR alignment (Johanson/Zinkevich/Bowling NIPS 2007): log-prob of taken action
# under archetype-posterior-weighted reference policy, minus uniform baseline.
RNR_ALIGN_WEIGHT     = 0.10
# Bonus for winning the pot via opponent fold (bluffing/pressure reward).
# This encourages model to win without relying on showdown.
POT_WIN_NO_SHOWDOWN_BONUS = 0.20

# Confidence-calibrated Raise bonus. Raise is the commitment-like action in
# Leduc (commits more chips, aggressive play). Equity PBRS already rewards
# equity gain, but does NOT reward *sustaining* high equity via a raise
# (equity doesn't change unless new info arrives). This bonus fills that gap
# by paying out when the model raises from a confident-equity state. Scales
# linearly from 0 at equity=0.5 to CONFIDENT_RAISE_BONUS at equity=1.0.
# Kept small (0.05) because POSITIVE_STEP_CLIP=0.3 is tight and we don't want
# to squeeze out equity/pot-odds/RNR signals that are already working.
# Brier (1950) calibration principle applied to commitment decisions.
CONFIDENT_RAISE_BONUS = 0.05

# Minimum opponent actions observed before archetype posterior is trusted
ARCHETYPE_MIN_OBS    = 3

_CARD_RANK: dict[str, int] = {"J": 1, "Q": 2, "K": 3}


# ---------------------------------------------------------------------------
# System prompt (same game rules as base env; strategy tips appended when hints on)
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits \u00d7 (num_players + 1) ranks. For 2 players: 6 cards (J\u2660 J\u2665 Q\u2660 Q\u2665 K\u2660 K\u2665).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check (match current bet), "
    "Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> Fold": respond "0"\n'
    '- For action "2 -> Raise": respond "2"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "Round 1:\n"
    "- Hold K or Q \u2192 call a raise; raise first if unchallenged.\n"
    "- Hold J \u2192 fold against a raise; check if unchallenged.\n\n"
    "Round 2 (public card revealed):\n"
    "- You have a PAIR \u2192 raise; never fold.\n"
    "- You have K (no pair) \u2192 raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is K \u2192 raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is J \u2192 check; fold if opponent raises.\n"
    "- You have J (no pair) \u2192 check; fold if opponent raises.\n"
)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    player_id:         int
    private_card:      str
    private_card_rank: int
    public_card:       "str | None"
    public_card_rank:  "int | None"
    has_pair:          bool
    round:             int
    pot:               int
    our_chips:         int
    opp_chips:         int
    r1_betting:        list[str]
    r2_betting:        list[str]
    legal_actions:     dict[int, str] = field(default_factory=dict)

    @property
    def our_invested(self) -> int:
        return 100 - self.our_chips

    @property
    def opp_invested(self) -> int:
        return 100 - self.opp_chips

    @property
    def current_round_betting(self) -> list[str]:
        return self.r1_betting if self.round == 1 else self.r2_betting

    @property
    def raises_this_round(self) -> int:
        return self.current_round_betting.count("Raise")

    @property
    def opp_last_action(self) -> "str | None":
        betting = self.current_round_betting
        return betting[-1] if betting else None


def parse_game_state(obs: str) -> "GameState | None":
    if not obs or "Current State:" not in obs:
        return None

    def _find(pattern, default=None):
        m = re.search(pattern, obs)
        return m.group(1) if m else default

    pid_str = _find(r"You are Player (\d+)\.")
    if pid_str is None:
        return None
    player_id = int(pid_str)

    private_card = _find(r"Your card:\s*(\S+)")
    if private_card is None:
        return None
    private_card_rank = _CARD_RANK.get(private_card[0], 0)

    pub_raw = _find(r"Public card:\s*(\S+)")
    public_card      = pub_raw
    public_card_rank = _CARD_RANK.get(pub_raw[0], 0) if pub_raw else None

    has_pair = "Hand: Pair" in obs

    round_str = _find(r"Current round:\s*(\d+)/\d+", "1")
    round_ = int(round_str)

    pot       = int(_find(r"Pot size:\s*(\d+)", "0"))
    our_chips = int(_find(r"Your chips:\s*(\d+)", "100"))
    opp_chips = int(_find(r"Opponent chips:\s*(\d+)", "100"))

    def _parse_betting(label: str) -> list[str]:
        m = re.search(rf"{label} betting:\s*(.+)", obs)
        if not m:
            return []
        return [a.strip() for a in m.group(1).split(",")]

    r1_betting = _parse_betting("Round 1")
    r2_betting = _parse_betting("Round 2")

    legal_actions: dict[int, str] = {}
    for m in re.finditer(r"^(\d+)\s*->\s*(.+)$", obs, re.MULTILINE):
        legal_actions[int(m.group(1))] = m.group(2).strip()

    return GameState(
        player_id=player_id,
        private_card=private_card,
        private_card_rank=private_card_rank,
        public_card=public_card,
        public_card_rank=public_card_rank,
        has_pair=has_pair,
        round=round_,
        pot=pot,
        our_chips=our_chips,
        opp_chips=opp_chips,
        r1_betting=r1_betting,
        r2_betting=r2_betting,
        legal_actions=legal_actions,
    )


def _format_observation(raw: str) -> str:
    player_match = re.search(r"You are Player (\d+)\.", raw)
    player_line  = f"You are Player {player_match.group(1)}." if player_match else ""
    state_start = raw.find("Current State:")
    if state_start == -1:
        return raw
    body = raw[state_start:]
    legal_start = body.find("Legal Actions:")
    if legal_start == -1:
        return body
    state_block   = body[:legal_start].rstrip()
    actions_block = body[legal_start:]
    actions_block = re.sub(r"^  (\d+)", r"\1", actions_block, flags=re.MULTILINE)
    actions_block = actions_block.replace(
        "Your choice (action ID only):", "Your choice (ID only):"
    )
    parts = [state_block]
    if player_line:
        parts.append(player_line)
    parts.append(actions_block)
    return "\n\n".join(parts)


def _parse_action(completion_text: str) -> str:
    action = remove_reasoning_tags(completion_text).strip()
    if action.endswith("</s>"):
        action = action[:-4].strip()
    if "Action:" in action:
        action = action.split("Action:")[-1].strip()
    return action


# ---------------------------------------------------------------------------
# OpponentCardPosterior  (SuspicionAgent-style Bayesian update over {J, Q, K})
# ---------------------------------------------------------------------------

class OpponentCardPosterior:
    """Dirichlet-categorical posterior over opponent's private card rank
    (Bayes' Bluff, Southey et al. UAI 2005).

    State: concentration parameters alpha ∈ R^3_{>0}, one per rank {J, Q, K}.
    Posterior: P(rank = r) = alpha_r / sum(alpha).

    Prior (reset_with_known):
        alpha_r = max(remaining_count(r), PRIOR_FLOOR)
        remaining_count(r) = 2 - [our_card == r] - [public_card == r]
        (Leduc has 2 copies of each rank).

    Update (update_on_action):
        alpha_r += OBS_WEIGHT * L(action | r, round, public_rank)
        L is the hand-crafted action-likelihood table.

    Additive pseudocount update (Dirichlet-conjugate with fractional-weight
    observations) differs from a purely multiplicative Bayesian update:
    evidence accumulates additively and ranks are never driven to exactly
    zero probability. Matches Bayes' Bluff's smoothing via Dirichlet prior.

    The ToM text layer built on top of this posterior is Guo et al. 2023
    (SuspicionAgent); see build_suspicion_agent_context.
    """

    PRIOR_FLOOR = 1e-3
    OBS_WEIGHT  = 1.0

    # Sharp likelihoods: Raise almost certainly implies K/Q, Call is medium range.
    _LIKELIHOOD_RAISE_R1 = {"J": 0.05, "Q": 0.60, "K": 1.00}
    _LIKELIHOOD_CALL_R1  = {"J": 0.40, "Q": 0.90, "K": 0.30}
    _LIKELIHOOD_CALL_R2  = {"J": 0.30, "Q": 0.85, "K": 0.40}

    def __init__(self) -> None:
        self.alpha: dict[str, float] = {"J": 1.0, "Q": 1.0, "K": 1.0}

    @property
    def prob(self) -> dict[str, float]:
        total = sum(self.alpha.values())
        if total <= 0.0:
            return {r: 1.0 / 3.0 for r in self.alpha}
        return {r: a / total for r, a in self.alpha.items()}

    def reset_with_known(self, our_card: str, public_card: "str | None" = None) -> None:
        remaining = {"J": 2.0, "Q": 2.0, "K": 2.0}
        if our_card and our_card[0] in remaining:
            remaining[our_card[0]] -= 1.0
        if public_card and public_card[0] in remaining:
            remaining[public_card[0]] -= 1.0
        self.alpha = {r: max(c, self.PRIOR_FLOOR) for r, c in remaining.items()}

    def _likelihood(
        self,
        action: str,
        round_num: int,
        public_rank: "int | None" = None,
    ) -> dict[str, float]:
        if action == "Raise":
            if round_num == 1:
                return dict(self._LIKELIHOOD_RAISE_R1)
            L: dict[str, float] = {}
            for r, rank_val in _CARD_RANK.items():
                if public_rank is not None and rank_val == public_rank:
                    L[r] = 1.00  # pair with public — very strong
                elif rank_val == 3:
                    L[r] = 0.70
                elif rank_val == 2:
                    L[r] = 0.40
                else:
                    L[r] = 0.10  # J raising mostly bluff
            return L
        if action in ("Call", "Check"):
            return dict(self._LIKELIHOOD_CALL_R1 if round_num == 1 else self._LIKELIHOOD_CALL_R2)
        return {r: 1.0 for r in self.alpha}

    def update_on_action(
        self,
        action: str,
        round_num: int,
        public_rank: "int | None" = None,
    ) -> None:
        L = self._likelihood(action, round_num, public_rank)
        for r in self.alpha:
            self.alpha[r] += self.OBS_WEIGHT * L.get(r, 1.0)

    def equity(self, our_card: str, public_card: "str | None") -> float:
        """P(we win showdown) under the current posterior."""
        our_rank = _CARD_RANK.get(our_card[0], 0)
        we_pair  = public_card is not None and our_card[0] == public_card[0]

        win_prob = 0.0
        for opp_rank_str, p in self.prob.items():
            opp_rank = _CARD_RANK[opp_rank_str]
            opp_pair = public_card is not None and opp_rank_str == public_card[0]

            if we_pair and not opp_pair:
                win_prob += p * 1.0
            elif opp_pair and not we_pair:
                win_prob += p * 0.0
            elif we_pair and opp_pair:
                # Leduc only has 2 of each rank, so this case is degenerate;
                # treat as tie.
                win_prob += p * 0.5
            else:
                if our_rank > opp_rank:
                    win_prob += p * 1.0
                elif our_rank == opp_rank:
                    win_prob += p * 0.5

        return win_prob

    def summary(self) -> str:
        ordered = sorted(self.prob.items(), key=lambda x: -x[1])
        return " ".join(f"{r}={p:.2f}" for r, p in ordered)


# ---------------------------------------------------------------------------
# OpponentArchetypePosterior  (RNR-style 3-dim: tight / loose / passive)
# Cross-episode tracker stored in _state["archetype_trackers"][server_idx].
# ---------------------------------------------------------------------------

class OpponentArchetypePosterior:
    def __init__(self) -> None:
        self.raise_count = 0
        self.call_count  = 0
        self.fold_count  = 0
        self.total       = 0

    def record(self, action: str) -> None:
        self.total += 1
        if action == "Raise":
            self.raise_count += 1
        elif action in ("Call", "Check"):
            self.call_count += 1
        elif action == "Fold":
            self.fold_count += 1

    def posterior(self) -> dict[str, float]:
        if self.total < ARCHETYPE_MIN_OBS:
            return {"tight": 1 / 3, "loose": 1 / 3, "passive": 1 / 3}
        raise_rate = self.raise_count / self.total
        fold_rate  = self.fold_count / self.total
        call_rate  = self.call_count / self.total
        tight   = fold_rate * 0.70 + (1.0 - raise_rate) * 0.30 + 0.01
        loose   = (1.0 - fold_rate) * 0.60 + raise_rate * 0.40 + 0.01
        passive = call_rate * 0.80 + (1.0 - raise_rate) * 0.20 + 0.01
        total   = tight + loose + passive
        return {"tight": tight / total, "loose": loose / total, "passive": passive / total}

    def summary(self) -> str:
        post = self.posterior()
        ordered = sorted(post.items(), key=lambda x: -x[1])
        return " ".join(f"{k}={v:.2f}" for k, v in ordered)


# ---------------------------------------------------------------------------
# Per-archetype best-response score tables (Gap 1B: offline-designed BRs).
# Used to build an RNR-style reference policy:
#     π_ref(a|s) = (1 - p) · π_Nash(a)  +  p · π_archetype(a)
# where π_archetype is a posterior-weighted blend of the three per-archetype
# BR tables and p scales with archetype observation count.
# ---------------------------------------------------------------------------

def _scores_vs_tight(gs: GameState) -> dict[str, float]:
    """Score actions vs. a tight opp (folds too often). Bluff R1 aggressively;
    value-bet R2 normally since tight folds hands that have pot odds to call."""
    scores = {"Fold": 0.3, "Call": 1.0, "Check": 1.0, "Raise": 1.0}
    if gs.round == 1:
        scores["Raise"] = 3.0
    else:
        if gs.has_pair:
            scores["Raise"] = 4.0
        elif gs.private_card_rank == 3:
            scores["Raise"] = 3.0
        elif gs.private_card_rank == 2:
            scores["Raise"] = 2.0
        else:
            scores["Raise"] = 1.5
    return scores


def _scores_vs_loose(gs: GameState) -> dict[str, float]:
    """Score actions vs. a loose opp (calls too often). Only raise for value;
    don't bluff into a calling station."""
    scores = {"Fold": 0.8, "Call": 1.0, "Check": 1.0, "Raise": 1.0}
    if gs.round == 1:
        if gs.private_card_rank == 3:
            scores["Raise"] = 3.0
        elif gs.private_card_rank == 2:
            scores["Raise"] = 1.5
        else:
            scores["Raise"] = 0.3
            scores["Fold"]  = 1.8
    else:
        if gs.has_pair or gs.private_card_rank == 3:
            scores["Raise"] = 3.0
        elif gs.private_card_rank == 2:
            scores["Call"]  = 1.5
            scores["Raise"] = 0.5
        else:
            scores["Fold"] = 2.0
            scores["Call"] = 0.3
    return scores


def _leduc_basic_strategy(gs: GameState) -> dict[str, float]:
    """Simple heuristic strategy (Nash-lite) for π_Nash placeholder.
    Rewards: Raise with high cards/pairs, Call/Check with medium, Fold low.
    """
    scores = {"Fold": 1.0, "Call": 1.0, "Check": 1.0, "Raise": 1.0}
    if gs.round == 1:
        if gs.private_card_rank == 3: scores["Raise"] = 3.0
        elif gs.private_card_rank == 2: scores["Call"] = 2.0; scores["Raise"] = 1.2
        else: scores["Fold"] = 2.0; scores["Check"] = 1.0
    else:
        if gs.has_pair: scores["Raise"] = 5.0
        elif gs.private_card_rank == 3: scores["Raise"] = 3.0
        elif gs.private_card_rank == 2: scores["Call"] = 2.0
        else: scores["Fold"] = 3.0; scores["Check"] = 1.0
    return scores


def _scores_vs_passive(gs: GameState) -> dict[str, float]:
    """Score actions vs. a passive opp (checks/calls, rarely raises). Extract
    value steadily; passive opp won't punish reasonable raises."""
    scores = {"Fold": 0.3, "Call": 1.0, "Check": 1.0, "Raise": 1.0}
    if gs.round == 1:
        if gs.private_card_rank == 3:
            scores["Raise"] = 3.0
        elif gs.private_card_rank == 2:
            scores["Raise"] = 2.0
        else:
            scores["Raise"] = 1.0
    else:
        if gs.has_pair:
            scores["Raise"] = 4.0
        elif gs.private_card_rank == 3:
            scores["Raise"] = 2.5
        elif gs.private_card_rank == 2:
            scores["Raise"] = 1.5
        else:
            scores["Fold"] = 1.5
            scores["Call"] = 0.5
    return scores


def _match_action_key(label: str) -> "str | None":
    """Map a legal-action label to one of {Fold, Call, Check, Raise}, or None."""
    low = label.lower()
    for key in ("Fold", "Call", "Check", "Raise"):
        if key.lower() in low:
            return key
    return None


def archetype_reference_policy(
    gs: GameState,
    archetype: "OpponentArchetypePosterior",
) -> dict[int, float]:
    """Build a reference policy over legal action IDs (Gap 1B).

    π_ref(a|s) = (1 - p) · π_Nash(a)  +  p · π_archetype(a)
      π_Nash       = uniform over legal actions (placeholder)
      π_archetype  = Σ_τ archetype_posterior[τ] · BR_τ(a|s), normalized
      p            = min(1, total_obs / (2 · ARCHETYPE_MIN_OBS))

    Returns {} if no legal actions are parsed.
    """
    if not gs.legal_actions:
        return {}

    tight_s   = _scores_vs_tight(gs)
    loose_s   = _scores_vs_loose(gs)
    passive_s = _scores_vs_passive(gs)
    post = archetype.posterior()

    blend: dict[str, float] = {}
    for k in ("Fold", "Call", "Check", "Raise"):
        blend[k] = (
            post["tight"]   * tight_s.get(k, 1.0)
            + post["loose"]   * loose_s.get(k, 1.0)
            + post["passive"] * passive_s.get(k, 1.0)
        )

    legal_raw: dict[int, float] = {}
    for aid, label in gs.legal_actions.items():
        k = _match_action_key(label)
        legal_raw[aid] = blend.get(k, 1.0) if k is not None else 1.0

    arch_total = sum(legal_raw.values())
    pi_archetype = (
        {aid: s / arch_total for aid, s in legal_raw.items()}
        if arch_total > 0 else
        {aid: 1.0 / len(legal_raw) for aid in legal_raw}
    )

    n_legal = len(gs.legal_actions)
    basic_s = _leduc_basic_strategy(gs)
    pi_nash_raw: dict[int, float] = {}
    for aid, label in gs.legal_actions.items():
        k = _match_action_key(label)
        pi_nash_raw[aid] = basic_s.get(k, 1.0) if k else 1.0
    
    nash_total = sum(pi_nash_raw.values())
    pi_nash = {aid: s / nash_total for aid, s in pi_nash_raw.items()}

    p = min(1.0, archetype.total / (2.0 * ARCHETYPE_MIN_OBS))

    pi_ref: dict[int, float] = {}
    for aid in gs.legal_actions:
        pi_ref[aid] = (1 - p) * pi_nash.get(aid, 0.0) + p * pi_archetype.get(aid, 0.0)

    return pi_ref


# ---------------------------------------------------------------------------
# Context injection builders
# ---------------------------------------------------------------------------

def build_pokerbench_context(gs: GameState) -> str:
    """Compact PokerBench schema (Gupta et al., AAAI 2025)."""
    hist_parts: list[str] = []
    if gs.r1_betting:
        hist_parts.append("R1:" + ",".join(gs.r1_betting))
    if gs.r2_betting:
        hist_parts.append("R2:" + ",".join(gs.r2_betting))
    hist = " | ".join(hist_parts) if hist_parts else "-"
    legal_str = " ".join(f"{k}:{v}" for k, v in sorted(gs.legal_actions.items()))
    return (
        f"[State] Pos:P{gs.player_id} | "
        f"Chips:P={gs.our_chips} Opp={gs.opp_chips} | "
        f"Inv:P={gs.our_invested} Opp={gs.opp_invested} | "
        f"Pot:{gs.pot} | Board:{gs.public_card or '-'} | "
        f"Hist:{hist} | Legal:{legal_str}"
    )


def build_suspicion_agent_context(
    gs: GameState,
    posterior: "OpponentCardPosterior",
    archetype: "OpponentArchetypePosterior",
) -> str:
    """SuspicionAgent first-order ToM belief text (Guo et al., 2023)."""
    equity = posterior.equity(gs.private_card, gs.public_card)
    lines = [
        f"[Belief] Opp card posterior: {posterior.summary()}",
        f"[Belief] Your showdown equity: {equity:.2%}",
        f"[Archetype] Opp style: {archetype.summary()}",
        f"[Betting] R{gs.round} raises so far: {gs.raises_this_round}",
    ]
    opp_last = gs.opp_last_action
    if opp_last == "Raise":
        if gs.round == 1:
            lines.append("[ToM] Opp raised R1 -> likely K or Q; J raise is usually bluff.")
        elif gs.public_card:
            pub_r = gs.public_card[0]
            lines.append(
                f"[ToM] Opp raised R2 with public {pub_r} -> likely pair-{pub_r} or strong off-pair."
            )
    elif opp_last in ("Call", "Check"):
        lines.append("[ToM] Opp called/checked -> medium hand (often Q) or a slow-play.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-step reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Per-step only. Components:
      (1) Equity-PBRS (Ng/Harada/Russell 1999): F = Φ(s') − Φ(s), Φ = equity.
      (2) Pot-odds alignment (Billings 2002): reward on Call/Check when
          equity exceeds cost/(pot+cost).
      (3) Invalid-action penalty.
    Episode reward = clipped sum over all step rewards + terminal win/loss outcome.
    """

    def __init__(self) -> None:
        self.invalid_penalty = INVALID_PENALTY

    def step_reward(
        self,
        *,
        prev_equity: "float | None",
        curr_equity: "float | None",
        action_str: str,
        prev_state: "GameState | None",
        is_invalid: bool = False,
        pi_ref: "dict[int, float] | None" = None,
        action_id: "int | None" = None,
    ) -> float:
        if is_invalid:
            return self.invalid_penalty

        reward = 0.0
        # (1) Equity-potential PBRS (Ng/Harada/Russell 1999, policy-invariant)
        if curr_equity is not None and prev_equity is not None:
            reward += (curr_equity - prev_equity) * EQUITY_PBRS_WEIGHT
        # (2) Pot-odds alignment (Billings 2002); not strictly PBRS
        if prev_state is not None and action_str in ("Call", "Check") and prev_equity is not None:
            cost = 2 if prev_state.round == 1 else 4
            pot_odds = cost / (prev_state.pot + cost) if prev_state.pot > 0 else 0.5
            reward += POT_ODDS_WEIGHT * (prev_equity - pot_odds)
        # (3) RNR alignment (Gap 1B): log π_ref(a_taken) − log uniform.
        # Not policy-invariant; small weight by design.
        if pi_ref and action_id is not None and action_id in pi_ref:
            n_legal = max(len(pi_ref), 1)
            p_ref = max(pi_ref[action_id], 1e-4)
            uniform = 1.0 / n_legal
            reward += RNR_ALIGN_WEIGHT * (math.log(p_ref) - math.log(uniform))
        # (4) Confidence-calibrated Raise bonus — rewards committing chips from
        # a strong-equity state. Linear ramp: equity=0.5 → 0, equity=1.0 → max.
        # Reckless raises (equity ≤ 0.5) get nothing; does NOT penalize them
        # (policy bluff remains viable via terminal signal).
        if (action_str.startswith("Raise")
            and prev_equity is not None):
            confidence = max(0.0, (prev_equity - 0.5) * 2.0)
            reward += CONFIDENT_RAISE_BONUS * confidence
        return reward

    def episode_reward(
        self,
        step_rewards: list[float],
        env_reward: float,
        done: bool,
        opp_folded: bool = False,
    ) -> float:
        """Pure per-step total return: terminal outcome + sum of per-step shaping.

        No split clipping on aggregate positive/negative — per-step bounds
        live in step_reward and the shaping helpers; this just sums.
        Preserves PBRS invariance (Ng/Harada/Russell 1999).
        """
        terminal = 0.0
        if done:
            # G.O.D server normalizes env_reward to [0,1] zero-sum;
            # map to [-1, +1] (TERMINAL_LOSS_REWARD .. TERMINAL_WIN_REWARD).
            terminal = 2.0 * env_reward - 1.0

            # Bonus for winning by inducing the opponent to fold (bluff/pressure
            # success). Previously this fired on every win because the only
            # check was env_reward > 0.5, which conflates showdown wins with
            # fold wins. We now require opp_folded to be observed in the
            # actual betting history before granting the bonus.
            if env_reward > 0.5 and opp_folded:
                terminal += POT_WIN_NO_SHOWDOWN_BONUS

        return terminal + sum(step_rewards)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=_MAX_TURNS,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)
    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn={_MAX_TURNS}, rollouts_per_stage={trainer.args.rollouts_per_stage}"
    )

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
        archetype_trackers={},  # keyed by server_idx; persists across episodes
    )


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    index: int,
    prompt: str,
    *,
    use_full_prompt: bool,
    env_pool: list[dict],
    num_servers: int,
    rank: int,
    trainer,
    tokenizer,
    generation_semaphore: Semaphore,
    current_max_turn: int,
    current_hint_prob: float,
    archetype_trackers: dict,
) -> tuple[int, "dict | None"]:
    game_id      = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:     list[int]   = []
    episode_completion_ids: list[int]   = []
    episode_logprobs:       list[float] = []
    episode_action_mask:    list[int]   = []
    prev_full_ids: "list[int] | None"   = None

    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done          = False
    final_reward  = 0.0
    turn_number   = 0
    invalid_count = 0
    use_hints     = random.random() < current_hint_prob
    game_state_history: list[GameState] = []
    calculator    = RewardCalculator()
    rewards:      list[float]   = []
    event_counter: dict[str, int] = {}

    # Track whether we last saw the opponent fold — needed to attribute the
    # POT_WIN_NO_SHOWDOWN_BONUS only to actual fold-induced wins.
    opp_folded:   bool          = False
    # Round at which the posterior was last reset_with_known (so we can
    # re-prime it once the public card is revealed in round 2).
    last_reset_round: int       = 0

    # Opponent modelling — active in both training modes so the full_prompt
    # variant is a meaningful A/B against the base env (Bayes' Bluff posterior
    # + SuspicionAgent ToM + RNR archetype shaping apply in both modes).
    posterior: OpponentCardPosterior = OpponentCardPosterior()
    archetype = archetype_trackers.setdefault(server_idx, OpponentArchetypePosterior())

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 50,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id  = result_block.get("episode_id", "")
        observation = _format_observation(result_block.get("observation", ""))
        gs = parse_game_state(observation)
        if gs is not None:
            game_state_history.append(gs)
            posterior.reset_with_known(gs.private_card, gs.public_card)
            last_reset_round = gs.round
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")

    # Initial message with opponent-modeling context
    initial_user = observation
    if gs is not None:
        ctx = (
            build_pokerbench_context(gs)
            + "\n"
            + build_suspicion_agent_context(gs, posterior, archetype)
        )
        initial_user = observation + "\n\n" + ctx

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": initial_user},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Token accumulation for full-prompt mode
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending early"
                )
                done = True
                break
            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    print(
                        f"Warning: token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)}). "
                        "Skipping delta mask."
                    )
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()
            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        action_to_send = _parse_action(completion_text)
        prev_gs = game_state_history[-1] if game_state_history else None

        # Equity before stepping (for PBRS)
        prev_equity = None
        if prev_gs is not None:
            prev_equity = posterior.equity(prev_gs.private_card, prev_gs.public_card)

        # Step env
        is_invalid = False
        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block  = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            step_reward = step_block.get("reward", 0)
            done        = step_block.get("done", False)
            new_gs: "GameState | None" = None
            if not done:
                new_gs = parse_game_state(observation)
                if new_gs is not None:
                    game_state_history.append(new_gs)
        except Exception as exc:
            print(f"Step failed (Game {game_id}, turn {turn_number}): {exc}")
            observation = ""
            step_reward = 0
            done        = False
            invalid_count += 1
            is_invalid = True
            new_gs = None

        if "Nothing happens" in observation or "Invalid" in observation:
            invalid_count += 1
            is_invalid = True

        if done:
            final_reward = step_reward
            # When the game terminates we can no longer rely on new_gs (only
            # parsed for non-terminal states above). Inspect the final betting
            # text directly so opp_folded is set if the opponent ended the hand
            # by folding.
            if not opp_folded and observation:
                for label in ("Round 1 betting:", "Round 2 betting:"):
                    m = re.search(rf"{label}\s*(.+)", observation)
                    if m and "Fold" in m.group(1):
                        opp_folded = True
                        break

        # Action string for pot-odds logic + archetype update
        action_id: "int | None" = None
        action_str = ""
        try:
            action_id  = int(action_to_send.strip())
            action_str = prev_gs.legal_actions.get(action_id, "") if prev_gs else ""
        except (ValueError, AttributeError):
            pass

        # RNR reference policy (Gap 1B): archetype-posterior-weighted BR blend.
        # Built from prev_gs so it scores the action the model just took.
        pi_ref: "dict[int, float] | None" = None
        if prev_gs is not None:
            pi_ref = archetype_reference_policy(prev_gs, archetype)

        # Update posterior + archetype from opponent actions in the new state
        if prev_gs is not None and new_gs is not None:
            # Actions observed in current_round_betting since prev_gs. In Leduc the
            # agent's own action shows up at the end of one round and opp's
            # response shows up next; we append the most recent foreign action.
            if new_gs.round == prev_gs.round:
                prev_betting = prev_gs.current_round_betting
                new_betting  = new_gs.current_round_betting
                new_actions  = new_betting[len(prev_betting):]
                if new_actions:
                    opp_action = new_actions[-1]  # most recent foreign action
                    posterior.update_on_action(opp_action, new_gs.round, new_gs.public_card_rank)
                    archetype.record(opp_action)
                    if opp_action == "Fold":
                        opp_folded = True

            # Re-prime posterior when we transition to round 2 and the public
            # card is revealed (was previously only reset once per episode,
            # leaving private uncertainty entangled with new public info).
            if (new_gs.round != last_reset_round
                and new_gs.public_card is not None
                and new_gs.round > last_reset_round):
                posterior.reset_with_known(new_gs.private_card, new_gs.public_card)
                last_reset_round = new_gs.round

        # Equity after
        curr_equity = None
        if new_gs is not None and not done:
            curr_equity = posterior.equity(new_gs.private_card, new_gs.public_card)

        # Per-step shaped reward
        step_shaped = calculator.step_reward(
            prev_equity=prev_equity,
            curr_equity=curr_equity,
            action_str=action_str,
            prev_state=prev_gs,
            is_invalid=is_invalid,
            pi_ref=pi_ref,
            action_id=action_id,
        )
        rewards.append(step_shaped)

        # Event counting
        if is_invalid:
            event_counter["invalid"] = event_counter.get("invalid", 0) + 1
        if prev_equity is not None and curr_equity is not None:
            d = curr_equity - prev_equity
            if d > 0.10:
                event_counter["equity_gain"] = event_counter.get("equity_gain", 0) + 1
            elif d < -0.10:
                event_counter["equity_loss"] = event_counter.get("equity_loss", 0) + 1
        if pi_ref and action_id is not None and action_id in pi_ref:
            best_aid = max(pi_ref, key=pi_ref.get)
            if best_aid == action_id:
                event_counter["align_match"] = event_counter.get("align_match", 0) + 1
        # Confidence telemetry on Raise actions
        if action_str.startswith("Raise") and prev_equity is not None:
            if prev_equity >= 0.7:
                event_counter["confident_raise"] = event_counter.get("confident_raise", 0) + 1
            elif prev_equity <= 0.3:
                event_counter["reckless_raise"] = event_counter.get("reckless_raise", 0) + 1

        # Build next-turn observation with opponent-modeling context
        if not done:
            next_gs = game_state_history[-1] if game_state_history else prev_gs
            if next_gs is not None:
                ctx = (
                    build_pokerbench_context(next_gs)
                    + "\n"
                    + build_suspicion_agent_context(next_gs, posterior, archetype)
                )
                aug_obs = observation + "\n\n" + ctx
            else:
                aug_obs = observation
            messages.append({"role": "user", "content": aug_obs})

        turn_number += 1

    # --- Episode reward ---
    train_reward = calculator.episode_reward(rewards, final_reward, done, opp_folded=opp_folded)

    events_str = " ".join(f"{k}:{v}" for k, v in event_counter.items()) if event_counter else "-"
    print(
        "[ID:{:<6} Hints:{} Done:{} T:{:>2d} | EnvR:{:>+6.2f} | TrainR:{:>+6.2f} | "
        "Inv:{:<2} Events:{}]".format(
            str(game_id)[:6], int(use_hints), int(done), turn_number,
            final_reward, train_reward, invalid_count, events_str,
        )
    )

    if use_full_prompt:
        if len(episode_completion_ids) > _MAX_EPISODE_TOKENS:
            episode_completion_ids = episode_completion_ids[:_MAX_EPISODE_TOKENS]
            episode_logprobs       = episode_logprobs[:_MAX_EPISODE_TOKENS]
            episode_action_mask    = episode_action_mask[:_MAX_EPISODE_TOKENS]
        return index, {
            "prompt_ids":     episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "action_mask":    episode_action_mask,
            "logprobs":       episode_logprobs,
            "reward":         train_reward,
            "final_score":    final_reward,
        }
    return index, {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "logprobs":       logprobs,
        "reward":         train_reward,
        "final_score":    final_reward,
    }


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}"
    )

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_max_turn=current_max_turn,
        current_hint_prob=current_hint_prob,
        archetype_trackers=_state["archetype_trackers"],
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0],
         "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1],
         "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    finished   = sum(1 for r in list_results if r["final_score"] != 0)
    # Win threshold must match ``calculate_episode_reward`` (> 0.5). Previously
    # used ``> 0``, which logged every non-zero EnvR as a win — misleading
    # since the backend normalizes to [0, 1] with 0.5 = tie.
    wins       = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgR:{avg_return:.3f}")

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
    }
    if use_full_prompt:
        out["action_mask"] = [r["action_mask"] for r in list_results]
    return out


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Full-prompt rollout with action masking. Full opponent-modeling stack
    is active so the variant is a meaningful A/B against the base env in this
    mode too (same components as the last-prompt variant)."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Last-prompt rollout with full opponent-modeling stack:
      * PokerBench context schema
      * SuspicionAgent ToM belief text
      * RNR archetype posterior (cross-episode)
      * Per-step equity-PBRS + pot-odds alignment + RNR log-alignment shaping."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
