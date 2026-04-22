"""
leduc_poker_opponent_modeling.py
================================
Leduc Poker episode runner **with Bayesian opponent modelling**.

Extends leduc_poker_env.py by adding:
  - OpponentRangeModel  : Bayesian posterior over the opponent's hole card
                          (J/Q/K), updated after each betting action they take.
  - OpponentTendencyTracker: tracks fold/call/raise ratios to classify the
                              opponent as Tight-Passive, Loose-Aggressive, etc.
  - Augmented observation: appends a concise "Opponent Read" block to every
                           non-terminal observation so the LLM sees live
                           inference about the opponent before acting.

Architecture mirrors gin_rummy_opponent_modeling.py:
  - Opponent models are instantiated ONLY in last-prompt mode (use_full_prompt=False).
  - full-prompt mode runs identically to leduc_poker_env.py.
  - Public API is identical: rollout_full_prompt_and_completion_parallelized_curriculum
                              rollout_last_prompt_and_completion_parallelized_curriculum
"""

import functools
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
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants (inherited from leduc_poker_env.py)
# ---------------------------------------------------------------------------

_SELECTED_GAME = "leduc_poker"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 4096
_TIMEOUT = 2400
_INVALID_PENALTY = -0.1
_MAX_TURNS = 10

_BASE_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits × (num_players + 1) ranks. For 2 players: 6 cards (J♠ J♥ Q♠ Q♥ K♠ K♥).\n\n"
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
    '- For action "0 -> roll": respond "0"\n'
    '- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "Round 1:\n"
    "- Hold K or Q → call a raise; raise first if unchallenged.\n"
    "- Hold J → fold against a raise; check if unchallenged.\n\n"
    "Round 2 (public card revealed):\n"
    "- You have a PAIR → raise; never fold.\n"
    "- You have K (no pair) → raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is K → raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is J → check; fold if opponent raises.\n"
    "- You have J (no pair) → check; fold if opponent raises.\n"
)


# ---------------------------------------------------------------------------
# Game state dataclass and parser (identical to leduc_poker_env.py)
# ---------------------------------------------------------------------------

_CARD_RANK: dict[str, int] = {"J": 1, "Q": 2, "K": 3}


@dataclass
class GameState:
    """Structured representation of a Leduc Poker observation."""
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
    legal_actions:     dict[int, str]

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

    @property
    def opp_raised_this_round(self) -> bool:
        return "Raise" in self.current_round_betting

    @property
    def can_raise(self) -> bool:
        return 2 in self.legal_actions

    @property
    def can_fold(self) -> bool:
        return 0 in self.legal_actions

    @property
    def hand_strength(self) -> int:
        if self.has_pair:
            return 4
        return self.private_card_rank

    @property
    def is_strong(self) -> bool:
        return self.hand_strength >= 3

    @property
    def is_weak(self) -> bool:
        return self.hand_strength == 1


def parse_game_state(obs: str) -> "GameState | None":
    """Parse a formatted Leduc Poker observation into a GameState."""
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

    pot      = int(_find(r"Pot size:\s*(\d+)", "0"))
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


# ---------------------------------------------------------------------------
# *** OPPONENT MODELING LAYER ***
# ---------------------------------------------------------------------------

class OpponentRangeModel:
    """
    Bayesian posterior over the opponent's private card (J / Q / K).

    Prior
    -----
    Since the deck has 2 copies of each rank and we know our own card,
    the prior is P(J) = P(Q) = P(K) = 1/3, adjusted if we hold one copy.

    Update rules (heuristic Bayesian)
    -----------------------------------
    Leduc poker is solved → optimal play is well-characterised:
      R1 Raise  → likely K (0.5×) or Q (0.3×), unlikely J (0.2×)
      R1 Check  → likely J (0.4×) or Q (0.35×), less likely K (0.25×)
      R1 Fold   → impossible (would mean folding before round 2)
      R2 Raise  → very likely Pair (0.6×) or K (0.3×); unlikely J/Q no-pair
      R2 Check  → likely J-no-pair (0.45×) or Q-no-pair (0.3×)
      R2 Fold   → very likely J-no-pair (0.55×) or Q-pubK (0.3×)

    After each update the posterior is renormalized.
    """

    CARDS = ["J", "Q", "K"]

    # Likelihood multipliers for each (round, action) combination
    # Format: {(round, action): {card: multiplier}}
    _LIKELIHOODS: dict[tuple, dict[str, float]] = {
        (1, "Raise"): {"J": 0.20, "Q": 0.35, "K": 0.55},
        (1, "Call"):  {"J": 0.35, "Q": 0.40, "K": 0.25},
        (1, "Check"): {"J": 0.40, "Q": 0.35, "K": 0.25},
        (1, "Fold"):  {"J": 0.70, "Q": 0.25, "K": 0.05},
        (2, "Raise"): {"J": 0.10, "Q": 0.25, "K": 0.30},   # pair gets extra below
        (2, "Call"):  {"J": 0.25, "Q": 0.40, "K": 0.35},
        (2, "Check"): {"J": 0.50, "Q": 0.35, "K": 0.15},
        (2, "Fold"):  {"J": 0.60, "Q": 0.30, "K": 0.10},
    }

    def __init__(self, our_card_rank: int) -> None:
        # Prior: uniform, but if we hold a copy of a rank there's one fewer in deck
        prior = {c: 1.0 for c in self.CARDS}
        # Slight adjustment: reduce rank matching our own card (one less in deck)
        our_rank_str = {1: "J", 2: "Q", 3: "K"}.get(our_card_rank)
        if our_rank_str:
            prior[our_rank_str] = 0.5   # one copy taken by us
        total = sum(prior.values())
        self._posterior: dict[str, float] = {c: v / total for c, v in prior.items()}
        self.has_pair_evidence: bool = False

    def update(self, round_: int, action_str: str, public_card_rank: "int | None" = None) -> None:
        """Update posterior after observing the opponent's action in `round_`."""
        key = (round_, action_str)
        likelihoods = self._LIKELIHOODS.get(key)
        if likelihoods is None:
            return

        # Pair bonus for R2 Raise: if opp raises in R2 there's a strong chance
        # they have a pair (private == public).  We can't know for sure, but
        # we can note "pair evidence" and adjust K upward since K no-pair also raises.
        if round_ == 2 and action_str == "Raise" and public_card_rank is not None:
            pub_str = {1: "J", 2: "Q", 3: "K"}.get(public_card_rank, "")
            if pub_str:
                # Opponent might have pair with pub card's rank
                pair_boost = 0.4
                likelihoods = dict(likelihoods)  # copy to avoid mutating class var
                likelihoods[pub_str] = min(likelihoods[pub_str] + pair_boost, 1.0)

        # Bayesian update: posterior ∝ prior × likelihood
        unnorm = {c: self._posterior[c] * likelihoods.get(c, 1.0) for c in self.CARDS}
        total = sum(unnorm.values())
        if total > 0:
            self._posterior = {c: v / total for c, v in unnorm.items()}

    def most_likely_card(self) -> str:
        return max(self._posterior, key=lambda c: self._posterior[c])

    def probability(self, card: str) -> float:
        return self._posterior.get(card, 0.0)

    def summary(self) -> str:
        probs = sorted(self._posterior.items(), key=lambda x: -x[1])
        prob_str = "  ".join(f"{c}:{p:.0%}" for c, p in probs)
        ml = probs[0][0]
        conf = probs[0][1]
        lines = [f"[RangeModel] Opp card estimate: {prob_str}"]
        if conf >= 0.55:
            lines.append(f"[RangeModel] Most likely: {ml} (confidence {conf:.0%})")
            if ml == "K":
                lines.append("[RangeModel] Caution: opponent likely has K — be careful raising/calling")
            elif ml == "J":
                lines.append("[RangeModel] Opportunity: opponent likely has J — raise/stay in")
        return "\n".join(lines)


class OpponentTendencyTracker:
    """
    Tracks fold / call / raise counts for the opponent to classify their
    playing style: Tight-Passive, Loose-Aggressive, Bluff-Heavy, etc.

    Useful for adjusting our aggression level.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {"Fold": 0, "Call": 0, "Check": 0, "Raise": 0}
        self.total: int = 0

    def record(self, action_str: str) -> None:
        if action_str in self.counts:
            self.counts[action_str] += 1
            self.total += 1

    def raise_rate(self) -> float:
        return self.counts["Raise"] / self.total if self.total > 0 else 0.0

    def fold_rate(self) -> float:
        return self.counts["Fold"] / self.total if self.total > 0 else 0.0

    def style(self) -> str:
        if self.total < 2:
            return "Unknown"
        rr = self.raise_rate()
        fr = self.fold_rate()
        if rr >= 0.5:
            return "Loose-Aggressive (LAG)"
        elif rr >= 0.25 and fr <= 0.15:
            return "Loose-Passive (LP)"
        elif rr <= 0.15 and fr >= 0.30:
            return "Tight-Passive (TP)"
        elif rr >= 0.25 and fr >= 0.25:
            return "Tight-Aggressive (TAG)"
        return "Balanced"

    def summary(self) -> str:
        if self.total < 2:
            return ""
        style = self.style()
        rr    = self.raise_rate()
        fr    = self.fold_rate()
        lines = [
            f"[TendencyTracker] Opp style: {style}",
            f"[TendencyTracker] Raise rate: {rr:.0%}  Fold rate: {fr:.0%}",
        ]
        if "LAG" in style:
            lines.append("[TendencyTracker] Tip: opp raises a lot — weight toward calling/folding with weak hand")
        elif "TP" in style:
            lines.append("[TendencyTracker] Tip: opp is passive — bluff/raise more freely")
        return "\n".join(lines)


def _build_opponent_read(
    range_model: "OpponentRangeModel | None",
    tendency: "OpponentTendencyTracker | None",
) -> str:
    """Combine all opponent model summaries into a clean context block."""
    parts = []
    if range_model is not None:
        parts.append(range_model.summary())
    if tendency is not None:
        t = tendency.summary()
        if t:
            parts.append(t)
    return "\n".join(parts)


def _augment_observation(obs: str, opponent_read: str) -> str:
    if not opponent_read:
        return obs
    return obs + "\n\n--- Opponent Read ---\n" + opponent_read


# ---------------------------------------------------------------------------
# Reward calculator (identical to leduc_poker_env.py)
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Shaped reward calculator for Leduc Poker training."""

    SIGNALS = {
        "fold_pair":         -2.0,
        "fold_k":            -1.5,
        "fold_kq_r1_raise":  -1.5,
        "fold_q_pubk_raise": -0.5,
        "fold_j_r1_raise":   +0.3,
        "fold_j_r2_raise":   +0.2,
        "fold_q_pubj_raise": +0.2,
        "raise_pair_r2":     +0.3,
        "raise_k_r2":        +0.2,
        "call_kq_r1_raise":  +0.2,
    }

    def __init__(self, terminal_weight: float = 1.0, gamma: float = 0.9):
        self.terminal_weight = terminal_weight
        self.gamma = gamma

    def calculate_step_reward(self, gs: "GameState | None", action_str: str, env_reward: float) -> float:
        reward = 0.0

        if gs is not None:
            pub = gs.public_card_rank or 0

            if action_str == "Fold":
                if gs.has_pair:
                    reward += self.SIGNALS["fold_pair"]
                elif gs.private_card_rank == 3:
                    reward += self.SIGNALS["fold_k"]
                elif gs.round == 1 and gs.opp_last_action == "Raise":
                    if gs.private_card_rank >= 2:
                        reward += self.SIGNALS["fold_kq_r1_raise"]
                    else:
                        reward += self.SIGNALS["fold_j_r1_raise"]
                elif gs.round == 2 and gs.opp_last_action == "Raise":
                    if gs.private_card_rank == 1:
                        reward += self.SIGNALS["fold_j_r2_raise"]
                    elif gs.private_card_rank == 2 and pub == 1:
                        reward += self.SIGNALS["fold_q_pubj_raise"]
                    elif gs.private_card_rank == 2 and pub == 3:
                        reward += self.SIGNALS["fold_q_pubk_raise"]

            elif action_str == "Raise":
                if gs.round == 2 and gs.has_pair:
                    reward += self.SIGNALS["raise_pair_r2"]
                elif gs.round == 2 and gs.private_card_rank == 3 and not gs.has_pair:
                    reward += self.SIGNALS["raise_k_r2"]

            elif action_str in ("Call", "Check"):
                if gs.round == 1 and gs.opp_last_action == "Raise" and gs.private_card_rank >= 2:
                    reward += self.SIGNALS["call_kq_r1_raise"]

        if env_reward != 0.0:
            reward += env_reward * self.terminal_weight

        return reward

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        if not rewards:
            return 0.0
        T = len(rewards)
        return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=_MAX_TURNS,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    """Set up server pool and curriculum once per process (no-op afterwards)."""
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
    )


# ---------------------------------------------------------------------------
# Observation formatter (identical to leduc_poker_env.py)
# ---------------------------------------------------------------------------

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


def _parse_action(completion_text: str, legal_map: "dict[int, str] | None" = None) -> str:
    """
    Extract action ID from model output.

    If ``legal_map`` is provided and the extracted token does not map to a
    legal action, returns an empty string so the caller can use the fallback.
    """
    action = completion_text.strip()
    if action.endswith("</s>"):
        action = action[:-4].strip()
    if "Action:" in action:
        action = action.split("Action:")[-1].strip()
    # Keep only the first token (model may output "2 Raise" instead of "2")
    action = action.split()[0] if action.split() else action
    if legal_map is not None:
        try:
            if int(action) not in legal_map:
                return ""
        except (ValueError, TypeError):
            return ""
    return action


def _select_fallback_action(gs: "GameState | None") -> str:
    """
    Nash-inspired fallback when the model produces an invalid action.

    Decision hierarchy (Leduc solved strategy):
      - Pair or K  → Raise > Call
      - J vs raise → Fold > Call
      - Otherwise  → Call > Fold
    """
    if gs is None:
        return "1"
    la = gs.legal_actions
    if not la:
        return "1"

    def _pick(*preferred: int) -> str:
        for a in preferred:
            if a in la:
                return str(a)
        return str(next(iter(la)))

    if gs.has_pair or gs.private_card_rank == 3:
        return _pick(2, 1)  # Raise > Call
    if gs.private_card_rank == 1 and gs.opp_raised_this_round:
        return _pick(0, 1)  # Fold > Call
    return _pick(1, 0)      # Call > Fold


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
    current_hint_prob: float,
) -> tuple[int, "dict | None"]:
    game_id = int(prompt)
    server_idx = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt state (overwritten each turn)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done                 = False
    final_reward         = 0.0
    episode_reward       = 0.0
    turn_number          = 0
    invalid_count        = 0
    consecutive_invalids = 0
    use_hints            = random.random() < current_hint_prob
    game_state_history: list[GameState] = []
    calculator           = RewardCalculator()
    rewards:             list[float] = []

    # Opponent models — only active in last-prompt mode
    range_model: "OpponentRangeModel | None"        = None
    tendency:    "OpponentTendencyTracker | None"   = None

    # Reset environment
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
            # Initialize opponent models once we know our card
            if not use_full_prompt:
                range_model = OpponentRangeModel(gs.private_card_rank)
                tendency    = OpponentTendencyTracker()
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _BASE_SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": observation},
    ]

    while not done and turn_number < _MAX_TURNS:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation (full-prompt mode) ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
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
                        "Skipping delta mask for this turn."
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

        # --- Parse and send action ---
        prev_gs   = game_state_history[-1] if game_state_history else None
        legal_map = prev_gs.legal_actions if prev_gs else None
        action_to_send = _parse_action(completion_text, legal_map)

        # Fallback: Nash-based selection when model output is invalid
        if not action_to_send:
            action_to_send = _select_fallback_action(prev_gs)
            consecutive_invalids += 1
            invalid_count += 1
            # Escalating penalty: -0.10, -0.15, -0.20, ...
            penalty = _INVALID_PENALTY + 0.05 * max(0, consecutive_invalids - 1)
            episode_reward += penalty
        else:
            consecutive_invalids = 0  # reset on valid action

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
            if not done:
                gs = parse_game_state(observation)
                if gs is not None:
                    game_state_history.append(gs)
        except Exception as exc:
            print(f"Step failed (Game {game_id}, turn {turn_number}): {exc}")
            observation = ""
            step_reward = 0
            done        = False
            invalid_count += 1
            episode_reward += _INVALID_PENALTY

        if "Nothing happens" in observation or "Invalid" in observation:
            invalid_count += 1
            episode_reward += _INVALID_PENALTY

        if done:
            final_reward = step_reward

        # --- Update opponent models (last-prompt only) ---
        if not use_full_prompt and not done and prev_gs is not None:
            curr_gs = game_state_history[-1] if game_state_history else None
            if curr_gs is not None:
                # Detect opponent's last action by comparing betting histories
                prev_betting = prev_gs.r1_betting if curr_gs.round == 1 else prev_gs.r2_betting
                curr_betting = curr_gs.r1_betting if curr_gs.round == 1 else curr_gs.r2_betting
                if len(curr_betting) > len(prev_betting):
                    opp_action = curr_betting[-1]
                    if range_model is not None:
                        range_model.update(
                            curr_gs.round, opp_action,
                            public_card_rank=curr_gs.public_card_rank
                        )
                    if tendency is not None:
                        tendency.record(opp_action)

        # Augment observation with opponent read (last-prompt only, non-terminal)
        if not use_full_prompt and not done:
            curr_gs2 = game_state_history[-1] if game_state_history else None
            opp_read = _build_opponent_read(range_model, tendency)
            observation = _augment_observation(observation, opp_read)

        try:
            action_str = prev_gs.legal_actions.get(int(action_to_send.strip()), "") if prev_gs else ""
        except (ValueError, AttributeError):
            action_str = ""
        step_shaped = calculator.calculate_step_reward(prev_gs, action_str, step_reward if done else 0.0)
        rewards.append(step_shaped)

        messages.append({"role": "user", "content": observation})
        turn_number += 1

    train_reward = calculator.calculate_discounted_return(rewards) + episode_reward
    print(
        "[ID:{:<6} Done:{} T:{:>2d} | Hints:{:<2} | EnvR:{:>6.2f} | TrainR:{:>6.2f} | Inv:{:<2}]".format(
            str(game_id)[:6], int(done), turn_number, int(use_hints), final_reward, train_reward, invalid_count,
        )
    )

    # --- Build result ---
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
    else:
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
    """Common dispatch + aggregation logic for both rollout variants."""
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: hint_prob={current_hint_prob:.2f}")

    run = functools.partial(
        _run_episode,
        use_full_prompt=use_full_prompt,
        env_pool=_state["env_pool"],
        num_servers=_state["num_servers"],
        rank=_state["rank"],
        trainer=trainer,
        tokenizer=trainer.processing_class,
        generation_semaphore=_state["generation_semaphore"],
        current_hint_prob=current_hint_prob,
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    finished   = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}")

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
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _MAX_TURNS,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs.

    Enables full Bayesian opponent modelling (OpponentRangeModel +
    OpponentTendencyTracker).  The LLM sees an 'Opponent Read' block appended
    to each observation.
    """
    return _dispatch(prompts, trainer, use_full_prompt=False)
