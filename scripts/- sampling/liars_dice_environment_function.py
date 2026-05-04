import json
import math
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock, Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

GAME_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999), "goof_spiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}

SELECTED_GAME = "liars_dice"
REQUEST_TIMEOUT_SECONDS = 2400
INIT_TIMEOUT_SECONDS = 300
MAX_EPISODE_TOKENS = 16384
MAX_PROMPT_LEN = 16384 - 512

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations": 225,
    "mcts_num_rollouts": 1,
}

# --- Per-step scoring constants (winner-inspired) ---
SCORE_TEMPERATURE = 0.5        # exponential sharpening for probability -> score
GAMMA = 0.9                    # discount factor for return calculation
TERMINAL_WEIGHT = 10.0         # multiplier for env terminal reward
BLUFF_PROB_THRESHOLD = 0.35    # bid prob below this -> classified as bluff
RISKY_LIAR_PROB_MIN = 0.35     # liar call prob lower bound for "risky"
RISKY_LIAR_PROB_MAX = 0.60     # liar call prob upper bound for "risky"
BLUFF_WIN_BONUS = 0.5          # bonus per bluff in winning episode
RISKY_LIAR_WIN_BONUS = 0.5     # bonus per risky liar call in winning episode
RISKY_BONUS_MAX_COUNT = 2      # cap risky move bonuses
SHUFFLE_PROB = 0.5             # probability of shuffling action list
INVALID_STEP_PENALTY = -1.0    # hard penalty for invalid actions (per step)

# --- Legacy constants kept for compatibility ---
NOOP_PENALTY = 0.03
TRUNCATION_PENALTY = 0.20
INVALID_ACTION_PENALTY = 0.10  # kept for context injection logic only
CONSECUTIVE_INVALID_ESCALATION = 0.05
BLUFF_ZONE_THRESHOLD = 0.40
TOM_MIXED_STRATEGY_BONUS = 0.02
RAG_ALIGNMENT_BONUS = 0.02
CALL_QUALITY_BONUS = 0.08
CALL_QUALITY_PENALTY = 0.06
PASS_MISSED_CHALLENGE_PENALTY = 0.06
BID_PLAUSIBILITY_BONUS = 0.04
BID_PLAUSIBILITY_PENALTY = 0.04
BID_AGGRESSIVE_PENALTY = 0.02
CALL_TIMING_BONUS = 0.03
FIRST_BIDDER_CONSERVATIVE_BONUS = 0.03
TERMINAL_REWARD_CLIP = 1.00    # only used in _extract_terminal_reward

def _bid_probability(bid_qty: int, bid_face: int, own_dice: list[int], total_dice: int, wild_six: bool = True) -> float:
    """P(bid is true) given observable game state. Uses scipy.binom if available, else pure math."""
    if bid_face == 6:
        our_count = sum(1 for d in own_dice if d == 6)
        p_hit = 1 / 6
    else:
        our_count = sum(1 for d in own_dice if d == bid_face or (wild_six and d == 6))
        p_hit = 2 / 6

    still_needed = bid_qty - our_count
    if still_needed <= 0:
        return 1.0

    n_hidden = total_dice - len(own_dice)
    if n_hidden <= 0:
        return 0.0

    try:
        from scipy.stats import binom
        return float(1.0 - binom.cdf(still_needed - 1, n=n_hidden, p=p_hit))
    except ImportError:
        return _binomial_tail_probability(n_hidden, p_hit, still_needed)


def _probability_to_score(prob: float) -> float:
    """Sharpened probability -> score using exponential temperature scaling."""
    a = SCORE_TEMPERATURE
    return (math.exp(prob / a) - 1.0) / (math.exp(1.0 / a) - 1.0)


def _shuffle_observation_actions(obs: str, legal_action_map: dict[str, str]) -> str:
    """Randomly shuffle the displayed action order in the observation (50% chance)."""
    if random.random() >= SHUFFLE_PROB or not legal_action_map:
        return obs
    items = list(legal_action_map.items())
    random.shuffle(items)
    action_block = "Legal Actions:\n" + "\n".join(f"{aid} -> {lbl}" for aid, lbl in items)
    obs = re.sub(
        r"Legal Actions:\n(?:[ \t]*\d+[ \t]*->[ \t]*\S.*(?:\n|$))+",
        action_block + "\n",
        obs,
    )
    return obs


class DiceRAG:
    """Lightweight knowledge base complementing CFR/Bayesian for Liar's Dice.

    Fills blind spots:
    - Early game: Bayesian has no opponent bid data
    - New dice patterns: CFR table empty
    - MCTS counter: Exploit MCTS bidding tendencies
    Memory: ~2KB. Latency: <0.1ms.
    """

    def retrieve(self, state: dict, max_entries: int = 2) -> tuple[str, str]:
        """Return (advice_text, recommended_action: 'bid'|'call'|'')."""
        matches = []
        own_dice = state.get("own_dice") or []
        total_dice = int(state.get("total_dice") or 0)
        current_bid = state.get("current_bid")

        if not own_dice or total_dice <= 0:
            return "", ""

        wild_six = bool(state.get("wild_six_enabled"))

        # --- Bluff zone detection ---
        if current_bid:
            qty, face = current_bid
            own_support = sum(1 for d in own_dice if d == face or (wild_six and d == 6 and face != 6))
            need_from_opp = max(qty - own_support, 0)
            opp_dice = max(total_dice - len(own_dice), 0)
            ratio = qty / max(total_dice, 1)

            if ratio >= 0.55:
                matches.append((5, f"[RAG] Bid {qty}x face-{face} claims {ratio:.0%} of all dice. EXTREME bluff zone. Call Liar!", "call"))
            elif ratio >= 0.42 and own_support == 0:
                matches.append((4, f"[RAG] Bid needs {need_from_opp} from {opp_dice} opp dice, you have 0 support. Likely false. Call.", "call"))
            elif need_from_opp == 0:
                matches.append((3, f"[RAG] You already have {own_support} support for bid {qty}x{face}. Safe to raise.", "bid"))

        # --- Strong hand ---
        face_counts = Counter(own_dice)
        best_face, best_count = face_counts.most_common(1)[0] if face_counts else (1, 0)
        if best_count >= 3:
            matches.append((4, f"[RAG] Strong hand: {best_count}x face-{best_face}. Bid this face confidently.", "bid"))
        elif best_count == 1 and not current_bid:
            matches.append((2, "[RAG] Weak hand (all unique). Bid conservatively or prepare to call.", "bid"))

        # --- Endgame ---
        if total_dice <= 3:
            matches.append((4, "[RAG] ENDGAME: Few dice remain. Challenge aggressively - bluffs are easy to spot.", "call"))

        if not matches:
            return "", ""
        matches.sort(key=lambda x: -x[0])
        lines = [m[1] for m in matches[:max_entries]]
        return "\n".join(lines), matches[0][2]

RULESET_CLASSIC = "classic"
RULESET_LIARS_DIE = "liars_die"

STRATEGY_TIPS_CLASSIC = """
STRATEGY TIPS:
- Keep bids minimally stronger than current bid when uncertain (+1 quantity or +1 face).
- Use your own dice + wild 6s to estimate plausible total counts.
- Expected count for any face ≈ total_dice/6 (or total_dice/3 with wild-6).
- Bids claiming >40% of total dice are in the BLUFF ZONE — challenging is reasonable.
- Bids claiming >50% of total dice are almost certainly bluffs — call Liar immediately.
- If P(bid is true) < 30%, calling Liar is +EV. If P(bid is true) > 70%, raising is safer.
- MIXED STRATEGY REQUIRED: Do not be predictable. Occasionally bluff even with moderate hands
  to prevent your opponent from reading your bidding pattern.
- After opponent makes large jump (>=4 bid ranks): high probability of bluff — consider calling.
- Opponent bidding conservatively (small jumps): likely has a strong hand — be cautious.
- Early game (many dice): bluffs harder to detect, moderate bids safer.
- Late game (<=4 total dice): bids highly transparent — challenge freely when bid exceeds expected.
- If you bid FIRST this round, you set the tone; bid based on your own dice strength.
- If you bid SECOND this round, use opponent's bid as information to update your estimate.

LEARN FROM CONTRAST:
✓ GOOD: Opponent bid 5×3 with 10 total dice (50%) → Called Liar → Won (bid was implausible)
✗ BAD:  Opponent bid 3×2 with 10 total dice (30%) → Called Liar → Lost (bid was plausible)
✓ GOOD: Had 3 fives, bid 4×5 (conservative +1) → Opponent raised → Safe position
✗ BAD:  Had 1 five, bid 5×5 (extreme bluff) → Opponent called Liar → Lost badly
✓ GOOD: Late game 4 dice, opponent bid 3×4 (75%) → Called Liar → Won
✗ BAD:  Early game 10 dice, opponent bid 4×2 (40%) → Called Liar → Lost (plausible with wild-6)

PAST EXPERIENCE:
- Bid conservatively (minimal increment) when uncertain → saved dice more often
- Called Liar on bids > 50% total dice → won 70%+ of those calls
- Bluffed with large jump when holding no support → opponent called, lost
- Opponent bid passively for 3 turns then big jump → was bluffing, calling won

META-AWARENESS (Theory of Mind):
- If you've been bidding conservatively, opponent thinks you're STRONG → a bluff bid NOW is more credible.
- If you've been calling Liar frequently, opponent thinks you're SUSPICIOUS → they may bid more honestly.
- If you raised aggressively last turn, opponent may think you're BLUFFING → they're more likely to call.
- Consider what your bidding pattern SIGNALS to the opponent, not just what you hold.
"""

STRATEGY_TIPS_LIARS_DIE = """
STRATEGY TIPS (Liar's die):
- You only know your own current roll; each claim names a die rank (face value).
- After a claim, the opponent may Doubt (the roll is revealed) or Accept (they reroll and must claim strictly higher).
- On Doubt, the claimant wins if their hidden roll is at least the claimed rank; otherwise the doubter wins.
- Low rolls: bluffing upward is often necessary—telling the truth with a very low roll loses often once play continues.
- After a high prior claim, Doubt is attractive—the claimant may be bluffing out of necessity.
- Sometimes Accepting preserves a chance to roll high and escalate, especially early.
"""

STRATEGY_TIPS = STRATEGY_TIPS_CLASSIC

REASONING_TAG_PAIRS = [
    ("think", "think"), ("thinking", "thinking"), ("reasoning", "reasoning"),
    ("thought", "thought"), ("reflection", "reflection"),
]

_ROLLOUT_STATE: dict = {}

class CFRTable:
    CFR_SHAPING_SCALE = 0.02

    def __init__(self) -> None:
        self._lock = Lock()
        self._regret: dict[tuple[str, int], float] = {}
        self._strategy_sum: dict[tuple[str, int], float] = {}
        self._episode_count: int = 0

    @staticmethod
    def _dice_pattern(own_dice: list[int]) -> str:
        return str(tuple(sorted(own_dice))) if own_dice else "()"

    def _get_regrets(self, info_key: str) -> dict[int, float]:
        result = {}
        with self._lock:
            for (ik, br), r in self._regret.items():
                if ik == info_key:
                    result[br] = r
        return result

    def get_strategy(self, own_dice: list[int], legal_bid_ranks: list[int]) -> dict[int, float]:
        if not own_dice or not legal_bid_ranks:
            n = max(len(legal_bid_ranks), 1)
            return {br: 1.0 / n for br in legal_bid_ranks}
        info_key = self._dice_pattern(own_dice)
        regrets = self._get_regrets(info_key)
        positive = {br: max(0.0, regrets.get(br, 0.0)) for br in legal_bid_ranks}
        total = sum(positive.values())
        if total <= 0:
            n = len(legal_bid_ranks)
            return {br: 1.0 / n for br in legal_bid_ranks}
        return {br: v / total for br, v in positive.items()}

    def cfr_shaping(self, own_dice: list[int], chosen_bid_rank: int, legal_bid_ranks: list[int]) -> float:
        strategy = self.get_strategy(own_dice, legal_bid_ranks)
        chosen_prob = strategy.get(chosen_bid_rank, 0.0)
        uniform_prob = 1.0 / max(len(legal_bid_ranks), 1)
        deviation = chosen_prob - uniform_prob
        max_deviation = max(1.0 - uniform_prob, uniform_prob)
        if max_deviation > 0:
            return self.CFR_SHAPING_SCALE * (deviation / max_deviation)
        return 0.0

    def update(self, own_dice: list[int], bid_rank_played: int, legal_bid_ranks: list[int], episode_reward: float) -> None:
        if not own_dice or not legal_bid_ranks:
            return
        info_key = self._dice_pattern(own_dice)
        strategy = self.get_strategy(own_dice, legal_bid_ranks)
        with self._lock:
            self._episode_count += 1
            for br in legal_bid_ranks:
                if br == bid_rank_played:
                    self._regret[(info_key, br)] = self._regret.get((info_key, br), 0.0) - episode_reward * 0.1
                else:
                    counterfactual = -episode_reward * 0.05
                    self._regret[(info_key, br)] = self._regret.get((info_key, br), 0.0) + counterfactual
                self._strategy_sum[(info_key, br)] = self._strategy_sum.get((info_key, br), 0.0) + strategy.get(br, 0.0)

    def stats(self) -> str:
        with self._lock:
            return f"CFRTable: {len(self._regret)} entries, {self._episode_count} episodes"

class BayesianOpponentInference:
    BLUFF_PROB = 0.15

    def __init__(self, n_dice: int = 5, wild_six: bool = True) -> None:
        self.n_dice = n_dice
        self.wild_six = wild_six
        self._all_rolls: list[tuple[int, ...]] = self._enumerate_rolls(n_dice)
        n = len(self._all_rolls)
        self._log_probs: list[float] = [-math.log(n)] * n
        self._roll_index: dict[tuple[int, ...], int] = {r: i for i, r in enumerate(self._all_rolls)}

    @staticmethod
    def _enumerate_rolls(n_dice: int) -> list[tuple[int, ...]]:
        if n_dice <= 0:
            return [()]
        result: list[tuple[int, ...]] = [()]
        for _ in range(n_dice):
            result = [r + (v,) for r in result for v in range(1, 7)]
        return result

    def _face_count(self, roll: tuple[int, ...], face: int) -> int:
        if self.wild_six and face != 6:
            return sum(1 for d in roll if d == face or d == 6)
        return sum(1 for d in roll if d == face)

    def update(self, bid: tuple[int, int], own_support: int) -> None:
        qty, face = bid
        need_from_opp = max(qty - own_support, 0)
        log_bluff = math.log(self.BLUFF_PROB)
        log_support = math.log(1.0 - self.BLUFF_PROB)
        try:
            import numpy as np
            if not hasattr(self, '_np_rolls') or self._np_rolls is None:
                self._np_rolls = np.array(self._all_rolls, dtype=np.int8)
            rolls = self._np_rolls
            if self.wild_six and face != 6:
                opp_support = np.sum((rolls == face) | (rolls == 6), axis=1)
            else:
                opp_support = np.sum(rolls == face, axis=1)
            log_likelihoods = np.where(opp_support >= need_from_opp, log_support, log_bluff)
            log_probs_arr = np.array(self._log_probs) + log_likelihoods
            max_lp = log_probs_arr.max()
            log_sum = max_lp + math.log(np.exp(log_probs_arr - max_lp).sum())
            self._log_probs = (log_probs_arr - log_sum).tolist()
        except ImportError:
            new_log_probs = []
            for i, roll in enumerate(self._all_rolls):
                opp_support = self._face_count(roll, face)
                log_likelihood = log_support if opp_support >= need_from_opp else log_bluff
                new_log_probs.append(self._log_probs[i] + log_likelihood)
            max_lp = max(new_log_probs)
            log_sum = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in new_log_probs))
            self._log_probs = [lp - log_sum for lp in new_log_probs]

    def expected_support(self, face: int) -> float:
        total = 0.0
        for i, roll in enumerate(self._all_rolls):
            prob = math.exp(self._log_probs[i])
            total += prob * self._face_count(roll, face)
        return total

    def bid_posterior_prob(self, bid: tuple[int, int], own_support: int) -> float:
        qty, face = bid
        need = max(qty - own_support, 0)
        return sum(math.exp(self._log_probs[i]) for i, roll in enumerate(self._all_rolls) if self._face_count(roll, face) >= need)

    def call_shaping(self, bid: tuple[int, int], own_support: int, call_quality_bonus: float, call_quality_penalty: float, episode_won: bool) -> float:
        p_true = self.bid_posterior_prob(bid, own_support)
        if episode_won:
            return call_quality_bonus * (1.0 + (1.0 - p_true))
        else:
            return -call_quality_penalty * (1.0 + p_true)

    def context_summary(self, face: int) -> str:
        exp = self.expected_support(face)
        return f"[Bayesian] Expected opp dice showing {face}: ~{exp:.1f}"

    def full_context_summary(self, own_dice: list[int]) -> str:
        if not own_dice:
            return ""
        lines = []
        face_counts = {}
        for d in own_dice:
            face_counts[d] = face_counts.get(d, 0) + 1
        for face in sorted(set(range(1, 7))):
            exp = self.expected_support(face)
            own = face_counts.get(face, 0)
            lines.append(f"  Face {face}: you have {own}, opp expected ~{exp:.1f}, total est ~{own + exp:.1f}")
        return "[Bayesian Opponent Inference]\n" + "\n".join(lines)

    def reset(self, n_dice: int | None = None, wild_six: bool | None = None) -> None:
        if n_dice is not None:
            self.n_dice = n_dice
        if wild_six is not None:
            self.wild_six = wild_six
        self._all_rolls = self._enumerate_rolls(self.n_dice)
        n = len(self._all_rolls)
        self._log_probs = [-math.log(n)] * n
        self._roll_index = {r: i for i, r in enumerate(self._all_rolls)}
        self._np_rolls = None

class BeliefState:
    """ReBeL-inspired formal belief state for Liar's Dice.

    Maintains P(opponent_support = k | observed_bids) for each face value as a
    marginal distribution updated sequentially via Bayesian likelihood weighting.
    Used to compute Call-Liar EV and to inject probabilistic context per turn.

    Reference: Brown et al. 2019 — ReBeL framework for imperfect-information games.
    """

    BLUFF_PRIOR: float = 0.15

    def __init__(self, n_opp_dice: int = 5, wild_six: bool = True) -> None:
        self.n_opp_dice = n_opp_dice
        self.wild_six = wild_six
        self._belief: dict[int, list[float]] = {}
        self._turns_observed: int = 0
        self._reset_belief()

    def _reset_belief(self) -> None:
        """Prior: Binomial(n_opp_dice, p) marginal per face value."""
        n = self.n_opp_dice
        for face in range(1, 7):
            p = 2.0 / 6.0 if (self.wild_six and face != 6) else 1.0 / 6.0
            self._belief[face] = [float(math.comb(n, k) * (p**k) * ((1-p)**(n-k))) for k in range(n + 1)]

    def update(self, bid: tuple[int, int], own_support: int) -> None:
        """Bayesian update of support distribution given opponent's bid."""
        qty, face = bid
        need_from_opp = max(qty - own_support, 0)
        n = self.n_opp_dice
        self._turns_observed += 1
        new_belief = []
        for k in range(n + 1):
            likelihood = (1.0 - self.BLUFF_PRIOR) if k >= need_from_opp else self.BLUFF_PRIOR
            new_belief.append(self._belief[face][k] * likelihood)
        total = sum(new_belief)
        if total > 0:
            self._belief[face] = [v / total for v in new_belief]

    def expected_support(self, face: int) -> float:
        """E[opponent dice showing face] under current belief."""
        return sum(k * p for k, p in enumerate(self._belief.get(face, [])))

    def call_ev(self, bid: tuple[int, int], own_support: int) -> float:
        """EV of calling Liar in [-1, +1]. Positive = calling is favorable."""
        qty, face = bid
        need = max(qty - own_support, 0)
        belief = self._belief.get(face, [])
        p_false = sum(belief[k] for k in range(min(need, len(belief))))
        return p_false - (1.0 - p_false)

    def confidence(self) -> float:
        """Posterior confidence level [0,1] based on observations so far."""
        return min(self._turns_observed / 5.0, 1.0)

    def context_summary(self, current_bid: tuple[int, int] | None, own_support: int = 0) -> str:
        """Short context string for injection into observation."""
        if current_bid is None or self._turns_observed == 0:
            return ""
        qty, face = current_bid
        exp = self.expected_support(face)
        ev = self.call_ev(current_bid, own_support)
        conf = self.confidence()
        ev_label = "CALL FAVORABLE" if ev > 0.10 else ("BORDERLINE" if ev > -0.10 else "RAISE FAVORABLE")
        return (
            f"[BeliefState] Opp support face {face}: ~{exp:.1f} (conf={conf:.0%})\n"
            f"  Call EV: {ev:+.2f} \u2192 {ev_label}"
        )

    def reset(self, n_opp_dice: int | None = None, wild_six: bool | None = None) -> None:
        if n_opp_dice is not None:
            self.n_opp_dice = n_opp_dice
        if wild_six is not None:
            self.wild_six = wild_six
        self._reset_belief()
        self._turns_observed = 0

class BidMomentumTracker:
    """Track bid aggressiveness momentum across turns via exponential moving average.

    Momentum = EMA of normalised bid jump size:
      jump == 2 (Nash-optimal minimal increment) maps to 0
      jump >= 6 (aggressive overbid) maps to +1

    - High momentum (>0.25): aggressive — mild negative shaping signal
    - Low momentum (<-0.05): conservative (Nash-optimal) — mild positive shaping signal

    Also tracks opponent bid jumps for context injection.
    """

    MOMENTUM_DECAY: float = 0.7
    MOMENTUM_SHAPING: float = 0.005

    def __init__(self) -> None:
        self._our_momentum: float = 0.0
        self._last_our_bid_rank: int | None = None
        self._last_opp_bid_rank: int | None = None
        self._our_jumps: list[int] = []
        self._opp_jumps: list[int] = []
        self._turn_count: int = 0

    def update_our_bid(self, bid_rank: int) -> None:
        """Update momentum after we place a bid."""
        if self._last_our_bid_rank is not None:
            jump = bid_rank - self._last_our_bid_rank
            self._our_jumps.append(jump)
            raw = (jump - 2) / 4.0
            self._our_momentum = (
                self._our_momentum * self.MOMENTUM_DECAY
                + raw * (1.0 - self.MOMENTUM_DECAY)
            )
        self._last_our_bid_rank = bid_rank
        self._turn_count += 1

    def update_opp_bid(self, opp_bid_rank: int) -> None:
        """Track opponent bid jumps (context only, not used for shaping)."""
        if self._last_opp_bid_rank is not None:
            self._opp_jumps.append(opp_bid_rank - self._last_opp_bid_rank)
        self._last_opp_bid_rank = opp_bid_rank

    def is_aggressive(self) -> bool:
        return self._our_momentum > 0.25

    def is_conservative(self) -> bool:
        return self._our_momentum < -0.05

    def momentum_shaping(self) -> float:
        """Small shaping: conservative (Nash-optimal) +0.005; aggressive -0.005;
        mixed strategy (high variance in jumps) +TOM_MIXED_STRATEGY_BONUS.
        Returns 0 until 2+ bids observed.
        """
        if self._turn_count < 2:
            return 0.0
        shaping = 0.0
        if self.is_conservative():
            shaping += self.MOMENTUM_SHAPING
        if self.is_aggressive():
            shaping -= self.MOMENTUM_SHAPING
        # Mixed strategy bonus: reward jump variance (unpredictability)
        if len(self._our_jumps) >= 2:
            avg_jump = sum(self._our_jumps) / len(self._our_jumps)
            variance = sum((j - avg_jump) ** 2 for j in self._our_jumps) / len(self._our_jumps)
            if variance >= 2.0:
                shaping += TOM_MIXED_STRATEGY_BONUS
        return shaping

    def context_summary(self) -> str:
        """Short context string for observation injection."""
        if self._turn_count == 0:
            return ""
        style = "AGGRESSIVE" if self.is_aggressive() else (
            "CONSERVATIVE" if self.is_conservative() else "MODERATE"
        )
        our_avg = sum(self._our_jumps) / len(self._our_jumps) if self._our_jumps else 0.0
        opp_avg = sum(self._opp_jumps) / len(self._opp_jumps) if self._opp_jumps else 0.0
        return (
            f"[BidMomentum] Style: {style} "
            f"(our avg jump={our_avg:.1f}, opp avg jump={opp_avg:.1f})"
        )

    def reset(self) -> None:
        self._our_momentum = 0.0
        self._last_our_bid_rank = None
        self._last_opp_bid_rank = None
        self._our_jumps = []
        self._opp_jumps = []
        self._turn_count = 0

class EndgameCFRSolver:
    """CFR solver specialised for late-game Liar's Dice (few dice remaining).

    Activates only when total_dice <= ENDGAME_THRESHOLD (default 4). In this regime
    the state space shrinks dramatically and decisions are highly consequential.
    Provides a stronger shaping signal than the general CFRTable and tracks a
    finer info-set key: (sorted_dice_pattern, total_dice, current_bid).

    Thread-safe; shared across episodes like CFRTable.
    """

    ENDGAME_THRESHOLD: int = 4
    ENDGAME_SHAPING_SCALE: float = 0.03

    def __init__(self) -> None:
        self._lock = Lock()
        self._regret: dict[tuple[int, int], float] = {}
        self._episode_count: int = 0

    def _info_key(
        self,
        own_dice: list[int],
        total_dice: int,
        current_bid: tuple[int, int] | None,
    ) -> int:
        return hash((tuple(sorted(own_dice)), total_dice, current_bid))

    def _get_strategy(self, info_key: int, legal_actions: list[int]) -> dict[int, float]:
        """Regret-matching strategy for the given info-set."""
        with self._lock:
            regrets = {a: self._regret.get((info_key, a), 0.0) for a in legal_actions}
        positive = {a: max(0.0, r) for a, r in regrets.items()}
        total = sum(positive.values())
        if total <= 0:
            n = max(len(legal_actions), 1)
            return {a: 1.0 / n for a in legal_actions}
        return {a: v / total for a, v in positive.items()}

    def endgame_shaping(
        self,
        own_dice: list[int],
        total_dice: int,
        current_bid: tuple[int, int] | None,
        chosen_bid_rank: int,
        legal_bid_ranks: list[int],
    ) -> float:
        """CFR-based shaping for bid decisions in endgame. Returns 0.0 outside endgame."""
        if total_dice > self.ENDGAME_THRESHOLD or not own_dice or not legal_bid_ranks:
            return 0.0
        ik = self._info_key(own_dice, total_dice, current_bid)
        strategy = self._get_strategy(ik, legal_bid_ranks)
        chosen_prob = strategy.get(chosen_bid_rank, 0.0)
        uniform_prob = 1.0 / max(len(legal_bid_ranks), 1)
        deviation = chosen_prob - uniform_prob
        max_dev = max(1.0 - uniform_prob, uniform_prob)
        if max_dev > 0:
            return self.ENDGAME_SHAPING_SCALE * (deviation / max_dev)
        return 0.0

    def call_shaping(
        self,
        own_dice: list[int],
        total_dice: int,
        episode_won: bool,
        call_quality_bonus: float,
        call_quality_penalty: float,
    ) -> float:
        """Additive call-liar shaping in endgame (on top of BayesianOpponentInference.call_shaping)."""
        if total_dice > self.ENDGAME_THRESHOLD or not own_dice:
            return 0.0
        multiplier = 1.0 + max(0.0, self.ENDGAME_THRESHOLD - total_dice) / self.ENDGAME_THRESHOLD
        if episode_won:
            return call_quality_bonus * multiplier * 0.5
        return -call_quality_penalty * multiplier * 0.5

    def update(
        self,
        own_dice: list[int],
        total_dice: int,
        current_bid: tuple[int, int] | None,
        action_taken: int,
        legal_bid_ranks: list[int],
        episode_reward: float,
    ) -> None:
        """Update endgame regret table after episode (same sign convention as CFRTable)."""
        if total_dice > self.ENDGAME_THRESHOLD or not own_dice or not legal_bid_ranks:
            return
        ik = self._info_key(own_dice, total_dice, current_bid)
        with self._lock:
            self._episode_count += 1
            for a in legal_bid_ranks:
                if a == action_taken:
                    self._regret[(ik, a)] = self._regret.get((ik, a), 0.0) - episode_reward * 0.1
                else:
                    self._regret[(ik, a)] = self._regret.get((ik, a), 0.0) - episode_reward * 0.05

    def stats(self) -> str:
        with self._lock:
            return f"EndgameCFR: {len(self._regret)} entries, {self._episode_count} episodes"

def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def _binomial_tail_probability(num_trials: int, success_prob: float, min_successes: int) -> float:
    """Pure-math binomial tail P(X >= min_successes). No scipy required."""
    if min_successes <= 0:
        return 1.0
    if num_trials <= 0:
        return 0.0
    success_prob = _clamp(success_prob, 0.0, 1.0)
    tail = 0.0
    for k in range(min_successes, num_trials + 1):
        tail += math.comb(num_trials, k) * (success_prob ** k) * ((1.0 - success_prob) ** (num_trials - k))
    return _clamp(tail, 0.0, 1.0)

def _ruleset_from_env() -> str | None:
    raw = (os.environ.get("LIARS_DICE_RULESET") or "").strip().lower()
    if raw in ("classic", "multi", "multi_dice"):
        return RULESET_CLASSIC
    if raw in ("liars_die", "liar_die", "fsicfr", "single_die"):
        return RULESET_LIARS_DIE
    return None

def _detect_ruleset_from_observation(observation: str) -> str:
    if not observation:
        return RULESET_CLASSIC
    if re.search(r'Current bid:\s*"', observation):
        return RULESET_CLASSIC
    if re.search(
        r"(?i)(your roll|you rolled|previous claim|opponent claim|their claim|must claim|strictly higher|doubt|accept)",
        observation,
    ):
        return RULESET_LIARS_DIE
    return RULESET_CLASSIC

def resolve_ruleset(observation: str) -> str:
    """Determine ruleset: env var override > auto-detect from observation."""
    env_ruleset = _ruleset_from_env()
    if env_ruleset is not None:
        return env_ruleset
    return _detect_ruleset_from_observation(observation)

def _extract_liars_die_state_features(observation: str) -> dict:
    """Extract state features for the single-die FSICFR Liar's die variant."""
    sides = 6
    m = re.search(r"(?:Die sides|Sides|S)\s*[:=]\s*(\d+)", observation, flags=re.IGNORECASE)
    if m:
        sides = max(2, int(m.group(1)))

    your_roll: int | None = None
    for pat in (
        r"(?:Your roll|You rolled|Private roll|Your die)\s*[: ]\s*(\d+)",
        r"(?i)rolled\s+a\s*(\d+)",
    ):
        m = re.search(pat, observation)
        if m:
            your_roll = int(m.group(1))
            break

    previous_claim: int | None = None
    for pat in (
        r"(?:Previous claim|Opponent claim|Their claim|Last claim)\s*[: ]\s*(\d+)",
        r"(?i)claimed\s+rank\s*(\d+)",
    ):
        m = re.search(pat, observation)
        if m:
            previous_claim = int(m.group(1))
            break

    min_next_claim: int | None = None
    m = re.search(
        r"(?i)(?:must (?:claim|bid)|claim) (?:at least|over|>|strictly higher than)\s*(\d+)",
        observation,
    )
    if m:
        min_next_claim = int(m.group(1))

    return {
        "ruleset": RULESET_LIARS_DIE,
        "die_sides": sides,
        "your_roll": your_roll,
        "previous_claim": previous_claim,
        "min_next_claim": min_next_claim,
        "own_dice": [your_roll] if your_roll is not None else [],
        "total_dice": 0,
        "current_bid": None,
        "wild_six_enabled": False,
    }

def _liars_die_parse_action(label: str) -> tuple[str, int | None]:
    """Parse a Liar's die action label into (kind, claim_rank)."""
    low = (label or "").strip().lower()
    if "doubt" in low:
        return "doubt", None
    if "accept" in low:
        return "accept", None
    m = re.search(r"(?:claim|rank)\s*(\d+)", low)
    if m:
        return "claim", int(m.group(1))
    m = re.search(r"(\d+)\s*-\s*(\d+)", label or "")
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a == 1:
            return "claim", b
    m = re.match(r"^\s*(\d+)\s*$", (label or "").strip())
    if m:
        return "claim", int(m.group(1))
    return "unknown", None

def _liars_die_compute_shaping(
    state: dict,
    action_kind: str,
    claim_rank: int | None,
) -> tuple[float, float, dict]:
    """Return (bid_shaping, decision_shaping, meta) for the Liar's die variant."""
    sides = max(2, int(state.get("die_sides") or 6))
    prev = state.get("previous_claim")
    your_roll = state.get("your_roll")
    meta: dict = {"current_bid_z": 0.0, "current_bid_truth_probability": 0.0}
    bid_shaping = 0.0
    decision_shaping = 0.0

    if action_kind in ("doubt", "accept"):
        if prev is None:
            return 0.0, 0.0, meta
        p = int(prev)
        truth_proxy = _clamp((sides - p + 1) / float(sides), 0.0, 1.0)
        meta["current_bid_truth_probability"] = truth_proxy
        meta["current_bid_z"] = (0.5 - truth_proxy) * 4.0
        if action_kind == "doubt":
            if truth_proxy <= 0.35:
                decision_shaping += BID_PLAUSIBILITY_BONUS * 0.5
            elif truth_proxy >= 0.55:
                decision_shaping -= BID_PLAUSIBILITY_PENALTY * 0.5
        else:
            if truth_proxy <= 0.15:
                decision_shaping -= PASS_MISSED_CHALLENGE_PENALTY * 0.8
            elif truth_proxy >= 0.5:
                decision_shaping += 0.01

    elif action_kind == "claim" and claim_rank is not None and your_roll is not None:
        c = int(claim_rank)
        r = int(your_roll)
        if c == r:
            bid_shaping += BID_PLAUSIBILITY_BONUS
        elif c > r:
            if c <= sides:
                bid_shaping += BID_PLAUSIBILITY_BONUS * 0.4
            if c - r > max(1, sides // 2):
                bid_shaping -= BID_PLAUSIBILITY_PENALTY * 0.5
        else:
            bid_shaping -= BID_PLAUSIBILITY_PENALTY

    return bid_shaping, decision_shaping, meta

def _estimate_bid_statistics(state_features: dict, bid: tuple[int, int]) -> dict:
    """Centralized bid statistics: known_support, unknown_dice, expected, std_dev, z_score, truth_probability."""
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    wild_six_enabled = bool(state_features.get("wild_six_enabled"))
    quantity, face = bid

    if total_dice <= 0 or not own_dice:
        return {"known_support": 0, "unknown_dice": 0, "expected_total": 0.0,
                "std_dev": 0.0, "z_score": 0.0, "truth_probability": 0.0}

    known_support = _count_face_support(own_dice, face, wild_six_enabled)
    unknown_dice = max(total_dice - len(own_dice), 0)
    per_die_p = 2.0 / 6.0 if (wild_six_enabled and face != 6) else 1.0 / 6.0

    if unknown_dice == 0:
        expected_total = float(known_support)
        std_dev = 0.0
    else:
        expected_total = known_support + unknown_dice * per_die_p
        std_dev = math.sqrt(unknown_dice * per_die_p * (1.0 - per_die_p))

    need = max(quantity - known_support, 0)
    truth_probability = _binomial_tail_probability(unknown_dice, per_die_p, need)

    if std_dev > 0:
        z_score = (quantity - expected_total) / std_dev
    elif quantity <= expected_total:
        z_score = -1.0
    else:
        z_score = 3.0

    return {
        "known_support": known_support,
        "unknown_dice": unknown_dice,
        "expected_total": expected_total,
        "std_dev": std_dev,
        "z_score": z_score,
        "truth_probability": truth_probability,
    }

def _score_challenge_decision(
    state_features: dict,
    chose_liar: bool,
    proposed_bid: tuple[int, int] | None,
) -> tuple[float, dict]:
    """Evaluate the quality of a challenge (Call Liar) decision. Returns (shaping, meta)."""
    current_bid = state_features.get("current_bid")
    if current_bid is None:
        return 0.0, {"current_bid_z": 0.0, "current_bid_truth_probability": 0.0}

    stats = _estimate_bid_statistics(state_features, current_bid)
    z = float(stats["z_score"])
    tp = float(stats["truth_probability"])
    reward = 0.0

    if not chose_liar and proposed_bid is not None:
        if tp <= 0.10:
            reward -= PASS_MISSED_CHALLENGE_PENALTY * (1.0 + _clamp((0.10 - tp) / 0.10, 0.0, 1.0))
        elif tp >= 0.55:
            reward += 0.01

    return reward, {"current_bid_z": z, "current_bid_truth_probability": tp}

def _select_fallback_action(
    legal_action_map: dict[str, str],
    state_features: dict,
    ruleset: str = RULESET_CLASSIC,
) -> str:
    """Stat-informed fallback when LLM output cannot be parsed."""
    if ruleset == RULESET_LIARS_DIE:
        doubt_ids = [aid for aid, lab in legal_action_map.items() if "doubt" in lab.lower()]
        prev = state_features.get("previous_claim")
        sides = max(2, int(state_features.get("die_sides") or 6))
        if doubt_ids and prev is not None:
            truth_proxy = _clamp((sides - int(prev) + 1) / float(sides), 0.0, 1.0)
            if truth_proxy <= 0.12:
                return doubt_ids[0]
        return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]

    liar_actions = [aid for aid, lbl in legal_action_map.items() if _is_liar_label(lbl)]
    current_bid = state_features.get("current_bid")
    if liar_actions and current_bid is not None:
        stats = _estimate_bid_statistics(state_features, current_bid)
        if float(stats["truth_probability"]) <= 0.08:
            return liar_actions[0]
    return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]

def extract_and_format_observation(obs_text: str) -> str:
    return obs_text or ""

class EpisodeTraceLogger:
    def __init__(self, trace_dir: str, rank: int):
        self.trace_dir = trace_dir
        self.rank = rank
        self._lock = Lock()
        self.log_path = os.path.join(self.trace_dir, f"liars_dice_episode_traces_rank{rank}.jsonl")
        self.max_text_chars = int(os.environ.get("EPISODE_TRACE_MAX_TEXT_CHARS", "4000"))
        self.sample_rate = float(os.environ.get("EPISODE_TRACE_SAMPLE_RATE", "1.0"))
        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[EPISODE_TRACE] Writing traces to {self.log_path}")
    def should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() <= self.sample_rate
    def clip_text(self, text: str) -> str:
        if not text or len(text) <= self.max_text_chars:
            return text or ""
        return text[: self.max_text_chars] + f"... [truncated {len(text) - self.max_text_chars} chars]"
    def log_episode(self, payload: dict) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")

class CurriculumScheduler:
    def __init__(self, initial_max_turn=2, final_max_turn=20, rollouts_per_stage=1280, initial_hint_prob=0.50, final_hint_prob=0.05, warmup_rollouts=128):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0
    def get_max_turn(self) -> int:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)
    def get_hint_prob(self) -> float:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = max(self.final_max_turn - self.initial_max_turn, 1)
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)
    def step(self, num_rollouts=1):
        self.total_rollouts += num_rollouts

def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(rf"<{tag_name}>.*?</{close_name}>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()

def _extract_legal_action_map(observation: str) -> dict[str, str]:
    if not observation:
        return {}
    match = re.search(r"Legal Actions:\s*\n(.*?)(?:\n\nYour choice|\nYour choice|\Z)", observation, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return {}
    block = match.group(1)
    mapping: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            action_id = left.strip()
            label = right.strip()
        else:
            action_id = line.strip()
            label = action_id
        if re.fullmatch(r"-?\d+", action_id):
            mapping[action_id] = label
    return mapping

def _extract_bid_tuple(label_or_text: str) -> tuple[int, int] | None:
    if not label_or_text:
        return None
    match = re.search(r"(\d+)\s*-\s*(\d+)", label_or_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))

def _extract_state_features(observation: str, ruleset: str = RULESET_CLASSIC) -> dict:
    if ruleset == RULESET_LIARS_DIE:
        return _extract_liars_die_state_features(observation)
    dice: list[int] = []
    dice_match = re.search(r"Your dice:\s*\[([^\]]*)\]", observation)
    if dice_match:
        dice_str = dice_match.group(1).strip()
        if dice_str:
            dice = [int(x.strip()) for x in dice_str.split(",") if x.strip().isdigit()]
    total_dice_match = re.search(r"Total dice in game:\s*(\d+)", observation)
    total_dice = int(total_dice_match.group(1)) if total_dice_match else 0
    current_bid_match = re.search(r'Current bid:\s*"([^"]+)"', observation)
    current_bid = _extract_bid_tuple(current_bid_match.group(1)) if current_bid_match else None
    return {"own_dice": dice, "total_dice": total_dice, "current_bid": current_bid, "wild_six_enabled": "wild" in observation.lower() and "6" in observation}

def _is_liar_label(label: str) -> bool:
    return "liar" in (label or "").strip().lower()

def _bid_rank(bid: tuple[int, int]) -> int:
    quantity, face = bid
    return quantity * 6 + face

def _count_face_support(own_dice: list[int], target_face: int, wild_six_enabled: bool) -> int:
    if wild_six_enabled and target_face != 6:
        return sum(1 for value in own_dice if value == target_face or value == 6)
    return sum(1 for value in own_dice if value == target_face)

def _score_bid_plausibility(state_features: dict, bid: tuple[int, int]) -> float:
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    current_bid = state_features.get("current_bid")
    wild_six_enabled = bool(state_features.get("wild_six_enabled"))
    if total_dice <= 0 or not own_dice:
        return 0.0
    quantity, face = bid
    known_support = _count_face_support(own_dice, face, wild_six_enabled)
    unknown_dice = max(total_dice - len(own_dice), 0)
    p = 2.0 / 6.0 if (wild_six_enabled and face != 6) else 1.0 / 6.0
    need_from_unknown = max(quantity - known_support, 0)
    if unknown_dice <= 0:
        p_bid_true = 1.0 if need_from_unknown <= 0 else 0.0
    elif need_from_unknown <= 0:
        p_bid_true = 1.0
    else:
        p_bid_true = _binomial_tail_probability(unknown_dice, p, need_from_unknown)
    reward = BID_PLAUSIBILITY_BONUS * (2.0 * p_bid_true - 1.0)
    if current_bid is not None:
        jump = _bid_rank(bid) - _bid_rank(current_bid)
        if jump <= 2:
            reward += 0.01
        elif jump >= 7:
            if p_bid_true < 0.30:
                reward -= 0.03
            else:
                reward += 0.01
    return reward

def _score_bid_quality(bid: tuple[int, int], previous_bid: tuple[int, int] | None, state_features: dict) -> float:
    """Penalize reckless overbids (jump >= 4 when bid exceeds expected total)."""
    if previous_bid is None:
        return 0.0
    prev_rank = _bid_rank(previous_bid)
    curr_rank = _bid_rank(bid)
    jump = curr_rank - prev_rank
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 1)

    if jump >= 4:
        qty, face = bid
        wild_six = bool(state_features.get("wild_six_enabled"))
        known_support = _count_face_support(own_dice, face, wild_six)
        expected_unknown = max(total_dice - len(own_dice), 0) * (2.0 / 6.0 if (wild_six and face != 6) else 1.0 / 6.0)
        expected_total = known_support + expected_unknown
        if qty > expected_total + 1:
            return -BID_AGGRESSIVE_PENALTY

    # --- M8: Position-Aware Bid Shaping ---
    if previous_bid is None:
        # First bidder: reward conservative opening bids
        qty, face = bid
        if total_dice > 0 and qty <= total_dice * 0.3:
            return FIRST_BIDDER_CONSERVATIVE_BONUS

    return 0.0

def _adaptive_bluff_threshold(total_dice: int) -> float:
    MAX_STARTING_DICE = 10
    ratio = min(max(total_dice, 0) / MAX_STARTING_DICE, 1.0)
    return 0.35 + 0.15 * ratio

def _compute_dice_context_summary(state_features: dict, current_bid: tuple[int, int] | None) -> str:
    own_dice: list[int] = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    wild_six = bool(state_features.get("wild_six_enabled"))
    if total_dice == 0:
        return ""
    p = 2.0 / 6.0 if wild_six else 1.0 / 6.0
    expected_per_face = round(total_dice * p, 1)
    threshold = _adaptive_bluff_threshold(total_dice)
    lines = [
        f"[Dice context] Total dice: {total_dice} | Your dice: {own_dice}",
        f"Expected any face across all dice: ~{expected_per_face} ({'wild-6 active: p=2/6' if wild_six else 'standard: p=1/6'})",
    ]
    if current_bid is not None:
        qty, face = current_bid
        ratio = qty / total_dice
        extreme_threshold = threshold + 0.10
        if ratio >= extreme_threshold:
            lines.append(f"Current bid {qty}×{face}: {ratio:.0%} of total dice — EXTREME BLUFF ZONE → Consider calling Liar")
        elif ratio >= threshold:
            lines.append(f"Current bid {qty}×{face}: {ratio:.0%} of total dice — BLUFF ZONE → Challenging is reasonable")
        else:
            lines.append(f"Current bid {qty}×{face}: {ratio:.0%} of total dice — within normal range")
    return "\n".join(lines)

def _classify_opponent_style(momentum_tracker: "BidMomentumTracker") -> str:
    """Classify opponent bidding style from jump history.

    Opponent patterns reveal hand strength.
    Conservative (small jumps) = likely strong; Aggressive (large jumps) = likely bluffing.
    Returns a short descriptive string for context injection.
    """
    opp_jumps = getattr(momentum_tracker, "_opp_jumps", [])
    if len(opp_jumps) < 2:
        return ""
    avg_jump = sum(opp_jumps) / len(opp_jumps)
    large_jump_count = sum(1 for j in opp_jumps if j >= 4)
    large_jump_ratio = large_jump_count / len(opp_jumps)
    if avg_jump >= 4.0 or large_jump_ratio >= 0.5:
        style = "AGGRESSIVE"
        advice = "likely bluffing frequently — challenge more confidently"
    elif avg_jump <= 2.0 and large_jump_ratio <= 0.1:
        style = "CONSERVATIVE"
        advice = "likely bidding strong hands — be cautious when challenging"
    else:
        style = "MODERATE"
        advice = "mixed pattern — use Kelly/Bayesian signals to decide"
    return f"[Opponent Style] {style} (avg jump={avg_jump:.1f}) — {advice}"

def kelly_bluff_context(
    state_features: dict,
    current_bid: tuple[int, int] | None,
    bayes_opp=None,
) -> str:
    """Kelly Criterion context for call-or-raise decisions.

    Kelly formula (zero-sum game, b=1): f* = p_win - p_lose = 2*p_true_if_raise - 1
    Applied to the current bid:
      f* > +0.15  → RAISE recommended (bid likely true, contesting is -EV)
      f* < -0.15  → CALL recommended  (bid likely false, calling is +EV)
      otherwise   → BORDERLINE

    Uses Bayesian posterior when available (turn 2+), Binomial CDF otherwise.
    Returns empty string if state information is insufficient.
    """
    if current_bid is None:
        return ""
    own_dice: list[int] = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    wild_six = bool(state_features.get("wild_six_enabled"))
    if total_dice <= 0:
        return ""
    qty, face = current_bid
    known_support = _count_face_support(own_dice, face, wild_six)
    unknown_dice = max(total_dice - len(own_dice), 0)
    p = 2.0 / 6.0 if (wild_six and face != 6) else 1.0 / 6.0
    need_from_unknown = max(qty - known_support, 0)
    use_bayesian = bayes_opp is not None and getattr(bayes_opp, "_bids_observed", 0) > 0
    if use_bayesian:
        p_true = bayes_opp.bid_posterior_prob(current_bid, known_support)
    elif unknown_dice <= 0:
        p_true = 1.0 if need_from_unknown <= 0 else 0.0
    elif need_from_unknown <= 0:
        p_true = 1.0
    else:
        p_true = _binomial_tail_probability(unknown_dice, p, need_from_unknown)
    kelly_f = (1.0 - p_true) - p_true
    if kelly_f < -0.15:
        recommendation = f"RAISE (f*={kelly_f:+.2f}, bid likely true p={p_true:.0%})"
    elif kelly_f > 0.15:
        recommendation = f"CALL  (f*={kelly_f:+.2f}, bid likely false p_false={(1-p_true):.0%})"
    else:
        recommendation = f"BORDERLINE (f*={kelly_f:+.2f})"
    return f"[Kelly] {recommendation}"

def _parse_action_id(completion_text: str, legal_action_map: dict[str, str], ruleset: str = RULESET_CLASSIC) -> str:
    if not legal_action_map:
        return ""
    cleaned = remove_reasoning_tags(completion_text or "")
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5]
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()
    for num in re.findall(r"-?\d+", cleaned):
        if num in legal_action_map:
            return num
    normalized = cleaned.strip().lower()
    for action_id, label in legal_action_map.items():
        if normalized == label.strip().lower():
            return action_id
    if ruleset == RULESET_LIARS_DIE:
        if "doubt" in normalized:
            for action_id, label in legal_action_map.items():
                if "doubt" in label.strip().lower():
                    return action_id
        if "accept" in normalized:
            for action_id, label in legal_action_map.items():
                if "accept" in label.strip().lower():
                    return action_id
    if "liar" in normalized:
        for action_id, label in legal_action_map.items():
            if _is_liar_label(label):
                return action_id
    bid_tuple = _extract_bid_tuple(cleaned)
    if bid_tuple is not None:
        for action_id, label in legal_action_map.items():
            if _extract_bid_tuple(label) == bid_tuple:
                return action_id
    _sf = _ROLLOUT_STATE.get("_current_state_features", {})
    if _sf and ruleset == RULESET_CLASSIC:
        _cb = _sf.get("current_bid")
        if _cb is not None:
            _own = _sf.get("own_dice") or []
            _total = int(_sf.get("total_dice") or 0)
            _wild = bool(_sf.get("wild_six_enabled"))
            if _total > 0 and _own:
                _qty, _face = _cb
                _known = _count_face_support(_own, _face, _wild)
                _unknown = max(_total - len(_own), 0)
                _p = 2.0 / 6.0 if (_wild and _face != 6) else 1.0 / 6.0
                _need = max(_qty - _known, 0)
                _truth_prob = _binomial_tail_probability(_unknown, _p, _need) if _unknown > 0 and _need > 0 else 1.0
                if _truth_prob <= 0.08:
                    for _aid, _lbl in legal_action_map.items():
                        if _is_liar_label(_lbl):
                            return _aid
    return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]

def _extract_terminal_reward(step_block: dict, observation_text: str) -> float:
    info = step_block.get("info", {}) if isinstance(step_block, dict) else {}
    cumulative_reward = info.get("cumulative_reward")
    if isinstance(cumulative_reward, (int, float)):
        return _clamp(float(cumulative_reward), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)
    your_return_match = re.search(r"Your Return:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    if your_return_match:
        return _clamp(float(your_return_match.group(1)), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)
    normalized_match = re.search(r"Normalized Score:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    result_match = re.search(r"Result:\s*(WIN|LOSS|DRAW)", observation_text or "", flags=re.IGNORECASE)
    if normalized_match:
        normalized_value = float(normalized_match.group(1))
        if result_match:
            result = result_match.group(1).upper()
            if result == "LOSS":
                normalized_value = -abs(normalized_value) if normalized_value != 0 else -1.0
            elif result == "WIN":
                normalized_value = abs(normalized_value) if normalized_value != 0 else 1.0
            else:
                normalized_value = 0.0
        return _clamp(normalized_value, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)
    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    return _clamp(step_reward, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

def _build_env_pool(server_urls: list[str]) -> list[dict[str, str]]:
    env_pool = []
    init_task_id = GAME_TO_TASK_ID_RANGE[SELECTED_GAME][0]
    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            payload = {"task_id": init_task_id, "seed": 42, **MCTS_CONFIG}
            res = requests.post(f"{base_url}/reset", json=payload, timeout=INIT_TIMEOUT_SECONDS)
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as e:
            raise RuntimeError(f"Failed to init server {base_url}: {e}") from e
    return env_pool

def _initialize_rollout_state(trainer) -> None:
    if _ROLLOUT_STATE.get("initialized", False):
        return
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")
    env_pool = _build_env_pool(server_urls)
    rollout_per_stage = int(getattr(trainer.args, "rollouts_per_stage", 1280))
    initial_max_turn = int(getattr(trainer.args, "initial_max_turn", 2))
    final_max_turn = int(os.environ.get("LIARS_DICE_FINAL_MAX_TURN", "20"))
    initial_hint_prob = float(os.environ.get("LIARS_DICE_INITIAL_HINT_PROB", "0.50"))
    final_hint_prob = float(os.environ.get("LIARS_DICE_FINAL_HINT_PROB", "0.05"))
    _ROLLOUT_STATE["rank"] = rank
    _ROLLOUT_STATE["env_pool"] = env_pool
    _ROLLOUT_STATE["num_servers"] = len(env_pool)
    _ROLLOUT_STATE["thread_pool"] = ThreadPoolExecutor(max_workers=len(env_pool))
    _ROLLOUT_STATE["generation_semaphore"] = Semaphore(1)
    rollout_warmup_rollouts = (
        int(trainer.args.rollout_warmup_rollouts)
        if getattr(trainer.args, "rollout_warmup_rollouts", None) is not None
        else rollout_per_stage
    )
    _ROLLOUT_STATE["curriculum"] = CurriculumScheduler(initial_max_turn=initial_max_turn, final_max_turn=final_max_turn, rollouts_per_stage=rollout_per_stage, initial_hint_prob=initial_hint_prob, final_hint_prob=final_hint_prob, warmup_rollouts=rollout_warmup_rollouts)
    print(f"[CURRICULUM] LD initialized: max_turn={initial_max_turn}->{final_max_turn}, rollouts_per_stage={rollout_per_stage}, warmup_rollouts={rollout_warmup_rollouts}")
    _ROLLOUT_STATE["cfr_table"] = CFRTable()
    _ROLLOUT_STATE["bayes_opp"] = BayesianOpponentInference(n_dice=5, wild_six=True)
    _ROLLOUT_STATE["endgame_cfr"] = EndgameCFRSolver()
    _ROLLOUT_STATE["dice_rag"] = DiceRAG()
    _ROLLOUT_STATE["initialized"] = True
    trace_enabled = _is_truthy_env(os.environ.get("EPISODE_TRACE_ENABLED", "1"))
    trace_dir = os.environ.get("EPISODE_TRACE_DIR", "").strip()
    _ROLLOUT_STATE["trace_logger"] = None
    if trace_enabled and trace_dir:
        try:
            _ROLLOUT_STATE["trace_logger"] = EpisodeTraceLogger(trace_dir=trace_dir, rank=rank)
        except Exception as e:
            print(f"[EPISODE_TRACE] Failed to initialize logger: {e}")

def _reset_environment(env_endpoint: str, game_id: int, timeout: int) -> tuple[str, str]:
    payload = {"task_id": game_id, "seed": random.randint(0, 2**31 - 1), **MCTS_CONFIG}
    reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=timeout)
    reset_res.raise_for_status()
    result_block = reset_res.json()["result"]
    episode_id = result_block.get("episode_id", "")
    raw_observation = result_block.get("observation", "")
    return episode_id, extract_and_format_observation(raw_observation)

def _step_environment(env_endpoint: str, episode_id: str, action_to_send: str, timeout: int) -> tuple[str, float, bool, dict]:
    step_payload = {"action": action_to_send, "episode_id": episode_id}
    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=timeout)
    step_res.raise_for_status()
    step_block = step_res.json()["result"]
    raw_observation = step_block.get("observation", "")
    formatted_observation = extract_and_format_observation(raw_observation)
    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    done = bool(step_block.get("done", False))
    return formatted_observation, step_reward, done, step_block

def _last_prompt_fallback_result() -> dict:
    return {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}

def _full_prompt_fallback_result() -> dict:
    return {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0}

def _execute_parallel_rollouts(prompts, executor, run_single_prompt, fallback_builder):
    results = [None] * len(prompts)
    futures = [executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)]
    for future in as_completed(futures):
        idx, res = future.result()
        results[idx] = res if res is not None else fallback_builder()
    return [r for r in results if r is not None]

def _log_batch_statistics(list_results: list[dict]) -> None:
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0.0
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.3f}")

def _bid_history_summary(bid_history: list[dict]) -> str:
    if not bid_history:
        return ""
    parts = []
    for entry in bid_history[-12:]:
        turn = entry.get("turn", "?")
        label = entry.get("label", "?")
        who = "us" if entry.get("is_ours") else "opp"
        parts.append(f"T{turn}({who}):{label}")
    return "Bid history: " + " | ".join(parts)

def _get_system_prompt(use_hints: bool, wild_six_enabled: bool = False, ruleset: str = RULESET_CLASSIC) -> str:
    if ruleset == RULESET_LIARS_DIE:
        system_prompt = """You are playing liars_dice (Liar's die \u2014 FSICFR-style single-die variant).

LIAR'S DIE (single die, rank claims):

Setup: Players use one s-sided die (often 6). The current roller observes the roll privately.

Goal: Either make a claim about die rank, or (as responder) choose Doubt or Accept.

Actions:
- Claim rank: Announce a die face rank (1..s). You may bluff (claim higher than your roll).
- Doubt: Challenge the opponent's claim. The hidden roll is revealed. If roll >= claimed rank, the claimant wins; otherwise the doubter wins.
- Accept: Do not reveal the roll. The die passes to you; you roll again, observe privately, and must claim a rank STRICTLY HIGHER than the previous claim.

You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.
"""
        if use_hints:
            system_prompt += "\n" + STRATEGY_TIPS_LIARS_DIE
        return system_prompt

    wild_note = ("- 6s are WILD: they count as any face value (e.g., a 6 supports any bid).\n" if wild_six_enabled else "- Standard rules: 6s count only as 6 (no wild dice).\n")
    system_prompt = ("You are playing liars_dice.\n\n# Game Rules\nLIAR'S DICE RULES:\n- Each player has hidden dice.\n- Players make increasing bids in format quantity-face.\n- You can either raise the bid or choose Liar to challenge.\n" + wild_note + "- Use ONLY legal action IDs from the current Legal Actions block.\n\n# Output Format\nRespond with only one legal action ID number.\nNo explanations.\n")
    if use_hints:
        system_prompt += "\n" + STRATEGY_TIPS_CLASSIC
    return system_prompt

def _rollout_parallelized_curriculum(prompts: list[str], trainer, include_action_mask: bool) -> dict[str, list]:
    _initialize_rollout_state(trainer)
    rank = _ROLLOUT_STATE["rank"]
    env_pool = _ROLLOUT_STATE["env_pool"]
    num_servers = _ROLLOUT_STATE["num_servers"]
    curriculum: CurriculumScheduler = _ROLLOUT_STATE["curriculum"]
    trace_logger = _ROLLOUT_STATE["trace_logger"]
    tokenizer = trainer.processing_class
    timeout = REQUEST_TIMEOUT_SECONDS
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)
        server_idx = (index + rank) % num_servers
        env_endpoint = env_pool[server_idx]["base_url"]
        invalid_count = 0
        consecutive_invalids = 0
        noop_count = 0
        consecutive_noops = 0
        done = False
        final_reward = 0.0
        turn_number = 0
        step_rewards: list[float] = []     # per-step reward list for discounted return
        bluff_count = 0                     # risky bid counter
        risky_liar_count = 0                # risky liar call counter
        last_action_prob = 0.0              # probability of last action taken
        accumulated_shaping_reward = 0.0    # legacy: kept for context injection/CFR only
        step_records = []
        termination_reason = "unknown"
        last_step_block: dict = {}
        bid_history: list[dict] = []
        wild_six_enabled: bool = False
        episode_bid_ranks_played: list[int] = []
        episode_legal_bid_ranks: list[int] = []
        episode_last_bid_total_dice: int = 0
        cfr_table: CFRTable | None = _ROLLOUT_STATE.get("cfr_table")
        bayes_opp: BayesianOpponentInference | None = _ROLLOUT_STATE.get("bayes_opp")
        endgame_cfr: EndgameCFRSolver | None = _ROLLOUT_STATE.get("endgame_cfr")
        if bayes_opp is not None:
            bayes_opp.reset(n_dice=5, wild_six=True)
        belief_state = BeliefState(n_opp_dice=5, wild_six=True)
        momentum_tracker = BidMomentumTracker()
        episode_own_dice: list[int] = []
        if include_action_mask:
            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            episode_action_mask: list[int] = []
            prev_full_ids: list[int] | None = None
        else:
            prompt_ids_last: list[int] = []
            completion_ids_last: list[int] = []
            logprobs_last: list[float] = []
        try:
            episode_id, formatted_observation = _reset_environment(env_endpoint=env_endpoint, game_id=game_id, timeout=timeout)
            print(f"[START] ID:{game_id} server={server_idx} ep={episode_id[:8] if episode_id else '?'}")
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            if trace_logger and trace_logger.should_log():
                trace_logger.log_episode({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "game_id": game_id, "status": "reset_failed", "error": str(e)})
            return index, None
        use_hints = random.random() < current_hint_prob
        ruleset = resolve_ruleset(formatted_observation)
        state_features_init = _extract_state_features(formatted_observation, ruleset=ruleset)
        wild_six_enabled = bool(state_features_init.get("wild_six_enabled"))
        episode_own_dice = list(state_features_init.get("own_dice") or [])
        if bayes_opp is not None and episode_own_dice:
            bayes_opp.reset(n_dice=5, wild_six=wild_six_enabled)
        # --- Action shuffle on initial observation ---
        init_action_map = _extract_legal_action_map(formatted_observation)
        formatted_observation = _shuffle_observation_actions(formatted_observation, init_action_map)
        messages = [
            {"role": "system", "content": _get_system_prompt(use_hints=use_hints, wild_six_enabled=wild_six_enabled, ruleset=ruleset)},
            {"role": "user", "content": formatted_observation},
        ]
        while not done and turn_number < current_max_turn:
            observation_before_action = formatted_observation
            legal_action_map = _extract_legal_action_map(observation_before_action)
            state_features = _extract_state_features(observation_before_action, ruleset=ruleset)
            state_features["wild_six_enabled"] = wild_six_enabled
            if bayes_opp is not None and episode_own_dice:
                opp_bid = state_features.get("current_bid")
                if opp_bid is not None:
                    opp_own_support = _count_face_support(episode_own_dice, opp_bid[1], wild_six_enabled)
                    bayes_opp.update(opp_bid, opp_own_support)
                    belief_state.update(opp_bid, opp_own_support)
                    momentum_tracker.update_opp_bid(_bid_rank(opp_bid))
            if not legal_action_map:
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                termination_reason = "no_legal_actions"
                break
            with _ROLLOUT_STATE["generation_semaphore"]:
                try:
                    rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]
                except Exception as e:
                    print(f"Warning: vLLM error at turn {turn_number} (game {game_id}): {type(e).__name__}: {e}")
                    termination_reason = "vllm_error"
                    done = True
                    break
            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            if include_action_mask:
                if len(prompt_ids) > MAX_PROMPT_LEN:
                    termination_reason = "max_prompt_len_exceeded"
                    break
                if turn_number == 0:
                    episode_prompt_ids = prompt_ids
                    prev_full_ids = prompt_ids.copy()
                else:
                    if prev_full_ids is None:
                        prev_full_ids = prompt_ids.copy()
                    elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                        prev_full_ids = prompt_ids.copy()
                    else:
                        delta_prompt_ids = prompt_ids[len(prev_full_ids):]
                        if delta_prompt_ids:
                            episode_completion_ids.extend(delta_prompt_ids)
                            episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                            episode_action_mask.extend([0] * len(delta_prompt_ids))
                        prev_full_ids = prompt_ids.copy()
                if completion_ids:
                    episode_completion_ids.extend(completion_ids)
                    episode_logprobs.extend(logprobs)
                    episode_action_mask.extend([1] * len(completion_ids))
                    if prev_full_ids is not None:
                        prev_full_ids = prev_full_ids + completion_ids
            else:
                prompt_ids_last = prompt_ids
                completion_ids_last = completion_ids
                logprobs_last = logprobs
            messages.append({"role": "assistant", "content": completion_text})
            _ROLLOUT_STATE["_current_state_features"] = state_features
            action_to_send = _parse_action_id(completion_text, legal_action_map, ruleset=ruleset)
            parse_failed = not action_to_send
            action_label = legal_action_map.get(action_to_send, "")
            liar_action = _is_liar_label(action_label)
            parsed_bid = _extract_bid_tuple(action_label)
            if parse_failed or action_to_send not in legal_action_map:
                invalid_count += 1
                consecutive_invalids += 1
                # --- Per-step: hard invalid penalty ---
                step_rewards.append(INVALID_STEP_PENALTY)
                last_action_prob = 0.0
                action_to_send = _select_fallback_action(legal_action_map, state_features, ruleset=ruleset)
                action_label = legal_action_map.get(action_to_send, "")
                liar_action = _is_liar_label(action_label)
                parsed_bid = _extract_bid_tuple(action_label)
            else:
                consecutive_invalids = 0
                # --- Per-step probability scoring ---
                step_score = 0.0
                action_prob = 0.0
                own_dice = state_features.get("own_dice") or episode_own_dice
                total_dice_now = int(state_features.get("total_dice") or 0)

                if parsed_bid is not None and total_dice_now > 0 and own_dice:
                    # BID action: score = f(P(bid is true))
                    bid_qty, bid_face = parsed_bid
                    action_prob = _bid_probability(bid_qty, bid_face, own_dice, total_dice_now, wild_six_enabled)
                    step_score = _probability_to_score(action_prob)

                    # Track bluffs
                    if action_prob < BLUFF_PROB_THRESHOLD:
                        bluff_count += 1

                    # Legacy: update CFR/momentum for analysis
                    br = _bid_rank(parsed_bid)
                    legal_brs = [
                        _bid_rank(_extract_bid_tuple(lbl))
                        for lbl in legal_action_map.values()
                        if _extract_bid_tuple(lbl) is not None
                    ]
                    if legal_brs:
                        episode_bid_ranks_played.append(br)
                        episode_legal_bid_ranks = list(set(episode_legal_bid_ranks + legal_brs))
                        if cfr_table is not None and episode_own_dice:
                            cfr_table.cfr_shaping(episode_own_dice, br, legal_brs)
                        momentum_tracker.update_our_bid(br)

                elif liar_action and state_features.get("current_bid") is not None:
                    # LIAR action: score = f(P(bid is false)) = f(1 - P(bid is true))
                    cb = state_features["current_bid"]
                    cb_qty, cb_face = cb
                    cb_own_support = _count_face_support(own_dice, cb_face, wild_six_enabled) if own_dice else 0
                    p_bid_true = _bid_probability(cb_qty, cb_face, own_dice, total_dice_now, wild_six_enabled) if total_dice_now > 0 and own_dice else 0.5
                    action_prob = 1.0 - p_bid_true  # P(calling liar is correct)
                    step_score = _probability_to_score(action_prob)

                    # Track risky liar calls
                    if RISKY_LIAR_PROB_MIN <= p_bid_true <= RISKY_LIAR_PROB_MAX:
                        risky_liar_count += 1

                elif ruleset == RULESET_LIARS_DIE:
                    # Liar's die variant: use legacy shaping
                    _ld_kind, _ld_rank = _liars_die_parse_action(action_label)
                    _ld_bid_s, _ld_dec_s, _ld_meta = _liars_die_compute_shaping(state_features, _ld_kind, _ld_rank)
                    action_prob = _ld_meta.get("current_bid_truth_probability", 0.5)
                    step_score = _probability_to_score(max(action_prob, 0.15))

                else:
                    step_score = 0.15
                    action_prob = 0.5

                last_action_prob = action_prob
                step_rewards.append(step_score)

            if parsed_bid is not None:
                bid_history.append({"turn": turn_number, "action_id": action_to_send, "label": action_label, "is_ours": True})
            elif liar_action:
                bid_history.append({"turn": turn_number, "action_id": action_to_send, "label": "LIAR", "is_ours": True})

            try:
                formatted_observation, step_reward, done, last_step_block = _step_environment(env_endpoint=env_endpoint, episode_id=episode_id, action_to_send=action_to_send, timeout=timeout)
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = ""
                step_reward = -0.01
                done = False
                invalid_count += 1
                consecutive_invalids += 1
                step_rewards.append(INVALID_STEP_PENALTY)
                last_step_block = {"reward": step_reward, "done": False}
            observation_lower = formatted_observation.lower()
            invalid_or_noop = "invalid" in observation_lower or "nothing happens" in observation_lower or "nothing happened" in observation_lower or action_to_send not in legal_action_map
            if invalid_or_noop:
                invalid_count += 1
                consecutive_invalids += 1
            else:
                consecutive_invalids = 0
            if formatted_observation == observation_before_action:
                noop_count += 1
                consecutive_noops += 1
            else:
                consecutive_noops = 0
            if done:
                final_reward = _extract_terminal_reward(last_step_block, formatted_observation)
                termination_reason = "done"
            else:
                next_state_features = _extract_state_features(formatted_observation)
                next_bid = next_state_features.get("current_bid")
                next_own_support = (
                    _count_face_support(episode_own_dice, next_bid[1], wild_six_enabled)
                    if next_bid and episode_own_dice else 0
                )
                history_summary = _bid_history_summary(bid_history)
                dice_context = _compute_dice_context_summary(next_state_features, next_bid)
                bayes_context = ""
                if bayes_opp is not None and episode_own_dice:
                    bayes_context = bayes_opp.full_context_summary(episode_own_dice)
                kelly_context = kelly_bluff_context(next_state_features, next_bid, bayes_opp)
                belief_context = belief_state.context_summary(next_bid, own_support=next_own_support)
                momentum_context = momentum_tracker.context_summary()
                opp_style_context = _classify_opponent_style(momentum_tracker)
                next_total_dice = int(next_state_features.get("total_dice") or 0)
                is_first_bidder = next_bid is None
                position_context = (
                    f"[Position] You bid FIRST this round (total dice: {next_total_dice}) — set the tone based on your hand."
                    if is_first_bidder else
                    f"[Position] You bid SECOND this round (total dice: {next_total_dice}) — use opponent's bid as information."
                )
                # --- Per-turn hybrid contrastive ---
                contrastive_context = ""
                if next_bid and next_total_dice > 0:
                    bid_ratio = next_bid[0] / max(next_total_dice, 1)
                    if bid_ratio >= 0.50:
                        contrastive_context = (
                            f"[Contrastive] Bid claims {next_bid[0]}/{next_total_dice} dice ({bid_ratio:.0%}) "
                            "-> VERY likely bluff. GOOD: Call Liar now. BAD: Raise into extreme bluff zone."
                        )
                    elif bid_ratio >= 0.40:
                        contrastive_context = (
                            f"[Contrastive] Bid is borderline ({bid_ratio:.0%} of dice). "
                            "GOOD: Call if your support is low. BAD: Blindly raise without dice support."
                        )
                if next_total_dice > 0 and next_total_dice <= 4 and not contrastive_context:
                    contrastive_context = (
                        f"[Contrastive] ENDGAME ({next_total_dice} dice left). "
                        "GOOD: Challenge aggressively, bids are transparent. BAD: Keep bidding conservatively."
                    )

                # --- RAG context injection ---
                rag_ctx = ""
                dice_rag = _ROLLOUT_STATE.get("dice_rag")
                if dice_rag:
                    rag_ctx, _ = dice_rag.retrieve(next_state_features)

                addendum_parts = [p for p in [
                    history_summary, dice_context, bayes_context,
                    kelly_context, belief_context, momentum_context,
                    opp_style_context, position_context, contrastive_context, rag_ctx,
                ] if p]
                obs_with_context = formatted_observation + "\n\n" + "\n".join(addendum_parts) if addendum_parts else formatted_observation
                # --- Action shuffle on subsequent observations ---
                next_action_map = _extract_legal_action_map(obs_with_context)
                obs_with_context = _shuffle_observation_actions(obs_with_context, next_action_map)
                messages.append({"role": "user", "content": obs_with_context})
                _MAX_HIST = 6
                if len(messages) > 2 + _MAX_HIST * 2:
                    messages = messages[:2] + messages[-(_MAX_HIST * 2):]
            step_records.append({"turn": turn_number, "ruleset": ruleset, "parsed_action": action_to_send, "action_label": action_label, "step_score": float(step_rewards[-1]) if step_rewards else 0.0, "action_prob": float(last_action_prob), "done": bool(done), "invalid_or_noop": invalid_or_noop, "parse_failed": bool(parse_failed)})
            turn_number += 1

        # ============================================================
        # REWARD CALCULATION: γ-discounted per-step return + terminal
        # ============================================================
        if not done:
            if termination_reason == "unknown":
                termination_reason = "max_turn_reached"
            final_reward = 0.0

        # Terminal reward scaled by TERMINAL_WEIGHT
        env_reward = final_reward  # raw 0/1/-1 from environment
        terminal_component = env_reward * TERMINAL_WEIGHT

        # γ-discounted sum of per-step scores
        discounted_sum = 0.0
        for i, sr in enumerate(step_rewards):
            discounted_sum += sr * (GAMMA ** i)

        # Risky-move bonuses (only on WIN)
        risky_bonus = 0.0
        if env_reward > 0:
            risky_moves = min(bluff_count + risky_liar_count, RISKY_BONUS_MAX_COUNT)
            risky_bonus = risky_moves * BLUFF_WIN_BONUS

        train_reward = discounted_sum + terminal_component + risky_bonus

        # CFR table update (legacy, for analysis)
        if cfr_table is not None and episode_own_dice and episode_bid_ranks_played and episode_legal_bid_ranks:
            last_bid_rank = episode_bid_ranks_played[-1]
            cfr_table.update(own_dice=episode_own_dice, bid_rank_played=last_bid_rank, legal_bid_ranks=episode_legal_bid_ranks, episode_reward=env_reward)
        if endgame_cfr is not None and episode_own_dice and episode_bid_ranks_played and episode_legal_bid_ranks:
            endgame_cfr.update(
                own_dice=episode_own_dice,
                total_dice=episode_last_bid_total_dice,
                current_bid=None,
                action_taken=episode_bid_ranks_played[-1],
                legal_bid_ranks=episode_legal_bid_ranks,
                episode_reward=env_reward,
            )

        # Winner-matching log format
        print(f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} | Reward: {train_reward:7.2f} | LastProb: {last_action_prob:6.3f} | EnvR: {env_reward:4.1f} | Bluffs:{bluff_count}  RiskyLiar:{risky_liar_count}  Inv:{invalid_count} ]")

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "game_id": game_id, "episode_id": episode_id, "environment": "liars_dice", "ruleset": ruleset, "status": "completed" if done else "truncated", "termination_reason": termination_reason, "turns": turn_number, "env_reward": float(env_reward), "discounted_sum": float(discounted_sum), "terminal_component": float(terminal_component), "risky_bonus": float(risky_bonus), "train_reward": float(train_reward), "invalid_count": invalid_count, "bluff_count": bluff_count, "risky_liar_count": risky_liar_count, "last_action_prob": float(last_action_prob), "steps": step_records})
        if include_action_mask:
            if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
                episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
                episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
                episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]
            return index, {"prompt_ids": episode_prompt_ids, "completion_ids": episode_completion_ids, "action_mask": episode_action_mask, "logprobs": episode_logprobs, "reward": train_reward, "final_score": env_reward}
        return index, {"prompt_ids": prompt_ids_last, "completion_ids": completion_ids_last, "logprobs": logprobs_last, "reward": train_reward, "final_score": env_reward}

    executor = _ROLLOUT_STATE["thread_pool"]
    fallback_builder = _full_prompt_fallback_result if include_action_mask else _last_prompt_fallback_result
    list_results = _execute_parallel_rollouts(prompts=prompts, executor=executor, run_single_prompt=run_single_prompt, fallback_builder=fallback_builder)
    curriculum.step(len(prompts))
    _log_batch_statistics(list_results)
    if include_action_mask:
        return {"prompt_ids": [r["prompt_ids"] for r in list_results], "completion_ids": [r["completion_ids"] for r in list_results], "action_mask": [r["action_mask"] for r in list_results], "logprobs": [r["logprobs"] for r in list_results], "env_rewards": [r["reward"] for r in list_results]}
    return {"prompt_ids": [r["prompt_ids"] for r in list_results], "completion_ids": [r["completion_ids"] for r in list_results], "logprobs": [r["logprobs"] for r in list_results], "env_rewards": [r["reward"] for r in list_results]}

def rollout_last_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=30):
    del max_turns
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=False)

def rollout_full_prompt_and_completion_parallelized_curriculum(prompts, trainer, max_turns=30):
    del max_turns
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=True)

def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
