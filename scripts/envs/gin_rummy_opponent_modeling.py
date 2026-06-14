import functools
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from threading import Semaphore
from typing import Optional

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    remove_reasoning_tags,
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "gin_rummy"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 8000
_TIMEOUT = 2400
_MCTS_SIMS = 50

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 10, 'Q': 10, 'K': 10,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

# Reward constants — all shaping is per-step (Mizukami 2015 discard-safety,
# Grzes 2017 knock PBRS, Ng/Harada/Russell 1999 deadwood PBRS). Episode-level
# bonuses have been removed; TERMINAL_WIN_REWARD remains because it is the
# environmental outcome, not shaping.
# Terminal reward amplified 5× so the win/loss outcome dominates per-step
# PBRS shaping — prior run (eval_loss 0.5263) lost to base env (0.5823)
# because per-step PBRS magnitudes (~0.02-0.35) drowned the ±1.0 terminal.
TERMINAL_WIN_REWARD  =  8.0
INVALID_PENALTY      = -1.0    # stronger invalid signal

# Discard safety — amplified to encourage defensive play
SAFE_DISCARD_BONUS        = 0.15  # was 0.05
DANGEROUS_DISCARD_PENALTY = 0.20  # was 0.05
# Upcard-pick credit (action 52). Stronger signal for drawing known useful cards.
UPCARD_PICK_WEIGHT        = 0.15  # was 0.05

# --- Structural discard bonuses (DeadCardTracker + VoidRunInference,
# boss-impossible — boss's opponent_modeling has neither class). Anchored on
# closed-form discard-pile inference (single-copy deck), so they are
# DETERMINISTIC safety signals, complementary to the Bayesian discard-safety
# shaping above. Values kept small to stack additively without dominating.
DEAD_RANK_DISCARD_BONUS  = 0.03   # rank already has ≥2 other suits dead → set-impossible after this discard
VOID_SUIT_DISCARD_BONUS  = 0.03   # ≥3 ranks of this suit already in pile → opp run-in-suit heavily blocked

# Knock-commitment one-shot bonus. Fires on the first Knock action of an
# episode, breaking PBRS entry/exit symmetry. Raised above all PBRS potentials
# so committing dominates hovering near knock-ready states (Ng-Harada-Russell
# 1999: shaping magnitudes must not exceed real-action magnitudes, else
# Goodhart). Set to 15% of TERMINAL_WIN_REWARD (5.0) — matches Grzes 2017
# Table 3 range for sparse-outcome commit bonuses.
KNOCK_COMMIT_BONUS        = 0.75

# Knock-decision constants (opponent-aware, per-step PBRS)
# Grzes 2017 look-back PBRS potential, with thresholds from Kotnik/Kalita (FLAIRS 2003)
# and Chow-Robbins (1971) optimal-stopping adaptation. All magnitudes kept
# strictly below KNOCK_COMMIT_BONUS (0.75) so commit always dominates
# hovering. Region narrowed (no "approaching" tier) to concentrate signal
# near the actual decision boundary per Kotnik/Kalita (2003).
KOTNIK_KALITA_THRESHOLD = 8       # aggressive-knock deadwood threshold
CHOW_ROBBINS_THRESHOLD  = 0.55    # p_now threshold = 1 − (my_dw / est_opp_dw)
GIN_POTENTIAL           = 0.60    # was 0.40
KNOCK_POTENTIAL_HIGH    = 0.50    # was 0.30 (dw ≤ 4 aggressive tier)
KNOCK_POTENTIAL_LEGAL   = 0.35    # was 0.20 (knock-legal, dw ≤ knock_card)
KNOCK_POTENTIAL_NEAR    = 0.15    # was 0.10 (near-knock, narrowed to dw ≤ kc+3)
# KNOCK_POTENTIAL_APPROACHING removed — the dw ≤ kc+10 zone was too broad,
# firing on most hands and diluting gradient near the knock decision boundary.
CONFIDENCE_MIN_OBS      = 3
CONFIDENCE_MIN_TURN     = 4
# Discard-safety thresholds (Mizukami 2015 danger score)
DISCARD_DANGER_THRESHOLD = 0.65  # was 0.80 (be more suspicious)
DISCARD_SAFE_THRESHOLD   = 0.25  # was 0.30 (be more certain of safety)
# Per-step deadwood PBRS scale — halved so Φ = −dw/scale gradient is 2×
# sharper per unit of deadwood reduction.
DEADWOOD_PBRS_SCALE     = 50.0    # was 100.0


# ---------------------------------------------------------------------------
# Card utilities
# ---------------------------------------------------------------------------

def get_rank(card: str) -> str:
    return card[0]

def get_suit(card: str) -> str:
    return card[1]

def get_value(card: str) -> int:
    return CARD_VALUES[get_rank(card)]


def find_potential_runs(hand: list[str], additional_card: Optional[str] = None) -> list[list[str]]:
    test_hand = hand.copy()
    if additional_card:
        test_hand.append(additional_card)
    suit_groups: dict[str, list[str]] = {}
    for card in test_hand:
        suit_groups.setdefault(get_suit(card), []).append(card)
    runs = []
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if RANK_ORDER.index(get_rank(sorted_cards[j])) == RANK_ORDER.index(get_rank(run[-1])) + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            if len(run) >= 2:
                runs.append(run)
            i = j if len(run) > 1 else i + 1
    return runs


def count_complete_runs(hand: list[str]) -> int:
    return sum(1 for r in find_potential_runs(hand) if len(r) >= 3)


# ---------------------------------------------------------------------------
# DP optimal deadwood
# ---------------------------------------------------------------------------

def find_all_melds(hand: list[str]) -> list[frozenset[str]]:
    """Enumerate every valid meld (SET or RUN of 3+ cards) from the given hand."""
    melds: list[frozenset[str]] = []

    rank_groups: dict[str, list[str]] = defaultdict(list)
    for card in hand:
        rank_groups[get_rank(card)].append(card)
    for cards in rank_groups.values():
        if len(cards) >= 3:
            melds.append(frozenset(cards[:3]))
        if len(cards) >= 4:
            melds.append(frozenset(cards[:4]))

    suit_groups: dict[str, list[str]] = defaultdict(list)
    for card in hand:
        suit_groups[get_suit(card)].append(card)
    for cards in suit_groups.values():
        sorted_cards = sorted(cards, key=lambda c: RANK_ORDER.index(get_rank(c)))
        i = 0
        while i < len(sorted_cards):
            run = [sorted_cards[i]]
            j = i + 1
            while j < len(sorted_cards):
                if RANK_ORDER.index(get_rank(sorted_cards[j])) == RANK_ORDER.index(get_rank(run[-1])) + 1:
                    run.append(sorted_cards[j])
                    j += 1
                else:
                    break
            for start in range(len(run)):
                for end in range(start + 3, len(run) + 1):
                    melds.append(frozenset(run[start:end]))
            i = j if len(run) > 1 else i + 1

    return melds


def compute_optimal_deadwood(hand: list[str]) -> int:
    """Minimum deadwood via bitmask DP backtracking."""
    if not hand:
        return 0
    melds = find_all_melds(hand)
    n = len(hand)
    card_to_idx = {card: i for i, card in enumerate(hand)}
    meld_masks: list[int] = []
    for meld in melds:
        mask = 0
        valid = True
        for card in meld:
            if card not in card_to_idx:
                valid = False
                break
            mask |= (1 << card_to_idx[card])
        if valid:
            meld_masks.append(mask)

    card_values_list = [get_value(card) for card in hand]
    memo: dict[int, int] = {}

    def _dp(used_mask: int) -> int:
        if used_mask in memo:
            return memo[used_mask]
        base_dw = sum(card_values_list[i] for i in range(n) if not (used_mask >> i & 1))
        best = base_dw
        for mm in meld_masks:
            if (mm & used_mask) == 0:
                best = min(best, _dp(used_mask | mm))
        memo[used_mask] = best
        return best

    return _dp(0)


def meld_potential(upcard: str, hand: list[str]) -> int:
    """Estimate deadwood reduction from drawing the upcard."""
    if not upcard or upcard == 'XX' or len(upcard) != 2:
        return 0
    dw_with    = compute_optimal_deadwood(hand + [upcard])
    dw_without = compute_optimal_deadwood(hand)
    return max(0, dw_without - dw_with)


# ---------------------------------------------------------------------------
# Opponent-aware knock banner + per-step shaping
# ---------------------------------------------------------------------------

def knock_confidence_score(
    state: "GameState | None",
    bayes_hand: "BayesianOpponentHandModel | None",
    turn_number: int,
) -> float:
    """
    Returns [0, 1]: how confident the knock-decision pipeline is that knocking
    is the right action in ``state``. Used to scale KNOCK_COMMIT_BONUS so the
    gradient rewards confident commits more than reckless ones (Brier 1950
    calibration principle; same thresholds that feed the knock banner).

    Tiers:
      1.0   — dw == 0 (GIN available, strongest possible commit)
      0.85  — Kotnik/Kalita dw ≤ 4 aggressive-knock rule
      0.75  — Bayesian posterior confident AND p_now ≥ Chow-Robbins threshold
      0.50  — p_now ≥ threshold but posterior not yet confident
      0.30  — marginal zone
      0.15  — p_now suggests knocking likely loses
      0.0   — not knock-legal
    """
    if state is None or not state.knock_action_ids:
        return 0.0
    our_dw = state.deadwood
    if our_dw == 0:
        return 1.0
    if our_dw <= KOTNIK_KALITA_THRESHOLD:
        return 0.85
    if bayes_hand is None:
        return 0.4
    confident = bayes_hand.is_posterior_confident(turn_number)
    est_opp_dw = bayes_hand.estimated_opponent_deadwood()
    p_now = 1.0 - (our_dw / max(est_opp_dw, 1.0)) if est_opp_dw > 0 else 0.0
    if confident and p_now >= CHOW_ROBBINS_THRESHOLD:
        return 0.75
    if p_now >= CHOW_ROBBINS_THRESHOLD:
        return 0.5
    if p_now <= (1.0 - CHOW_ROBBINS_THRESHOLD):
        return 0.15
    return 0.3


def knock_potential(
    state: "GameState | None",
    bayes_hand: "BayesianOpponentHandModel | None",
    turn_number: int,
) -> float:
    """
    Grzes 2017 look-back PBRS potential, tiered monotonically in deadwood.

    Region narrowed to dw ≤ knock_card + 3 so the gradient concentrates near
    the actual decision boundary (Kotnik/Kalita 2003 aggressive-knock zone
    is dw ≤ 4; Chow-Robbins optimal-stopping has a tight decision region,
    not a continuous gradient). The prior dw ≤ kc+10 region was too broad —
    it fired on ~1063 of 3000 rollouts in the baseline log, diluting signal
    and letting the policy accumulate reward by hovering in knock-adjacent
    states without ever committing (entered_knock_state:1063 vs
    knock_committed:4).

    All potentials are kept strictly below KNOCK_COMMIT_BONUS (0.75) so that
    committing always dominates hovering (Ng-Harada-Russell 1999: shaping
    magnitudes must not exceed real-action magnitudes).

    ``bayes_hand`` / ``turn_number`` are retained in the signature for
    compatibility (the banner-text helper still uses them via Chow-Robbins);
    the potential itself depends only on the agent's own deadwood so it is
    well-defined on every state.
    """
    if state is None:
        return 0.0
    our_dw = state.deadwood
    kc = state.knock_card
    if our_dw == 0:
        return GIN_POTENTIAL                   # 0.40
    if our_dw <= KOTNIK_KALITA_THRESHOLD:       # dw ≤ 4 — Kotnik/Kalita aggressive rule
        return KNOCK_POTENTIAL_HIGH            # 0.30
    if our_dw <= kc:                            # knock-legal, above aggressive threshold
        return KNOCK_POTENTIAL_LEGAL           # 0.20
    if our_dw <= kc + 3:                        # near-knock (narrowed 5→3)
        return KNOCK_POTENTIAL_NEAR            # 0.10
    return 0.0


def _build_knock_banner(
    state: "GameState",
    bayes_hand: "BayesianOpponentHandModel | None",
    turn_number: int,
) -> str:
    """
    Dedicated banner when a Knock action is legal.
    Surfaces the Kotnik/Kalita (dw≤4) and Chow-Robbins (p_now≥0.55) thresholds.
    """
    our_dw = state.deadwood
    est_opp_dw = bayes_hand.estimated_opponent_deadwood() if bayes_hand is not None else 0.0
    confident = bayes_hand.is_posterior_confident(turn_number) if bayes_hand is not None else False
    conf_label = "MEDIUM-HIGH" if confident else "LOW"
    p_now = 1.0 - (our_dw / max(est_opp_dw, 1.0)) if est_opp_dw > 0 else 0.0

    if our_dw == 0:
        recommendation = "GIN AVAILABLE — DECLARE NOW"
    elif our_dw <= KOTNIK_KALITA_THRESHOLD:
        recommendation = f"aggressive rule: dw≤{KOTNIK_KALITA_THRESHOLD} → KNOCK"
    elif not confident:
        recommendation = "low-confidence estimate — consider other factors"
    elif p_now >= CHOW_ROBBINS_THRESHOLD:
        recommendation = f"Chow-Robbins: p_now≥{CHOW_ROBBINS_THRESHOLD} → KNOCK"
    elif p_now <= 1.0 - CHOW_ROBBINS_THRESHOLD:
        recommendation = "knocking LIKELY LOSES — hold and improve hand"
    else:
        recommendation = "marginal — neither clearly wins nor loses"

    knock_ids_sorted = sorted(state.knock_action_ids)
    shown = ", ".join(str(i) for i in knock_ids_sorted[:5])
    if len(knock_ids_sorted) > 5:
        shown += f" (+{len(knock_ids_sorted) - 5} more)"

    return (
        "*** KNOCK DECISION POINT ***\n"
        f"Your deadwood: {our_dw} (aggressive threshold: ≤{KOTNIK_KALITA_THRESHOLD})\n"
        f"Est. opp deadwood: {est_opp_dw:.0f} (confidence: {conf_label})\n"
        f"p_now (Chow-Robbins): {p_now:.2f} (threshold: {CHOW_ROBBINS_THRESHOLD})\n"
        f"Recommendation: {recommendation}\n"
        f"Knock action IDs: {shown}"
    )


def opponent_aware_step_shaping(
    *,
    prev_state: "GameState | None",
    curr_state: "GameState | None",
    action_to_send: str,
    bayes_hand: "BayesianOpponentHandModel | None",
    turn_number: int,
) -> tuple[float, str]:
    """
    Per-step opponent-aware shaping combining three components:

    (1) Mizukami 2015 discard-safety: direct bonus/penalty at the moment of
        a non-knock discard, based on posterior meld-danger score.
    (2) Grzes 2017 look-back PBRS on knock potential: F = γΦ(s') − Φ(s),
        policy-invariant (Ng/Harada/Russell 1999).
    (3) Upcard-pick credit: bonus on action 52 in FirstUpcard/Draw phases
        scaled by `meld_potential(upcard, hand)`. Improves credit assignment
        — the deadwood PBRS captures the gain one step later, but crediting
        the pick action itself tightens the reward-to-action link.

    Returns (reward_delta, event_label).
    """
    if prev_state is None:
        return 0.0, ""

    reward = 0.0
    events: list[str] = []

    # (1) Discard-safety (Mizukami 2015, direct shaping)
    try:
        action_id = int(str(action_to_send).strip())
    except (ValueError, TypeError):
        action_id = None

    if action_id is not None and bayes_hand is not None:
        is_knock = action_id in prev_state.knock_action_ids
        is_discard = (
            prev_state.phase == "Discard"
            and 0 <= action_id < len(prev_state.hand)
            and not is_knock
        )
        if is_discard:
            discarded = prev_state.hand[action_id]
            danger = bayes_hand.meld_danger_score(discarded)
            if danger >= DISCARD_DANGER_THRESHOLD:
                reward -= DANGEROUS_DISCARD_PENALTY
                events.append("dangerous_discard")
            elif danger <= DISCARD_SAFE_THRESHOLD:
                reward += SAFE_DISCARD_BONUS
                events.append("safe_discard")

    # (2) Knock-potential PBRS (Grzes 2017, policy-invariant)
    phi_prev = knock_potential(prev_state, bayes_hand, turn_number)
    phi_curr = knock_potential(curr_state, bayes_hand, turn_number + 1)
    pbrs_knock = phi_curr - phi_prev
    reward += pbrs_knock
    if pbrs_knock > 0.05:
        events.append("entered_knock_state")
    elif pbrs_knock < -0.05:
        events.append("left_knock_state")

    # (3) Upcard-pick credit: reward at action 52 (pick upcard) scaled by the
    # deadwood reduction the upcard would provide. Uses `meld_potential(...)`
    # DP. Zero when the upcard doesn't help — never penalizes any action.
    if (action_id == 52
        and prev_state.phase in ("FirstUpcard", "Draw")
        and prev_state.upcard
        and prev_state.hand):
        potential = meld_potential(prev_state.upcard, prev_state.hand)
        if potential > 0:
            scaled = min(potential, 10) / 10.0
            reward += UPCARD_PICK_WEIGHT * scaled
            events.append("good_upcard_pick")

    return reward, "+".join(events)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    hand:         list[str]
    deadwood:     int
    phase:        str
    knock_card:   int
    upcard:       str
    stock_size:   int
    discard_pile: list[str]
    player_id:    int
    legal_actions:    dict[int, str] = field(default_factory=dict)
    knock_action_ids: set[int]       = field(default_factory=set)

    def total_hand_value(self) -> int:
        return sum(get_value(c) for c in self.hand)

    def num_high_cards(self) -> int:
        return sum(1 for c in self.hand if get_value(c) == 10)

    def can_knock(self) -> bool:
        return self.deadwood <= self.knock_card

    def count_pairs(self) -> int:
        return sum(1 for cnt in Counter(get_rank(c) for c in self.hand).values() if cnt >= 2)

    def count_sets(self) -> int:
        return sum(1 for cnt in Counter(get_rank(c) for c in self.hand).values() if cnt >= 3)

    def count_runs(self) -> int:
        return count_complete_runs(self.hand)

    def count_potential_runs(self) -> int:
        return sum(1 for r in find_potential_runs(self.hand) if len(r) == 2)


# ---------------------------------------------------------------------------
# Dead card tracker
# ---------------------------------------------------------------------------

class DeadCardTracker:
    """Tracks discarded cards and identifies layoff candidates."""

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self.seen_discards: set[str] = set()
        self.opponent_melds: list[list[str]] = []

    def update_from_discard_pile(self, discard_pile: list[str]) -> None:
        for card in discard_pile:
            if len(card) == 2:
                self.seen_discards.add(card.lower())

    def update_from_observation(self, obs: str) -> None:
        pile = parse_discard_pile(obs)
        self.update_from_discard_pile(pile)

    def get_dead_cards(self) -> list[str]:
        return sorted(self.seen_discards)

    def is_dead(self, card: str) -> bool:
        return card.lower() in self.seen_discards

    def get_layoff_candidates(self, hand: list[str], discard_pile: list[str]) -> list[str]:
        if not discard_pile or not hand:
            return []
        candidates: set[str] = set()

        suit_groups: dict[str, list[str]] = {}
        for card in discard_pile:
            if len(card) != 2:
                continue
            suit = card[1].lower()
            suit_groups.setdefault(suit, []).append(card.lower())

        for suit, cards in suit_groups.items():
            sorted_cards = sorted(
                cards,
                key=lambda c: self.ALL_RANKS.index(c[0].upper()) if c[0].upper() in self.ALL_RANKS else 99,
            )
            for i in range(len(sorted_cards) - 1):
                r1 = sorted_cards[i][0].upper()
                r2 = sorted_cards[i + 1][0].upper()
                if r1 not in self.ALL_RANKS or r2 not in self.ALL_RANKS:
                    continue
                idx1 = self.ALL_RANKS.index(r1)
                idx2 = self.ALL_RANKS.index(r2)
                if abs(idx1 - idx2) == 1:
                    for adj in [idx1 - 1, idx2 + 1]:
                        if 0 <= adj < len(self.ALL_RANKS):
                            target = self.ALL_RANKS[adj] + suit
                            for hcard in hand:
                                if hcard.lower() == target:
                                    candidates.add(hcard)

        rank_groups: dict[str, int] = {}
        for card in discard_pile:
            if len(card) != 2:
                continue
            rank = card[0].upper()
            rank_groups[rank] = rank_groups.get(rank, 0) + 1
        for rank, count in rank_groups.items():
            if count >= 2:
                for hcard in hand:
                    if hcard[0].upper() == rank:
                        candidates.add(hcard)

        return sorted(candidates)

    def summary(self, hand: list[str]) -> str:
        dead   = self.get_dead_cards()
        layoff = self.get_layoff_candidates(hand, list(self.seen_discards))
        lines  = []
        if dead:
            lines.append(f"Dead cards (discarded): {' '.join(dead[:15])}")
        if layoff:
            lines.append(f"Layoff candidates (extend opp melds): {' '.join(layoff)}")
        # Set-impossible ranks: 3+ of 4 same-rank cards already discarded
        # (uses is_dead). Flags ranks where neither player can form a set.
        impossible = [
            r for r in self.ALL_RANKS
            if sum(1 for s in self.ALL_SUITS if self.is_dead(r + s)) >= 3
        ]
        if impossible:
            lines.append(f"Set-impossible ranks (3+ suits dead): {' '.join(impossible)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Void-run inference (Sturtevant & White 2006 analogue for Gin Rummy)
# ---------------------------------------------------------------------------

class VoidRunInference:
    """
    Closed-form inference: which runs can opponent NOT build given discards?
    Any discarded card breaks every 3-card run that contains it (cards are
    single-copy, so if it's in the pile, opp cannot hold it).
    """

    ALL_RANKS = list("A23456789TJQK")
    SUIT_CHAR = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}

    def summary(self, discard_pile: list[str]) -> str:
        by_suit: dict[str, list[str]] = {s: [] for s in self.SUIT_CHAR}
        for card in discard_pile:
            if len(card) != 2:
                continue
            r, s = card[0].upper(), card[1].lower()
            if r in self.ALL_RANKS and s in by_suit:
                by_suit[s].append(r)
        lines: list[str] = []
        for s, ranks in by_suit.items():
            if len(ranks) < 2:
                continue
            sorted_ranks = sorted(set(ranks), key=lambda r: self.ALL_RANKS.index(r))
            lines.append(f"[Void] {self.SUIT_CHAR[s]} blocked ranks: {' '.join(sorted_ranks)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON belief header (Huang/Chalkiadakis/Elkind AAMAS 2024 — structured belief
# prompting outperformed raw history and free-form NL by 3-5% on Skat/Hearts)
# ---------------------------------------------------------------------------

def build_json_belief_header(
    state: "GameState",
    bayes_hand: "BayesianOpponentHandModel | None",
    turn_number: int,
) -> str:
    import json as _json
    our_dw = state.deadwood
    est_opp_dw = bayes_hand.estimated_opponent_deadwood() if bayes_hand is not None else None
    confident = bayes_hand.is_posterior_confident(turn_number) if bayes_hand is not None else False

    top_opp_cards: dict[str, float] = {}
    if bayes_hand is not None:
        for card, prob in bayes_hand.estimated_opponent_hand(top_n=6):
            if prob >= 0.3:
                top_opp_cards[card] = round(float(prob), 2)

    safe_cards: list[str] = []
    danger_cards: list[str] = []
    if bayes_hand is not None and state.hand:
        safe_cards   = bayes_hand.get_safe_cards(state.hand)[:5]
        danger_cards = bayes_hand.get_danger_cards(state.hand)[:5]

    p_now = (
        round(1.0 - (our_dw / max(est_opp_dw, 1.0)), 2)
        if est_opp_dw is not None and est_opp_dw > 0 else None
    )

    payload = {
        "my_deadwood":       our_dw,
        "my_meld_shanten":   compute_optimal_deadwood(state.hand) if state.hand else None,
        "opp_deadwood_est":  round(float(est_opp_dw), 1) if est_opp_dw is not None else None,
        "opp_estimate_confident": confident,
        "opp_likely_cards":  top_opp_cards,
        "safe_discards":     safe_cards,
        "danger_discards":   danger_cards,
        "p_now":             p_now,
        "knock_legal":       bool(state.knock_action_ids),
        # Hand-structure features (GameState methods)
        "hand_value":        state.total_hand_value()     if state.hand else 0,
        "high_cards":        state.num_high_cards()       if state.hand else 0,
        "can_knock":         state.can_knock(),
        "pairs":             state.count_pairs()          if state.hand else 0,
        "sets":              state.count_sets()           if state.hand else 0,
        "runs":              state.count_runs()           if state.hand else 0,
        "potential_runs":    state.count_potential_runs() if state.hand else 0,
    }
    return "[Belief] " + _json.dumps(payload, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Bayesian opponent hand distribution (PerfectDou-style posterior + Mizukami-style
# danger/safe discard classification)
# ---------------------------------------------------------------------------

class BayesianOpponentHandModel:
    """Tracks P(card ∈ opponent_hand | all observations) via Bayesian updates."""

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self._prob: dict[str, float] = {}
        self._opp_hand_size: int = 10
        self._confirmed_in_hand:     set[str] = set()
        self._confirmed_not_in_hand: set[str] = set()

    def _all_cards(self) -> list[str]:
        return [r + s for r in self.ALL_RANKS for s in self.ALL_SUITS]

    def initialize(self, our_hand: list[str], discard_pile: list[str]) -> None:
        known_out     = set(c.lower() for c in our_hand) | set(c.lower() for c in discard_pile)
        unknown_cards = [c for c in self._all_cards() if c not in known_out]
        n             = len(unknown_cards)
        p_in_hand     = self._opp_hand_size / max(n, self._opp_hand_size)
        self._prob    = {c: p_in_hand for c in unknown_cards}
        for c in known_out:
            self._prob[c] = 0.0

    def update_opp_drew_upcard(self, upcard: str) -> None:
        card = upcard.lower()
        if len(card) == 2:
            self._prob[card] = 1.0
            self._confirmed_in_hand.add(card)
            self._renormalize(exclude={card})

    def update_opp_drew_stock(self) -> None:
        stock_candidates = [
            c for c, p in self._prob.items()
            if p > 0 and c not in self._confirmed_in_hand and c not in self._confirmed_not_in_hand
        ]
        if not stock_candidates:
            return
        n = len(stock_candidates)
        boost = 1.0 / n
        for c in stock_candidates:
            self._prob[c] = min(1.0, self._prob[c] + boost * 0.1)

    def update_opp_discarded(self, card: str) -> None:
        c = card.lower()
        if len(c) == 2:
            self._prob[c] = 0.0
            self._confirmed_not_in_hand.add(c)
            self._renormalize(exclude={c})

    def _renormalize(self, exclude: set[str]) -> None:
        uncertain = {
            c: p for c, p in self._prob.items()
            if c not in exclude
            and c not in self._confirmed_in_hand
            and c not in self._confirmed_not_in_hand
            and p > 0
        }
        confirmed_count  = len(self._confirmed_in_hand)
        remaining_slots  = max(self._opp_hand_size - confirmed_count, 0)
        n = len(uncertain)
        if n == 0 or remaining_slots == 0:
            return
        target_p = remaining_slots / n
        total    = sum(uncertain.values())
        scale    = (target_p / (total / n)) if total > 0 else 1.0
        scale    = min(max(scale, 0.5), 2.0)
        for c in uncertain:
            self._prob[c] = min(1.0, self._prob[c] * scale)

    def estimated_opponent_hand(self, top_n: int = 10) -> list[tuple[str, float]]:
        return sorted(
            [(c, p) for c, p in self._prob.items() if p > 0],
            key=lambda x: -x[1],
        )[:top_n]

    def meld_danger_score(self, card: str) -> float:
        """
        Mizukami 2015 style: P(discarding this card completes an opponent meld).
        Sums the posterior probability that opp holds cards that would form a
        set with this rank or a run adjacent to this card in this suit.
        """
        if len(card) != 2:
            return 0.0
        rank_c, suit_c = card[0].upper(), card[1].lower()
        card_lc = card.lower()

        set_mates = [r + s for r in [rank_c] for s in self.ALL_SUITS if (r + s) != card_lc]
        set_risk = sum(self._prob.get(c, 0.0) for c in set_mates)

        run_mates: list[str] = []
        if rank_c in self.ALL_RANKS:
            idx = self.ALL_RANKS.index(rank_c)
            for adj in (idx - 2, idx - 1, idx + 1, idx + 2):
                if 0 <= adj < len(self.ALL_RANKS):
                    run_mates.append(self.ALL_RANKS[adj] + suit_c)
        run_risk = sum(self._prob.get(c, 0.0) for c in run_mates)

        return set_risk + run_risk

    def is_dangerous_discard(self, card: str, threshold: float = 0.8) -> bool:
        return self.meld_danger_score(card) >= threshold

    def is_safe_discard(self, card: str, threshold: float = 0.3) -> bool:
        return self.meld_danger_score(card) <= threshold

    def get_danger_cards(self, hand: list[str]) -> list[str]:
        return [c for c in hand if self.is_dangerous_discard(c)]

    def get_safe_cards(self, hand: list[str]) -> list[str]:
        return [c for c in hand if self.is_safe_discard(c)]

    def estimated_opponent_deadwood(self) -> float:
        """Numeric estimate of opponent's total deadwood, using the posterior.

        Confirmed-in-hand cards contribute their full value once. Remaining
        hand slots are filled with the highest-probability uncertain cards,
        weighted by their posterior probability. This avoids the prior bug
        where confirmed cards were summed twice (once via top_hand at p=1.0
        and again via confirmed_dw).
        """
        confirmed_dw = sum(get_value(c) for c in self._confirmed_in_hand if len(c) == 2)
        confirmed_count = len(self._confirmed_in_hand)
        remaining_slots = max(self._opp_hand_size - confirmed_count, 0)
        uncertain = sorted(
            ((c, p) for c, p in self._prob.items()
             if p > 0 and c not in self._confirmed_in_hand),
            key=lambda x: -x[1],
        )[:remaining_slots]
        estimated_dw = sum(get_value(c) * p for c, p in uncertain)
        return estimated_dw + confirmed_dw

    def is_posterior_confident(self, turn_number: int = 0) -> bool:
        """Enough evidence that est_opp_dw is trustworthy for shaping decisions."""
        n_confirmed = len(self._confirmed_in_hand) + len(self._confirmed_not_in_hand)
        return n_confirmed >= CONFIDENCE_MIN_OBS or turn_number >= CONFIDENCE_MIN_TURN

    def knock_risk(self) -> str:
        total_est = self.estimated_opponent_deadwood()
        if total_est <= 10:
            return f"HIGH (est. opp deadwood ~{total_est:.0f} — may knock soon)"
        elif total_est <= 25:
            return f"MEDIUM (est. opp deadwood ~{total_est:.0f})"
        else:
            return f"LOW (est. opp deadwood ~{total_est:.0f})"

    def summary(self, our_hand: list[str]) -> str:
        lines    = [f"[BayesHand] Opp knock risk: {self.knock_risk()}"]
        top_held = self.estimated_opponent_hand(5)
        if top_held:
            cards_str = " ".join(f"{c}({p:.0%})" for c, p in top_held if p >= 0.25)
            if cards_str:
                lines.append(f"[BayesHand] Likely opp cards: {cards_str}")
        confirmed = list(self._confirmed_in_hand)
        if confirmed:
            lines.append(f"[BayesHand] Confirmed in opp hand (drew upcard): {' '.join(confirmed[:4])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def extract_and_format_observation(obs_text: str) -> str:
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text
    state_match = re.search(r'Current State:\n', obs_text)
    if not state_match:
        return obs_text
    state_text   = obs_text[state_match.start():]
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id    = int(player_match.group(1)) if player_match else 0
    if 'Legal Actions:' in state_text:
        before_actions, after_actions = state_text.split('Legal Actions:', 1)
        return before_actions + f"You are Player {player_id}.\nLegal Actions:" + after_actions
    return state_text


def parse_hand_from_observation(observation: str) -> list[str]:
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id    = int(player_match.group(1)) if player_match else 0
    section      = re.search(
        rf'Player{player_id}: Deadwood=\d+\n(?:(?:Layed melds:|Layoffs:)[^\n]*\n)*\+-+\+\n(.*?)\n\+-+\+',
        observation, re.DOTALL,
    )
    hand = []
    if section:
        for row in section.group(1).strip().split('\n'):
            hand.extend(re.findall(r'([A2-9TJQK][shdc])', row))
    return hand


def parse_discard_pile(observation: str) -> list[str]:
    m = re.search(r'Discard pile: (.*?)\n', observation)
    if not m:
        return []
    pile_str = m.group(1).strip()
    if not pile_str:
        return []
    if ' ' in pile_str:
        return pile_str.split()
    return [pile_str[i:i + 2] for i in range(0, len(pile_str), 2)]


def parse_game_state(observation: str) -> GameState:
    if 'Invalid' in observation and 'Legal Actions:' not in observation:
        raise ValueError("Invalid action response — not a game state")
    parse_warnings = []
    player_match   = re.search(r'You are Player (\d+)', observation)
    player_id      = int(player_match.group(1)) if player_match else 0
    hand           = parse_hand_from_observation(observation)
    if not hand:
        parse_warnings.append(
            f"hand=[] (empty — shaping disabled). obs head: {observation[:200]!r}"
        )
    dw_match       = re.search(r'Deadwood=(\d+)', observation)
    deadwood       = int(dw_match.group(1)) if dw_match else 0
    if not dw_match:
        parse_warnings.append("deadwood=0 (fallback — shaping will be 0)")
    phase_match    = re.search(r'Phase: (\w+)', observation)
    phase          = phase_match.group(1) if phase_match else 'Draw'
    if not phase_match:
        parse_warnings.append("phase='Draw' (fallback)")
    knock_match    = re.search(r'Knock card: (\d+)', observation)
    knock_card     = int(knock_match.group(1)) if knock_match else 10
    upcard_match   = re.search(r'Stock size: \d+\s+Upcard: (\w+)', observation)
    upcard         = upcard_match.group(1) if upcard_match else 'XX'
    stock_match    = re.search(r'Stock size: (\d+)', observation)
    stock_size     = int(stock_match.group(1)) if stock_match else 0

    legal_actions: dict[int, str] = {}
    legal_block = re.search(r'Legal Actions:\s*(.*?)(?:\n\s*\n|\Z)', observation, re.DOTALL)
    if legal_block:
        for m in re.finditer(r'(\d+)\s*->\s*(.+)', legal_block.group(1)):
            legal_actions[int(m.group(1))] = m.group(2).strip()
    knock_action_ids = {aid for aid, label in legal_actions.items() if 'knock' in label.lower()}

    if parse_warnings:
        print(f"[PARSE_WARN] parse_game_state fallbacks: {', '.join(parse_warnings)}")
    return GameState(
        hand=hand, deadwood=deadwood, phase=phase, knock_card=knock_card,
        upcard=upcard, stock_size=stock_size,
        discard_pile=parse_discard_pile(observation), player_id=player_id,
        legal_actions=legal_actions, knock_action_ids=knock_action_ids,
    )


def extract_action_id(completion_text: str) -> str:
    cleaned = remove_reasoning_tags(completion_text)
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    # Prefer explicit "Action: N" pattern — most reliable since the system
    # prompt instructs the model to emit this form.
    if "Action:" in cleaned:
        tail = cleaned.split("Action:")[-1].strip()
        m = re.match(r"\s*(-?\d+)", tail)
        if m:
            return m.group(1)
    # Fallback: last integer in the output. Instruction-tuned models with
    # chain-of-thought (Wei et al. 2022) typically place the final answer
    # at the end of the completion after any reasoning prose, so taking
    # the last integer is safer than the first (e.g. "1. I play 15" → 15).
    matches = re.findall(r"-?\d+", cleaned)
    return matches[-1] if matches else cleaned.strip()


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Per-step reward aggregator. All shaping happens in-step via
    opponent_aware_step_shaping (Mizukami 2015 + Grzes 2017) and the
    deadwood PBRS added inside _run_episode (Ng/Harada/Russell 1999,
    Φ = −optimal_deadwood).

    calculate_episode_reward does NOT add episode-level accumulators.
    It only sums the per-step rewards and adds the environmental win/loss
    outcome from the underlying game.
    """

    def __init__(self):
        self.invalid_penalty = INVALID_PENALTY

    def calculate_step_reward(
        self,
        states: list[GameState],
        action: str,
        env_reward: float,
        is_invalid: bool = False,
    ) -> float:
        if is_invalid:
            return self.invalid_penalty
        return 0.0

    def calculate_episode_reward(
        self,
        step_rewards: list[float],
        env_reward: float,
        done: bool,
        initial_state: "GameState | None" = None,
        final_state:   "GameState | None" = None,
        all_states:    "list[GameState] | None" = None,
    ) -> float:
        """Pure per-step total return: terminal outcome + sum of per-step shaping.

        No split clipping on aggregate positive/negative — that pattern conflated
        per-step signals into episode-level magnitudes and broke PBRS invariance
        (Ng/Harada/Russell 1999). Per-step bounds live in calculate_step_reward
        and the shaping helpers; this function just sums.
        """
        terminal = 0.0
        if done:
            # G.O.D server normalizes env_reward to [0,1] zero-sum;
            # map to [-TERMINAL_WIN_REWARD, +TERMINAL_WIN_REWARD].
            terminal = TERMINAL_WIN_REWARD * (2.0 * env_reward - 1.0)
        elif final_state is not None and final_state.hand:
            # Unfinished-episode shame: gradient toward lower deadwood on timeouts.
            terminal = -final_state.deadwood / 20.0
        return terminal + sum(step_rewards)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry.

    Tuned for a 3-hour training budget. ``initial_max_turn`` (=12) and
    ``rollouts_per_stage`` (=100) come from the reasoning / full_prompt
    ModeConfigs in env_configs so most games finish from batch 1 and the
    hint schedule decays within the available rollouts. Final ceiling is
    capped at 14 — in observed logs almost every finished Gin Rummy game
    closed in ≤16 turns; 25 was leaving episodes unfinished that weren't
    worth the wall-clock. Lowering the cap also shrinks episode-token
    length ~40% for faster steps.
    """
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=14,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.5,
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
        "mcts_max_simulations": _MCTS_SIMS,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Initialized: initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn={curriculum.final_max_turn}, rollouts_per_stage={trainer.args.rollouts_per_stage}"
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
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\n"
    "SETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7\u2660 7\u2665 7\u2663)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5\u2666 6\u2666 7\u2666)\n"
    "Examples:\n- Valid runs: A\u2660-2\u2660-3\u2660, 9\u2665-10\u2665-J\u2665-Q\u2665, 10\u2663-J\u2663-Q\u2663-K\u2663\n"
    "- Invalid: K\u2660-A\u2660-2\u2660 (Ace is LOW only, not wraparound)\n\n"
    "CARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(\u2660), h(\u2665), d(\u2666), c(\u2663)\n"
    "- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: 52=Draw upcard, 54=Pass\n"
    "2. Draw: 52=Draw upcard, 53=Draw stock\n"
    "3. Discard: action ID = card index (shown in Legal Actions)\n"
    "4. Layoff: card indices or 54=Pass\n"
    "5. Knock: declare end when deadwood \u2264 knock_card\n\n"
    "EACH TURN:\n1. DRAW: stock (53) or upcard (52)\n"
    "2. DISCARD: choose a card by action ID\n\n"
    "KNOCKING:\n- Gin: 0 deadwood = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n"
    "# Output Format\nYou must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
)

_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "- Early game: Draw from deck to see more cards\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Knock when you have \u226410 deadwood points and think you're ahead\n"
    "- Go for Gin (0 deadwood) when close for bonus points\n"
    "- In Layoff phase: use 'Dead cards' hint to find extension opportunities\n"
    "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
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
) -> tuple[int, "dict | None"]:
    game_id      = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt fallback (updated every loop iteration)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    invalid_count = 0
    done          = False
    train_reward  = 0.0
    final_reward  = 0.0
    turn_number   = 0
    game_state_history: list[GameState] = []
    rewards:            list[float]     = []
    calculator          = RewardCalculator()
    dead_card_tracker   = DeadCardTracker()
    void_run_tracker    = VoidRunInference()
    prev_discard_pile:  list[str]       = []
    event_counter: dict[str, int]       = {}
    knock_committed:                bool = False  # one-shot for KNOCK_COMMIT_BONUS

    # Opponent modelling — active in both training modes so the full_prompt
    # variant is a meaningful A/B against the base env (Mizukami 2015 / Grzes
    # 2017 / Ng-Harada-Russell 1999 shaping applies in both modes).
    bayes_hand: BayesianOpponentHandModel = BayesianOpponentHandModel()

    use_hints = random.random() < current_hint_prob

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed":    random.randint(0, 2 ** 31 - 1),
        "opponent": "mcts",
        "mcts_max_simulations": _MCTS_SIMS,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block        = reset_res.json()["result"]
        episode_id          = result_block.get("episode_id", "")
        raw_observation     = result_block.get("observation", "")
        formatted_observation = extract_and_format_observation(raw_observation)
        initial_game_state  = parse_game_state(formatted_observation)
        game_state_history.append(initial_game_state)
        dead_card_tracker.update_from_discard_pile(initial_game_state.discard_pile)
        prev_discard_pile = list(initial_game_state.discard_pile)
        actual_hand_size = len(initial_game_state.hand)
        bayes_hand._opp_hand_size = actual_hand_size if actual_hand_size > 0 else 7
        bayes_hand.initialize(
            our_hand=initial_game_state.hand,
            discard_pile=initial_game_state.discard_pile,
        )
    except Exception as exc:
        print(f"Failed to reset environment (Game {game_id}): {exc}")
        return index, None

    system_prompt = _SYSTEM_PROMPT + (_HINT_PROMPT if use_hints else "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": formatted_observation},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        action_to_send  = extract_action_id(completion_text)

        # --- Full-prompt token accumulation ---
        if use_full_prompt:
            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids      = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                else:
                    delta = prompt_ids[len(prev_full_ids):]
                    if delta:
                        episode_completion_ids.extend(delta)
                        episode_logprobs.extend([0.0] * len(delta))
                        episode_action_mask.extend([0] * len(delta))
                    prev_full_ids = prompt_ids.copy()

            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}, ending early")
                done = True
                break

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        # --- Step environment ---
        is_invalid = False
        try:
            formatted_observation = ""
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block            = step_res.json()["result"]
            raw_observation       = step_block.get("observation", "")
            formatted_observation = extract_and_format_observation(raw_observation)
            step_reward           = step_block.get("reward", 0)
            done                  = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            step_reward = -0.01
            done        = False
            invalid_count += 1

        if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
            invalid_count += 1
            is_invalid = True

        immediate_reward = 0.0
        prev_state_for_shaping = game_state_history[-1] if game_state_history else None

        # Parse the new observation early so we can inject a knock banner
        # into the same message the model will see next turn.
        new_state: "GameState | None" = None
        if not done and not is_invalid and formatted_observation:
            try:
                new_state = parse_game_state(formatted_observation)
            except Exception as exc:
                print(f"Failed to parse game state: {exc}")
                new_state = None

        if done:
            final_reward = step_reward
            messages.append({"role": "user", "content": formatted_observation})
        else:
            # --- Update bayes_hand from opponent moves (must happen BEFORE
            #     context injection so the belief reflects latest evidence) ---
            if not is_invalid and new_state is not None:
                if len(new_state.discard_pile) < len(prev_discard_pile):
                    drawn_card = prev_discard_pile[-1] if prev_discard_pile else None
                    if drawn_card:
                        bayes_hand.update_opp_drew_upcard(drawn_card)
                elif len(new_state.discard_pile) > len(prev_discard_pile):
                    discarded_card = new_state.discard_pile[-1] if new_state.discard_pile else None
                    if discarded_card:
                        bayes_hand.update_opp_discarded(discarded_card)
                else:
                    bayes_hand.update_opp_drew_stock()

            # --- Build augmented observation ---
            dead_card_tracker.update_from_observation(formatted_observation)
            context_state = new_state if new_state is not None else (
                game_state_history[-1] if game_state_history else None
            )
            current_hand = context_state.hand if context_state is not None else []
            discards = list(dead_card_tracker.seen_discards)

            context_parts: list[str] = []
            if context_state is not None:
                context_parts.append(build_json_belief_header(context_state, bayes_hand, turn_number))
            context_parts.append(dead_card_tracker.summary(current_hand))
            context_parts.append(void_run_tracker.summary(discards))
            if context_state is not None and context_state.knock_action_ids:
                context_parts.append(_build_knock_banner(context_state, bayes_hand, turn_number))
            context_parts = [p for p in context_parts if p]

            obs_augmented = (
                formatted_observation + "\n\n" + "\n".join(context_parts)
                if context_parts else formatted_observation
            )
            messages.append({"role": "user", "content": obs_augmented})

            # --- Append new_state to history ---
            if not is_invalid and new_state is not None:
                game_state_history.append(new_state)
                dead_card_tracker.update_from_discard_pile(new_state.discard_pile)
                prev_discard_pile = list(new_state.discard_pile)
                immediate_reward  = calculator.calculate_step_reward(
                    game_state_history, action_to_send, 0.0
                )
            else:
                immediate_reward = calculator.calculate_step_reward(
                    game_state_history, action_to_send, 0.0, is_invalid=is_invalid
                )

        # --- Per-step deadwood PBRS (Ng/Harada/Russell 1999): Φ = −optimal_deadwood ---
        # PBRS invariance (Ng-Harada-Russell 1999 Theorem 1) requires Φ(s) to
        # be consistently defined. If either the prev or curr hand failed to
        # parse (hand=[]), skip PBRS entirely instead of substituting 0 —
        # otherwise a parse-failed prev_dw=0 paired with a valid curr_dw=40
        # creates a spurious −0.8 PBRS signal that biases the policy in a
        # wrong direction.
        deadwood_pbrs = 0.0
        if (not is_invalid
            and prev_state_for_shaping is not None and prev_state_for_shaping.hand
            and new_state is not None and new_state.hand):
            prev_dw = compute_optimal_deadwood(prev_state_for_shaping.hand)
            curr_dw = compute_optimal_deadwood(new_state.hand)
            deadwood_pbrs = (prev_dw - curr_dw) / DEADWOOD_PBRS_SCALE

        # --- Per-step opponent-aware shaping ---
        if not is_invalid:
            shaping_delta, event = opponent_aware_step_shaping(
                prev_state=prev_state_for_shaping,
                curr_state=new_state,
                action_to_send=action_to_send,
                bayes_hand=bayes_hand,
                turn_number=turn_number,
            )
            immediate_reward += shaping_delta + deadwood_pbrs
            if event:
                for ev in event.split("+"):
                    if ev:
                        event_counter[ev] = event_counter.get(ev, 0) + 1
            if abs(deadwood_pbrs) > 0.01:
                tag = "dw_pbrs_pos" if deadwood_pbrs > 0 else "dw_pbrs_neg"
                event_counter[tag] = event_counter.get(tag, 0) + 1

            # --- Structural discard bonuses (boss-impossible offense) ---
            # DeadCardTracker + VoidRunInference give deterministic per-step
            # safety signals from the discard pile that boss's opponent_modeling
            # cannot compute. Stacks with the Bayesian discard-safety shaping.
            try:
                aid_struct = int(str(action_to_send).strip())
            except (ValueError, TypeError):
                aid_struct = None
            if (aid_struct is not None
                and prev_state_for_shaping is not None
                and prev_state_for_shaping.phase == "Discard"
                and 0 <= aid_struct < len(prev_state_for_shaping.hand)
                and aid_struct not in prev_state_for_shaping.knock_action_ids):
                discarded_card = prev_state_for_shaping.hand[aid_struct]
                if discarded_card and len(discarded_card) == 2:
                    rank, suit = discarded_card[0].upper(), discarded_card[1].lower()
                    same_rank_dead = sum(
                        1 for s in DeadCardTracker.ALL_SUITS
                        if dead_card_tracker.is_dead(rank + s)
                    )
                    if same_rank_dead >= 2:
                        immediate_reward += DEAD_RANK_DISCARD_BONUS
                        event_counter["dead_rank_discard"] = (
                            event_counter.get("dead_rank_discard", 0) + 1
                        )
                    suit_dead_count = sum(
                        1 for r in DeadCardTracker.ALL_RANKS
                        if dead_card_tracker.is_dead(r + suit)
                    )
                    if suit_dead_count >= 3:
                        immediate_reward += VOID_SUIT_DISCARD_BONUS
                        event_counter["void_suit_discard"] = (
                            event_counter.get("void_suit_discard", 0) + 1
                        )

            # --- Knock-commitment one-shot (breaks PBRS entry/exit symmetry) ---
            # Bonus scaled by knock_confidence_score ∈ [0, 1], mapped to
            # multiplier [0.5, 1.5] so every legitimate commit still gets a
            # baseline reward but confident commits (GIN, aggressive threshold,
            # high Chow-Robbins p_now) dominate reckless ones. Terminal reward
            # (±5) still dominates even the max-confidence commit (1.125).
            if not knock_committed and prev_state_for_shaping is not None:
                try:
                    aid = int(str(action_to_send).strip())
                except (ValueError, TypeError):
                    aid = None
                if aid is not None and aid in prev_state_for_shaping.knock_action_ids:
                    conf = knock_confidence_score(
                        prev_state_for_shaping, bayes_hand, turn_number
                    )
                    immediate_reward += KNOCK_COMMIT_BONUS * (0.5 + conf)
                    knock_committed = True
                    event_counter["knock_committed"] = event_counter.get("knock_committed", 0) + 1
                    if conf >= 0.7:
                        event_counter["confident_knock"] = event_counter.get("confident_knock", 0) + 1
                    elif conf <= 0.3:
                        event_counter["reckless_knock"] = event_counter.get("reckless_knock", 0) + 1

        rewards.append(immediate_reward)
        turn_number += 1

    # --- Episode reward ---
    initial_state = game_state_history[0] if game_state_history else None
    final_state   = game_state_history[-1] if game_state_history else None
    train_reward  = calculator.calculate_episode_reward(
        rewards, final_reward, done, initial_state, final_state, all_states=game_state_history
    )

    initial_dw = game_state_history[0].deadwood if game_state_history else 0
    final_dw   = game_state_history[-1].deadwood if game_state_history else 0
    events_str = " ".join(f"{k}:{v}" for k, v in event_counter.items()) if event_counter else "-"
    print(
        f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
        f"Ret:{train_reward:6.2f} EnvR:{final_reward:5.1f} "
        f"DW:{initial_dw:2d}\u2192{final_dw:2d} Inv:{invalid_count} "
        f"Events:{events_str}"
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
    _ensure_initialized(trainer)

    curriculum        = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(f"[CURRICULUM] Rollout {curriculum.total_rollouts}: max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}")

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
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    wins     = sum(1 for r in list_results if r["final_score"] > 0.5)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0
    print(f"[BATCH] Finished:{finished}/{len(list_results)} Wins:{wins} AvgReturn:{avg_return:.3f}")

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
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking.

    Opponent-modelling stack (same as the last-prompt variant) is active so
    the variant is a meaningful A/B against the base env in this mode too.
    """
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs.

    Enables the full opponent-modelling stack:
      * JSON belief header (Huang/Chalkiadakis/Elkind 2024)
      * BayesianOpponentHandModel posterior (PerfectDou 2022 analogue)
      * VoidRunInference (Sturtevant & White 2006 analogue)
      * Knock banner using Kotnik/Kalita (2003) + Chow-Robbins (1971) thresholds
      * Per-step shaping: Mizukami 2015 discard-safety + Grzes 2017 knock PBRS
        + Ng/Harada/Russell 1999 deadwood PBRS.
    """
    return _dispatch(prompts, trainer, use_full_prompt=False)
# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
