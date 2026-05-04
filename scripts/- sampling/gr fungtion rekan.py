import functools
import random
import re
from collections import Counter
from concurrent.futures import as_completed
from dataclasses import dataclass
from threading import Semaphore
from typing import Optional

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "gin_rummy"
_MAX_EPISODE_TOKENS = 16384
_TIMEOUT = 2400
_MCTS_SIMS = 50

# ---------------------------------------------------------------------------
# Model-aware tier configs (hybrid approach: auto-detect + config dict)
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = {
    "small": {   # 3B models: clean prompt, no Bayesian injection
        "inject_bayesian_context": False,
        "max_prompt_len": 5000,
    },
    "medium": {  # 4B models: clean prompt, slightly more context
        "inject_bayesian_context": False,
        "max_prompt_len": 6000,
    },
    "large": {   # 7B models: Bayesian injection ON, more context
        "inject_bayesian_context": True,
        "max_prompt_len": 8000,
    },
}


def _detect_tier(trainer) -> tuple[str, str]:
    """Auto-detect model size tier from trainer's loaded model.

    Checks multiple sources for the model name (model config, tokenizer,
    trainer args) to ensure robust detection across all HuggingFace models.

    Returns (tier, model_name) tuple.
    """
    model_name = ""
    # Source 1: model config (most reliable — always set by from_pretrained)
    if hasattr(trainer, "model") and hasattr(trainer.model, "config"):
        model_name = getattr(trainer.model.config, "_name_or_path", "")
    # Source 2: tokenizer (also reliable)
    if not model_name and hasattr(trainer, "processing_class"):
        model_name = getattr(trainer.processing_class, "name_or_path", "")
    # Source 3: trainer args (fallback)
    if not model_name:
        model_name = getattr(trainer.args, "model", "")

    name_lower = model_name.lower()
    if "7b" in name_lower:
        return "large", model_name or "unknown"
    if "4b" in name_lower:
        return "medium", model_name or "unknown"
    return "small", model_name or "unknown"

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'T': 10, 'J': 11, 'Q': 12, 'K': 13,
}
RANK_ORDER = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

REASONING_TAG_PAIRS = [
    ("think", "think"), ("thinking", "thinking"), ("reasoning", "reasoning"),
    ("thought", "thought"), ("reflection", "reflection"),
]

# Bayesian-informed bonus/penalty (supplements existing step rewards)
# Scaled 10x from original (0.02-0.03) to be meaningful vs base rewards (2-20)
BAYES_SAFE_DISCARD_BONUS       = 0.3
BAYES_DANGEROUS_DISCARD_PENALTY = 0.5
BAYES_DRAW_UPCARD_BONUS        = 0.3
BAYES_DRAW_UPCARD_PENALTY      = 0.2


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


def would_complete_run(hand: list[str], card: str) -> bool:
    current = sum(1 for r in find_potential_runs(hand) if len(r) >= 3)
    return sum(1 for r in find_potential_runs(hand, card) if len(r) >= 3) > current


def would_improve_run(hand: list[str], card: str) -> bool:
    rank_idx = RANK_ORDER.index(get_rank(card))
    suit = get_suit(card)
    for existing in hand:
        if get_suit(existing) != suit:
            continue
        if abs(rank_idx - RANK_ORDER.index(get_rank(existing))) == 1:
            for run in find_potential_runs(hand + [card]):
                if card in run and len(run) >= 3:
                    return True
    return False


def would_complete_set(hand: list[str], card: str) -> bool:
    return sum(1 for c in hand if get_rank(c) == get_rank(card)) >= 2


def would_improve_set(hand: list[str], card: str) -> bool:
    return sum(1 for c in hand if get_rank(c) == get_rank(card)) == 1


def meld_potential(upcard: str, hand: list[str]) -> int:
    """Estimate how many melds the upcard could participate in."""
    if not upcard or upcard == 'XX' or len(upcard) != 2:
        return 0
    potential = 0
    if would_complete_set(hand, upcard):
        potential += 2
    elif would_improve_set(hand, upcard):
        potential += 1
    if would_complete_run(hand, upcard):
        potential += 2
    elif would_improve_run(hand, upcard):
        potential += 1
    return potential


def is_in_meld(hand: list[str], card: str) -> bool:
    """Check if a card is part of a completed meld (set of 3+ or run of 3+).

    Used to prevent discard bonus from encouraging meld-breaking.
    """
    rank = get_rank(card)
    # Check if part of a set (3+ same rank)
    same_rank = [c for c in hand if get_rank(c) == rank]
    if len(same_rank) >= 3:
        return True
    # Check if part of a completed run (3+ consecutive same suit)
    for run in find_potential_runs(hand):
        if card in run and len(run) >= 3:
            return True
    return False


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


def extract_action_id(completion_text: str) -> str:
    cleaned = remove_reasoning_tags(completion_text)
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()
    match = re.search(r"-?\d+", cleaned)
    return match.group(0) if match else cleaned.strip()


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
# Observation helpers
# ---------------------------------------------------------------------------

def extract_and_format_observation(obs_text: str) -> str:
    if 'Invalid action:' in obs_text and 'Legal Actions:' in obs_text:
        return obs_text
    state_match = re.search(r'Current State:\n(.*)', obs_text, re.DOTALL)
    if not state_match:
        return obs_text
    state_text = state_match.group(0)
    player_match = re.search(r'You are Player (\d+)', obs_text)
    player_id = int(player_match.group(1)) if player_match else 0
    current_state_text, legal_action_text = state_text.split('Legal Actions:')
    return current_state_text + f"You are Player {player_id}.\nLegal Actions:" + legal_action_text


def parse_hand_from_observation(observation: str) -> list[str]:
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    section = re.search(
        rf'Player{player_id}: Deadwood=\d+\n\+-+\+\n(.*?)\n\+-+\+', observation, re.DOTALL
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
    player_match = re.search(r'You are Player (\d+)', observation)
    player_id = int(player_match.group(1)) if player_match else 0
    hand = parse_hand_from_observation(observation)
    dw_match = re.search(r'Deadwood=(\d+)', observation)
    deadwood = int(dw_match.group(1)) if dw_match else 0
    phase_match = re.search(r'Phase: (\w+)', observation)
    phase = phase_match.group(1) if phase_match else 'Draw'
    knock_match = re.search(r'Knock card: (\d+)', observation)
    knock_card = int(knock_match.group(1)) if knock_match else 10
    upcard_match = re.search(r'Stock size: \d+\s+Upcard: (\w+)', observation)
    upcard = upcard_match.group(1) if upcard_match else 'XX'
    stock_match = re.search(r'Stock size: (\d+)', observation)
    stock_size = int(stock_match.group(1)) if stock_match else 0
    return GameState(
        hand=hand, deadwood=deadwood, phase=phase, knock_card=knock_card,
        upcard=upcard, stock_size=stock_size,
        discard_pile=parse_discard_pile(observation), player_id=player_id,
    )


# ---------------------------------------------------------------------------
# Bayesian opponent models (new — compensates 3B model's limited reasoning)
# ---------------------------------------------------------------------------

class BayesianOpponentModel:
    """Track which ranks/suits opponent is collecting based on discard pile deltas.

    When the upcard disappears → opponent drew it → boost heat for that rank/suit.
    When a new card appears on the pile → opponent discarded it → cool that rank/suit.

    Integration: instantiated per-episode in _run_episode(), context
    injected into observations via summary() in last-prompt mode only.
    """

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self.rank_heat: dict[str, float] = {r: 0.0 for r in self.ALL_RANKS}
        self.suit_heat: dict[str, float] = {s: 0.0 for s in self.ALL_SUITS}
        self.opp_draws:    list[str] = []
        self.opp_discards: list[str] = []

    def _update_heat(self, card: str, weight: float) -> None:
        if len(card) != 2:
            return
        rank = card[0].upper()
        suit = card[1].lower()
        if rank in self.rank_heat:
            self.rank_heat[rank] = max(-3.0, min(3.0, self.rank_heat[rank] + weight))
        if suit in self.suit_heat:
            self.suit_heat[suit] = max(-3.0, min(3.0, self.suit_heat[suit] + weight))

    def update_on_opponent_draw(self, drawn_card: str) -> None:
        """Opponent drew an upcard → they want that rank/suit."""
        self.opp_draws.append(drawn_card)
        self._update_heat(drawn_card, weight=1.0)
        # Also boost adjacent ranks in the same suit (run extension)
        if len(drawn_card) == 2:
            rank, suit = drawn_card[0].upper(), drawn_card[1].lower()
            if rank in self.ALL_RANKS:
                idx = self.ALL_RANKS.index(rank)
                for adj_idx in [idx - 1, idx + 1]:
                    if 0 <= adj_idx < len(self.ALL_RANKS):
                        self._update_heat(self.ALL_RANKS[adj_idx] + suit, weight=0.3)

    def update_on_opponent_discard(self, discarded_card: str) -> None:
        """Opponent discarded → they don't want that rank/suit."""
        self.opp_discards.append(discarded_card)
        self._update_heat(discarded_card, weight=-0.5)

    def update_from_discard_pile_delta(
        self, prev_discard_pile: list[str], curr_discard_pile: list[str]
    ) -> None:
        """Detect opponent draw/discard from pile changes."""
        prev_set = list(prev_discard_pile)
        curr_set = list(curr_discard_pile)
        if prev_set and (not curr_set or len(curr_set) < len(prev_set)):
            top_card = prev_set[-1] if prev_set else None
            if top_card:
                self.update_on_opponent_draw(top_card)
        elif curr_set and len(curr_set) > len(prev_set):
            self.update_on_opponent_discard(curr_set[-1])

    def is_dangerous_discard(self, card: str) -> bool:
        if len(card) != 2:
            return False
        rank, suit = card[0].upper(), card[1].lower()
        return self.rank_heat.get(rank, 0.0) >= 1.0 or self.suit_heat.get(suit, 0.0) >= 1.5

    def is_safe_discard(self, card: str) -> bool:
        if len(card) != 2:
            return False
        rank, suit = card[0].upper(), card[1].lower()
        return self.rank_heat.get(rank, 0.0) <= -0.5 and self.suit_heat.get(suit, 0.0) <= -0.5

    def get_danger_cards(self, hand: list[str]) -> list[str]:
        return [c for c in hand if self.is_dangerous_discard(c)]

    def get_safe_cards(self, hand: list[str]) -> list[str]:
        return [c for c in hand if self.is_safe_discard(c)]

    def summary(self, hand: list[str]) -> str:
        """Compact Bayesian context for prompt injection.

        Kept short for the 3B model — highlights dangerous/safe discards
        and opponent's collection direction.
        """
        danger    = self.get_danger_cards(hand)
        safe      = self.get_safe_cards(hand)
        hot_suits = [s for s, h in self.suit_heat.items() if h >= 1.5]
        hot_ranks = [r for r, h in self.rank_heat.items() if h >= 1.0]
        lines = []
        if hot_suits:
            lines.append(f"[Bayesian] Opp building: {', '.join(hot_suits).upper()} suit(s)")
        if hot_ranks:
            lines.append(f"[Bayesian] Opp wants rank(s): {' '.join(hot_ranks)}")
        if danger:
            lines.append(f"[Bayesian] DANGER discards (helps opp): {' '.join(danger[:6])}")
        if safe:
            lines.append(f"[Bayesian] Safe discards (opp ignores): {' '.join(safe[:6])}")
        return "\n".join(lines)


class BayesianOpponentHandModel:
    """Track P(card ∈ opponent_hand | all observations) via Bayesian updates.

    Maintains per-card probability estimates based on:
    - Cards confirmed in opponent's hand (drew upcard)
    - Cards confirmed not in hand (discarded)
    - Prior from remaining unknown cards
    """

    ALL_RANKS = list("A23456789TJQK")
    ALL_SUITS = list("shdc")

    def __init__(self) -> None:
        self._prob: dict[str, float] = {}
        self._opp_hand_size: int = 10
        self._confirmed_in_hand:     set[str] = set()
        self._confirmed_not_in_hand: set[str] = set()

    def _all_cards(self) -> list[str]:
        return [r.lower() + s for r in self.ALL_RANKS for s in self.ALL_SUITS]

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
        boost = 1.0 / len(stock_candidates)
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

    def knock_risk(self) -> str:
        top_hand = self.estimated_opponent_hand(10)
        if not top_hand:
            return "Unknown"
        estimated_dw  = sum(get_value(c.upper()) * p for c, p in top_hand)
        confirmed_dw  = sum(get_value(c.upper()) for c in self._confirmed_in_hand if len(c) == 2)
        total_est     = estimated_dw + confirmed_dw
        if total_est <= 10:
            return f"HIGH (~{total_est:.0f}pts)"
        elif total_est <= 25:
            return f"MEDIUM (~{total_est:.0f}pts)"
        else:
            return f"LOW (~{total_est:.0f}pts)"

    def summary(self, our_hand: list[str]) -> str:
        """Compact Bayesian context for prompt injection."""
        lines = [f"[BayesHand] Opp knock risk: {self.knock_risk()}"]
        top_held = self.estimated_opponent_hand(5)
        if top_held:
            cards_str = " ".join(f"{c}({p:.0%})" for c, p in top_held if p >= 0.25)
            if cards_str:
                lines.append(f"[BayesHand] Likely opp cards: {cards_str}")
        confirmed = list(self._confirmed_in_hand)
        if confirmed:
            lines.append(f"[BayesHand] Confirmed opp cards: {' '.join(confirmed[:4])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Shaped reward calculator for Gin Rummy with Bayesian awareness.

    Inherits the same step-reward architecture as the base gin_rummy_env.py
    RewardCalculator and adds Bayesian-informed discard/draw shaping.
    """

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma
        # --- Reward capping (Lilianweng audit: anti-reward-hacking) ---
        self.step_reward_cap = 25.0
        self.episode_reward_cap = 100.0
        # --- Base signals ---
        self.deadwood_weight = 0.5
        self.high_card_penalty = -0.5  # Increased: K=13pts, must discard fast
        self.pair_bonus = 2.0
        self.set_bonus = 8.0
        self.potential_run_bonus = 2.0
        self.run_bonus = 10.0
        self.break_pair_penalty = -1.0   # Reduced: allow flexibility
        self.break_set_penalty = -4.0    # Reduced: don't freeze model
        self.break_run_penalty = -5.0    # Reduced: allow strategic discards
        self.knock_ready_bonus = 25.0    # Increased: knock ASAP!
        self.knock_action_bonus = 15.0   # NEW: reward for choosing to KNOCK
        self.missed_knock_penalty = -8.0 # NEW: penalty for NOT knocking when able
        self.stock_draw_bonus = 0.5      # NEW: prefer stock (info hiding)
        self.discard_highest_bonus = 2.0  # NEW: reward discarding highest non-meld card
        self.discard_useful_penalty = -1.5
        self.missed_opportunity_penalty = -2.0
        self.picked_up_useless_upcard_penalty = -2.0

    def calculate_step_reward(
        self,
        states: list[GameState],
        action: str,
        env_reward: float,
        *,
        bayesian_model: "BayesianOpponentModel | None" = None,
    ) -> float:
        """Per-step shaped reward: base signals + Bayesian adjustments."""
        if len(states) < 2:
            return 0.0
        prev, curr = states[-2], states[-1]
        reward = 0.0

        # --- Base signals ---
        reward += self.deadwood_weight * (prev.deadwood - curr.deadwood)
        reward += self.high_card_penalty * curr.num_high_cards()

        pair_change = curr.count_pairs() - prev.count_pairs()
        reward += (self.pair_bonus if pair_change > 0 else self.break_pair_penalty) * abs(pair_change) if pair_change != 0 else 0

        set_change = curr.count_sets() - prev.count_sets()
        reward += (self.set_bonus if set_change > 0 else self.break_set_penalty) * abs(set_change) if set_change != 0 else 0

        run_change = curr.count_runs() - prev.count_runs()
        reward += (self.run_bonus if run_change > 0 else self.break_run_penalty) * abs(run_change) if run_change != 0 else 0

        potential_run_change = curr.count_potential_runs() - prev.count_potential_runs()
        if potential_run_change > 0:
            reward += self.potential_run_bonus * potential_run_change

        if curr.can_knock() and not prev.can_knock():
            reward += self.knock_ready_bonus

        if prev.phase == 'Discard' and len(curr.discard_pile) > len(prev.discard_pile):
            newly_discarded = [c for c in curr.discard_pile if c not in prev.discard_pile]
            if newly_discarded:
                dc = newly_discarded[0]
                if sum(1 for c in prev.hand if get_rank(c) == get_rank(dc)) >= 2:
                    reward += self.discard_useful_penalty
                if would_improve_run(prev.hand, dc):
                    reward += self.discard_useful_penalty
                # Smart discard bonus: reward discarding highest non-meld card
                if not is_in_meld(prev.hand, dc):
                    non_meld_cards = [c for c in prev.hand if not is_in_meld(prev.hand, c)]
                    if non_meld_cards:
                        highest_val = max(get_value(c) for c in non_meld_cards)
                        if get_value(dc) == highest_val:
                            reward += self.discard_highest_bonus

        if prev.phase == 'Draw' and prev.upcard != 'XX':
            upcard = prev.upcard
            if action == '53':  # drew from stock
                # Bonus for drawing from stock (information hiding — Claude strategy)
                reward += self.stock_draw_bonus
                if would_complete_set(prev.hand, upcard) or would_complete_run(prev.hand, upcard):
                    reward += self.missed_opportunity_penalty
            else:
                if not (would_complete_set(prev.hand, upcard) or would_complete_run(prev.hand, upcard)):
                    reward += self.picked_up_useless_upcard_penalty

        # --- Knock decision signals (THE most important decision in Gin Rummy) ---
        if action == '55' and prev.can_knock():
            # Model chose to KNOCK — this is almost always correct!
            reward += self.knock_action_bonus
        elif prev.can_knock() and prev.phase in ('Discard', 'Draw') and action != '55':
            # Model COULD knock but chose not to — penalize
            reward += self.missed_knock_penalty

        # --- Bayesian-informed adjustments (opponent modelling layer) ---
        if bayesian_model is not None:
            # Discard phase: bonus/penalty based on danger assessment
            if prev.phase == 'Discard' and len(curr.discard_pile) > len(prev.discard_pile):
                newly_discarded = [c for c in curr.discard_pile if c not in prev.discard_pile]
                if newly_discarded:
                    dc = newly_discarded[0]
                    if bayesian_model.is_safe_discard(dc):
                        reward += BAYES_SAFE_DISCARD_BONUS
                    elif bayesian_model.is_dangerous_discard(dc):
                        reward -= BAYES_DANGEROUS_DISCARD_PENALTY

            # Draw phase: bonus for drawing useful upcard, penalty for ignoring it
            if prev.phase in ('Draw', 'FirstUpcard') and prev.upcard != 'XX':
                potential = meld_potential(prev.upcard, prev.hand)
                if action == '52':  # drew upcard
                    if potential > 0:
                        scale = min(potential / 4.0, 1.0)
                        reward += BAYES_DRAW_UPCARD_BONUS * scale
                    else:
                        reward -= BAYES_DRAW_UPCARD_PENALTY

        if env_reward != 0.0:
            reward += max(min(env_reward * 100.0, 50.0), -50.0)

        return max(-self.step_reward_cap, min(self.step_reward_cap, reward))

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        """Discounted return G = Σ γ^(T-1-i) * r_i (capped)."""
        if not rewards:
            return 0.0
        T = len(rewards)
        result = sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))
        return max(-self.episode_reward_cap, min(self.episode_reward_cap, result))


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=30,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.5,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    if _state.get("initialized"):
        return

    # Auto-detect model tier
    tier, model_name = _detect_tier(trainer)
    model_config = _MODEL_CONFIGS[tier]
    print(f"[MODEL] Detected tier={tier!r} for model={model_name!r}")
    print(f"[MODEL] Config: inject_bayesian={model_config['inject_bayesian_context']}, max_prompt_len={model_config['max_prompt_len']}")

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
        f"final_max_turn=30, rollouts_per_stage={trainer.args.rollouts_per_stage}"
    )

    _state.update(
        initialized=True,
        rank=rank,
        env_pool=env_pool,
        num_servers=num_servers,
        thread_pool=thread_pool,
        generation_semaphore=generation_semaphore,
        curriculum=curriculum,
        model_config=model_config,
    )


# ---------------------------------------------------------------------------
# Prompts (concise — optimised for Llama-3.2-3B context window)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\n"
    "SETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7♠ 7♥ 7♣)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5♦ 6♦ 7♦)\n"
    "Examples:\n- Valid runs: A♠-2♠-3♠, 9♥-10♥-J♥-Q♥\n"
    "- Invalid: K♠-A♠-2♠ (Ace is LOW only)\n\n"
    "CARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(♠), h(♥), d(♦), c(♣)\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: 52=Draw upcard, 54=Pass\n"
    "2. Draw: 52=Draw upcard, 53=Draw stock\n"
    "3. Discard: action ID = card index (shown in Legal Actions)\n"
    "4. Layoff: card indices or 54=Pass\n"
    "5. Knock: declare end when deadwood ≤ knock_card\n\n"
    "KNOCKING:\n- Gin: 0 deadwood = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n"
    '# Output Format\nYour output must strictly follow this format: "Thought:\nyour thoughts ONLY in text.\n\nAction:\nONLY your action ID (a single number)."\n'
)

_HINT_PROMPT = (
    "\n\n**Think short and act quickly!**\n\n# Strategy Tips\n"
    "- KNOCK IMMEDIATELY when your deadwood is at or below the knock card value!\n"
    "- Prefer drawing from STOCK (action 53) to keep your hand hidden\n"
    "- Only draw the upcard (action 52) if it completes a meld\n"
    "- Discard your HIGHEST deadwood card first (K=13, Q=12, J=11)\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Go for Gin (0 deadwood) ONLY when very close, otherwise knock!\n"
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
    model_config: dict,
) -> tuple[int, "dict | None"]:
    """
    Run one Gin Rummy episode with Bayesian opponent modelling.

    Follows the same episode runner pattern as gin_rummy_env.py:
    - Parse GameState per turn
    - Track Bayesian models (BayesianOpponentModel + BayesianOpponentHandModel)
    - Inject Bayesian context into observations (last-prompt mode)
    - Collect step rewards → γ-discounted return
    """
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
    prev_discard_pile:  list[str]       = []

    # Opponent modelling — Bayesian models for reward shaping (all tiers)
    # Context injection only for 7B+ (model_config['inject_bayesian_context'])
    bayesian_model: "BayesianOpponentModel | None"     = None
    bayes_hand:     "BayesianOpponentHandModel | None" = None
    inject_context = model_config.get("inject_bayesian_context", False)
    if not use_full_prompt:
        bayesian_model = BayesianOpponentModel()
        bayes_hand     = BayesianOpponentHandModel()

    # Easy replay: 5% episodes always use hints to prevent forgetting (Lilianweng audit)
    EASY_REPLAY_PROB = 0.05
    use_hints = random.random() < max(current_hint_prob, EASY_REPLAY_PROB)

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed":    game_id,
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
        prev_discard_pile = list(initial_game_state.discard_pile)
        if not use_full_prompt and bayes_hand is not None:
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
    effective_max_turn = current_max_turn

    while not done and turn_number < effective_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids     = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs       = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        action_to_send  = extract_action_id(completion_text)

        # --- Full-prompt token accumulation ---
        if use_full_prompt:
            _max_prompt = model_config.get("max_prompt_len", 5000)
            if len(prompt_ids) > _max_prompt:
                print(f"Warning: Prompt exceeded {_max_prompt} tokens at turn {turn_number}, ending early")
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids      = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    print(f"Warning: token shift at turn {turn_number}. Skipping delta mask.")
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
        if done:
            final_reward = step_reward
            messages.append({"role": "user", "content": formatted_observation})
        else:
            # --- Dynamic knock reminder (inject when can_knock is True) ---
            knock_reminder = ""
            if game_state_history:
                latest_gs = game_state_history[-1]
                if latest_gs.can_knock():
                    knock_reminder = (
                        f"\n\n⚡ You CAN knock now! "
                        f"Deadwood ({latest_gs.deadwood}) ≤ knock card ({latest_gs.knock_card}). "
                        f"Choose action 55 to KNOCK!"
                    )

            # Bayesian context injection: ON for 7B+ (large tier), OFF for 3B-4B
            if inject_context and not use_full_prompt and bayesian_model is not None and bayes_hand is not None:
                current_hand = game_state_history[-1].hand if game_state_history else []
                bayes_summary      = bayesian_model.summary(current_hand)
                bayes_hand_summary = bayes_hand.summary(current_hand)
                context_parts      = [p for p in [bayes_summary, bayes_hand_summary] if p]
                obs_augmented = (
                    formatted_observation + "\n\n" + "\n".join(context_parts)
                    if context_parts else formatted_observation
                )
                messages.append({"role": "user", "content": obs_augmented + knock_reminder})
            else:
                messages.append({"role": "user", "content": formatted_observation + knock_reminder})

            # --- Parse game state and update trackers ---
            if not is_invalid:
                try:
                    game_state = parse_game_state(formatted_observation)
                except Exception as exc:
                    print(f"Failed to parse game state: {exc}")
                    immediate_reward = -10.0
                else:
                    game_state_history.append(game_state)

                    # Update Bayesian models (last-prompt mode)
                    if not use_full_prompt and bayesian_model is not None and bayes_hand is not None:
                        bayesian_model.update_from_discard_pile_delta(prev_discard_pile, game_state.discard_pile)
                        if len(game_state.discard_pile) < len(prev_discard_pile):
                            drawn_card = prev_discard_pile[-1] if prev_discard_pile else None
                            if drawn_card:
                                bayes_hand.update_opp_drew_upcard(drawn_card)
                        elif len(game_state.discard_pile) > len(prev_discard_pile):
                            discarded_card = game_state.discard_pile[-1] if game_state.discard_pile else None
                            if discarded_card:
                                bayes_hand.update_opp_discarded(discarded_card)
                        else:
                            bayes_hand.update_opp_drew_stock()

                    prev_discard_pile = list(game_state.discard_pile)
                    immediate_reward  = calculator.calculate_step_reward(
                        game_state_history, action_to_send, 0.0,
                        bayesian_model=bayesian_model,
                    )
            else:
                immediate_reward = -10.0

        if done:
            # Direct scaling — any win is positive, any loss is negative
            # No -0.5 offset: small-margin wins should still be rewarded
            if step_reward > 0.5:
                immediate_reward = step_reward * 60.0   # Win: 30-60 range
            elif step_reward > 0.0:
                immediate_reward = step_reward * 20.0   # Small win: 0-10 range
            else:
                immediate_reward = -30.0                # Loss: flat penalty

        rewards.append(immediate_reward)
        turn_number += 1

    # --- Episode reward ---
    train_reward = calculator.calculate_discounted_return(rewards)
    initial_dw = game_state_history[0].deadwood if game_state_history else 0
    final_dw   = game_state_history[-1].deadwood if game_state_history else 0
    print(
        f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
        f"Ret:{train_reward:6.2f} EnvR:{final_reward:5.1f} "
        f"DW:{initial_dw:2d}→{final_dw:2d} Inv:{invalid_count}"
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
        model_config=_state["model_config"],
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
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs.

    Enables full Bayesian opponent modelling (BayesianOpponentModel +
    BayesianOpponentHandModel) with context injection — compensates the
    3B model's limited opponent tracking.
    """
    return _dispatch(prompts, trainer, use_full_prompt=False)