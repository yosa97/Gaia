"""Leduc Poker with Bayesian opponent modeling.

Maintains a posterior P(opp_card | actions, public_card) across J/Q/K and
surfaces it to the LLM prompt each turn. Posterior updates are Bayesian with
CFR-style action-likelihood priors for each card and round.

Algorithms used
---------------
- Initial prior: 6-card deck (2 of each rank). After we hold our_card, opp's
  card is drawn from the remaining 5 cards → P(opp=R) = (2 - [our_card==R]) / 5.
- Public-card conditioning: joint Bayes
    P(opp=R | public=X, our_card) ∝ P(opp=R) × P(public=X | opp=R, our_card)
  where P(public=X | opp=R, our_card) = (# of X left after our and opp draws) / 4.
- Action likelihood: per-rank, per-round table approximating Nash raise/call/fold
  frequencies. R2 splits into pair vs no-pair branches based on whether opp's
  hypothesised rank matches the public card.

Imports the parser, formatter, action helper, and base prompt from
``leduc_poker_env`` so per-game logic stays in one place.
"""

import functools
import random
from concurrent.futures import as_completed
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
from envs.leduc_poker_env import (
    GameState,
    parse_game_state,
    _format_observation,
    _parse_action,
    _pot_odds_line,
    _BASE_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "leduc_poker"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 4096
_TIMEOUT = 2400
_MAX_TURNS = 10

# Validator-aligned reward scale ([-1, 1])
TERMINAL_WIN_REWARD   = 1.0
TERMINAL_LOSS_REWARD  = -1.0
INVALID_PENALTY       = -0.1
INVALID_TOTAL_CLIP    = -0.3
TERMINAL_REWARD_CLIP  = 1.0

# Belief-aware shaping bonuses (small, additive on top of terminal).
BELIEF_FOLD_K_BONUS    = 0.10  # we folded when belief said opp likely has K
BELIEF_RAISE_J_BONUS   = 0.10  # we raised when belief said opp likely has J


# ---------------------------------------------------------------------------
# Bayesian opponent card belief
# ---------------------------------------------------------------------------

class CardBelief:
    """Posterior distribution over opp's private rank ∈ {J, Q, K}.

    Maintains:
      - self._belief: dict[str, float] — posterior probabilities, sum = 1.
      - self.public_rank: str | None — public card rank once revealed.

    Updates:
      - update_public_card(rank): joint Bayesian conditioning.
      - update_action(action, round): multiplies by action-likelihood table.
    """

    RANKS = ("J", "Q", "K")
    RANK_VALUES = {"J": 1, "Q": 2, "K": 3}

    # CFR-style action likelihoods (rough Nash approximation).
    R1_LIKELIHOOD = {
        "K": {"Fold": 0.05, "Call": 0.20, "Check": 0.20, "Raise": 0.75},
        "Q": {"Fold": 0.10, "Call": 0.55, "Check": 0.55, "Raise": 0.35},
        "J": {"Fold": 0.40, "Call": 0.50, "Check": 0.50, "Raise": 0.10},
    }
    R2_NO_PAIR_LIKELIHOOD = {
        "K": {"Fold": 0.10, "Call": 0.40, "Check": 0.40, "Raise": 0.50},
        "Q": {"Fold": 0.30, "Call": 0.55, "Check": 0.55, "Raise": 0.15},
        "J": {"Fold": 0.70, "Call": 0.25, "Check": 0.25, "Raise": 0.05},
    }
    R2_PAIR_LIKELIHOOD = {
        # When opp's rank == public_rank, opp has a pair → near-always raises.
        "K": {"Fold": 0.0, "Call": 0.10, "Check": 0.10, "Raise": 0.90},
        "Q": {"Fold": 0.0, "Call": 0.10, "Check": 0.10, "Raise": 0.90},
        "J": {"Fold": 0.0, "Call": 0.10, "Check": 0.10, "Raise": 0.90},
    }
    SMOOTH_FLOOR = 0.01  # avoid zero-likelihood collapse

    def __init__(self, our_rank: int) -> None:
        self.our_rank_int = our_rank
        self.our_rank = self._rank_int_to_str(our_rank)
        self._belief = self._initial_prior(self.our_rank)
        self.public_rank: Optional[str] = None
        self._public_processed = False

    @staticmethod
    def _rank_int_to_str(r: int) -> str:
        return {1: "J", 2: "Q", 3: "K"}.get(r, "?")

    @classmethod
    def _initial_prior(cls, our_rank_str: str) -> dict[str, float]:
        counts = {"J": 2, "Q": 2, "K": 2}
        if our_rank_str in counts:
            counts[our_rank_str] -= 1
        total = sum(counts.values())
        if total <= 0:
            return {r: 1 / 3 for r in cls.RANKS}
        return {r: c / total for r, c in counts.items()}

    def update_public_card(self, public_rank: int) -> None:
        """Joint Bayesian update conditioning on the public card."""
        if self._public_processed:
            return
        public_str = self._rank_int_to_str(public_rank)
        self.public_rank = public_str
        counts_after_our_draw = {"J": 2, "Q": 2, "K": 2}
        if self.our_rank in counts_after_our_draw:
            counts_after_our_draw[self.our_rank] -= 1

        joint: dict[str, float] = {}
        for rank, p_rank in self._belief.items():
            counts_after_opp = dict(counts_after_our_draw)
            counts_after_opp[rank] = max(counts_after_opp.get(rank, 0) - 1, 0)
            denom = sum(counts_after_opp.values())
            if denom == 0:
                joint[rank] = 0.0
            else:
                p_public_given_rank = counts_after_opp.get(public_str, 0) / denom
                joint[rank] = p_rank * p_public_given_rank
        norm = sum(joint.values())
        if norm > 0:
            self._belief = {r: v / norm for r, v in joint.items()}
        self._public_processed = True

    def update_action(self, action: str, game_round: int) -> None:
        """Multiply belief by action-likelihood for the given round."""
        if action not in ("Fold", "Call", "Check", "Raise"):
            return
        new: dict[str, float] = {}
        norm = 0.0
        for rank, p_rank in self._belief.items():
            if game_round == 1:
                lk_table = self.R1_LIKELIHOOD.get(rank, {})
            else:
                if self.public_rank is not None and rank == self.public_rank:
                    lk_table = self.R2_PAIR_LIKELIHOOD.get(rank, {})
                else:
                    lk_table = self.R2_NO_PAIR_LIKELIHOOD.get(rank, {})
            p_action = max(lk_table.get(action, self.SMOOTH_FLOOR), self.SMOOTH_FLOOR)
            new[rank] = p_rank * p_action
            norm += new[rank]
        if norm > 0:
            self._belief = {r: v / norm for r, v in new.items()}

    def belief(self) -> dict[str, float]:
        return dict(self._belief)

    def most_likely_rank(self) -> tuple[str, float]:
        return max(self._belief.items(), key=lambda x: x[1])

    def p_pair(self) -> float:
        if self.public_rank is None:
            return 0.0
        return self._belief.get(self.public_rank, 0.0)

    def expected_strength(self) -> float:
        """Expected hand strength: rank value + pair boost weighted by belief."""
        s = 0.0
        for rank, p in self._belief.items():
            base = self.RANK_VALUES[rank]
            if self.public_rank is not None and rank == self.public_rank:
                base += 3.0
            s += p * base
        return s

    def summary(self) -> str:
        b = self._belief
        ml, ml_p = self.most_likely_rank()
        pair_str = ""
        if self.public_rank is not None and self.public_rank in b:
            pair_str = f"  P(pair)={self.p_pair():.0%}"
        return (
            f"[Belief] Opp card: J:{b.get('J', 0):.0%} Q:{b.get('Q', 0):.0%} "
            f"K:{b.get('K', 0):.0%} \u2192 likely {ml} ({ml_p:.0%}){pair_str}  "
            f"E[strength]={self.expected_strength():.2f}"
        )


def extract_opp_actions(betting: list[str], our_player_id: int) -> list[str]:
    """Filter betting list to only opp's actions, in chronological order.

    P0 acts first in each round → P0's actions are at even indices,
    P1's actions are at odd indices.
    """
    opp_pid = 1 - our_player_id
    return [a for i, a in enumerate(betting) if (i % 2) == opp_pid]


# ---------------------------------------------------------------------------
# Reward calculator (validator-aligned scale [-1, 1])
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Episode-level shaped reward for opponent-modeling Leduc Poker.

    Final reward is clipped to [-1, 1] for validator alignment.
    Components: terminal win/loss, invalid penalty, belief-aware bonuses.
    """

    def __init__(self) -> None:
        self.invalid_penalty = INVALID_PENALTY

    def calculate_episode_reward(
        self,
        won: bool,
        invalid_count: int,
        belief_fold_k_count: int,
        belief_raise_j_count: int,
        components: Optional[dict] = None,
    ) -> float:
        terminal = TERMINAL_WIN_REWARD if won else TERMINAL_LOSS_REWARD
        fold_k_bonus  = BELIEF_FOLD_K_BONUS  * min(belief_fold_k_count, 2)  if won else 0.0
        raise_j_bonus = BELIEF_RAISE_J_BONUS * min(belief_raise_j_count, 2) if won else 0.0
        invalid_total = max(invalid_count * INVALID_PENALTY, INVALID_TOTAL_CLIP)

        raw = terminal + fold_k_bonus + raise_j_bonus + invalid_total
        clipped = max(min(raw, TERMINAL_REWARD_CLIP), -TERMINAL_REWARD_CLIP)

        if components is not None:
            components["terminal"]      = components.get("terminal", 0.0)      + terminal
            components["fold_k_bonus"]  = components.get("fold_k_bonus", 0.0)  + fold_k_bonus
            components["raise_j_bonus"] = components.get("raise_j_bonus", 0.0) + raise_j_bonus
            components["invalid_total"] = components.get("invalid_total", 0.0) + invalid_total
            components["clip_delta"]    = components.get("clip_delta", 0.0)    + (clipped - raw)
        return clipped


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=_MAX_TURNS,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.75,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _current_mcts_sims(curriculum: CurriculumScheduler) -> int:
    turn_range = max(curriculum.final_max_turn - curriculum.initial_max_turn, 1)
    progress = (curriculum.get_max_turn() - curriculum.initial_max_turn) / turn_range
    progress = max(0.0, min(progress, 1.0))
    return int(10 + progress * (50 - 10))


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
        f"[CURRICULUM] Initialized (opp_modeling): initial_max_turn={trainer.args.initial_max_turn}, "
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
# Hint prompt (extends base with belief notes)
# ---------------------------------------------------------------------------

_HINT_PROMPT = (
    "\n\n# Strategy Guide (with Belief Model)\n"
    "ROUND 1 baseline:\n"
    "- K \u2192 Raise (strongest non-pair; build pot)\n"
    "- Q \u2192 Call (middle; wait for public card)\n"
    "- J \u2192 Call; fold to a second raise (weak; bad pot odds)\n\n"
    "ROUND 2 with public card visible:\n"
    "- Pair (your card matches public) \u2192 always Raise (dominant)\n"
    "- No pair + K \u2192 Call opponent raises\n"
    "- No pair + Q \u2192 Call if opp only called; Fold to raises\n"
    "- No pair + J \u2192 Fold to any Raise\n\n"
    "USING THE BELIEF LINE (in your observation):\n"
    "- 'Belief: ...likely K (X%) ...' \u2192 opp probably holds K. Avoid bluffing into them.\n"
    "  If you hold Q/J without pair, fold to raises.\n"
    "- 'Belief: ...likely J (X%) ...' \u2192 opp probably weak. Raise more aggressively.\n"
    "  If you hold K/Q, even a Q-call against their raise is +EV.\n"
    "- 'Belief: ...P(pair)=X%' (R2) \u2192 X is opp's chance of having a pair.\n"
    "  If P(pair) is high (\u226550%) and you don't have one, fold unless pot odds force.\n"
    "- 'Belief: E[strength]=...' \u2192 expected opp strength on a 1-6 scale (1=J, 6=K+pair).\n"
    "  Compare to your own hand strength to decide bet/fold.\n\n"
    "POT ODDS LINE (when present):\n"
    "- 'Win-rate to break-even: N%' tells you the minimum win rate at which calling is +EV.\n"
    "- Compare N% to (1 - opp_strength_advantage_estimate).\n\n"
    "EXPLOITING MCTS (1 random rollout, no cross-episode adaptation):\n"
    "- MCTS with 50 sims rarely converges to Nash on this 936-state game \u2014 the belief above is\n"
    "  a Nash-style approximation MCTS does not have.\n"
    "- MCTS folds too rarely vs aggressive R2 raises with pairs \u2014 raise pairs aggressively.\n"
    "- MCTS is overly passive with J \u2014 raise with K/Q in R1 to push them off marginal hands.\n"
    "- MCTS does NOT adapt to your betting pattern \u2014 consistent value-bets are safe.\n"
)


# ---------------------------------------------------------------------------
# Belief-aware decision tagging
# ---------------------------------------------------------------------------

def _tag_belief_decision(
    action_str: str,
    belief: CardBelief,
    has_pair: bool,
) -> tuple[bool, bool]:
    """Identify (folded_against_high_K_belief, raised_against_high_J_belief).

    These tags fire only when belief is confident (>= 0.5) and the agent's
    decision aligns with the model's recommendation.
    """
    b = belief.belief()
    folded_against_K = (
        action_str == "Fold"
        and b.get("K", 0.0) >= 0.5
        and not has_pair
    )
    raised_against_J = (
        action_str == "Raise"
        and b.get("J", 0.0) >= 0.5
    )
    return folded_against_K, raised_against_J


# ---------------------------------------------------------------------------
# Observation augmentation
# ---------------------------------------------------------------------------

def _augment_observation(
    observation: str,
    gs: "GameState | None",
    belief: "CardBelief | None",
) -> str:
    """Append pot-odds line and belief line to the observation."""
    parts = [observation]
    if gs is not None:
        pot_line = _pot_odds_line(gs)
        if pot_line:
            parts.append(pot_line)
    if belief is not None:
        parts.append(belief.summary())
    return "\n\n".join(parts)


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
    current_mcts_sims: int,
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

    # Last-prompt state
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    done = False
    final_reward = 0.0
    turn_number = 0
    invalid_count = 0
    use_hints = random.random() < current_hint_prob
    game_state_history: list[GameState] = []

    belief: Optional[CardBelief] = None
    opp_actions_seen_r1 = 0
    opp_actions_seen_r2 = 0
    belief_fold_k_count = 0
    belief_raise_j_count = 0
    components: dict[str, float] = {}
    calculator = RewardCalculator()

    # --- Reset env ---
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": current_mcts_sims,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block = reset_res.json()["result"]
        episode_id = result_block.get("episode_id", "")
        observation = _format_observation(result_block.get("observation", ""))
        gs = parse_game_state(observation)
        if gs is not None:
            game_state_history.append(gs)
            belief = CardBelief(gs.private_card_rank)
            if gs.public_card_rank:
                belief.update_public_card(gs.public_card_rank)
            opp_r1 = extract_opp_actions(gs.r1_betting, gs.player_id)
            for a in opp_r1[opp_actions_seen_r1:]:
                belief.update_action(a, game_round=1)
            opp_actions_seen_r1 = len(opp_r1)
            if gs.round >= 2:
                opp_r2 = extract_opp_actions(gs.r2_betting, gs.player_id)
                for a in opp_r2[opp_actions_seen_r2:]:
                    belief.update_action(a, game_round=2)
                opp_actions_seen_r2 = len(opp_r2)
            observation = _augment_observation(observation, gs, belief)
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
                        f"Warning: token shift at turn {turn_number}. Skipping delta mask."
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

        # Tag belief-aligned decision BEFORE stepping
        if prev_gs is not None and belief is not None:
            try:
                action_str = prev_gs.legal_actions.get(int(action_to_send.strip()), "")
            except (ValueError, AttributeError):
                action_str = ""
            if action_str:
                fold_k, raise_j = _tag_belief_decision(action_str, belief, prev_gs.has_pair)
                if fold_k:
                    belief_fold_k_count += 1
                if raise_j:
                    belief_raise_j_count += 1

        # --- Step env ---
        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = _format_observation(step_block.get("observation", ""))
            step_reward = step_block.get("reward", 0)
            done        = step_block.get("done", False)
            new_gs: "GameState | None" = None
            if not done:
                new_gs = parse_game_state(observation)
                if new_gs is not None:
                    game_state_history.append(new_gs)
                    if belief is not None and new_gs.public_card_rank and not belief._public_processed:
                        belief.update_public_card(new_gs.public_card_rank)
                    if belief is not None:
                        opp_r1 = extract_opp_actions(new_gs.r1_betting, new_gs.player_id)
                        for a in opp_r1[opp_actions_seen_r1:]:
                            belief.update_action(a, game_round=1)
                        opp_actions_seen_r1 = len(opp_r1)
                        if new_gs.round >= 2:
                            opp_r2 = extract_opp_actions(new_gs.r2_betting, new_gs.player_id)
                            for a in opp_r2[opp_actions_seen_r2:]:
                                belief.update_action(a, game_round=2)
                            opp_actions_seen_r2 = len(opp_r2)
                    observation = _augment_observation(observation, new_gs, belief)
        except Exception as exc:
            print(f"Step failed (Game {game_id}, turn {turn_number}): {exc}")
            observation = ""
            step_reward = 0
            done = False
            invalid_count += 1

        if "Nothing happens" in observation or "Invalid" in observation:
            invalid_count += 1

        if done:
            final_reward = step_reward

        messages.append({"role": "user", "content": observation})
        turn_number += 1

    # --- Episode reward ---
    won = final_reward > 0.5
    train_reward = calculator.calculate_episode_reward(
        won=won,
        invalid_count=invalid_count,
        belief_fold_k_count=belief_fold_k_count,
        belief_raise_j_count=belief_raise_j_count,
        components=components,
    )
    components["belief_fold_k_count"]  = float(belief_fold_k_count)
    components["belief_raise_j_count"] = float(belief_raise_j_count)
    if belief is not None:
        ml, ml_p = belief.most_likely_rank()
        components["final_belief_topP"] = float(ml_p)

    print(
        "[ID:{:<6} Done:{} T:{:>2d} | Hints:{} | EnvR:{:>5.2f} | TrainR:{:>5.2f} | "
        "FoldK:{} RaiseJ:{} Inv:{}]".format(
            str(game_id)[:6], int(done), turn_number, int(use_hints),
            final_reward, train_reward,
            belief_fold_k_count, belief_raise_j_count, invalid_count,
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
            "invalid_count":  invalid_count,
            "components":     components,
        }
    return index, {
        "prompt_ids":     prompt_ids,
        "completion_ids": completion_ids,
        "logprobs":       logprobs,
        "reward":         train_reward,
        "final_score":    final_reward,
        "invalid_count":  invalid_count,
        "components":     components,
    }


# ---------------------------------------------------------------------------
# Public rollout functions
# ---------------------------------------------------------------------------

def _dispatch(prompts, trainer, *, use_full_prompt: bool) -> dict[str, list]:
    _ensure_initialized(trainer)

    curriculum: CurriculumScheduler = _state["curriculum"]
    current_max_turn  = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    current_mcts_sims = _current_mcts_sims(curriculum)
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, "
        f"mcts_sims={current_mcts_sims}"
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
        current_hint_prob=current_hint_prob,
        current_mcts_sims=current_mcts_sims,
    )

    _fallback = (
        {"prompt_ids": [1], "completion_ids": [1], "action_mask": [0], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "invalid_count": 0, "components": {}}
        if use_full_prompt else
        {"prompt_ids": [1], "completion_ids": [1], "logprobs": [1.0], "reward": 0.0, "final_score": 0.0, "invalid_count": 0, "components": {}}
    )

    results = [None] * len(prompts)
    futures = [_state["thread_pool"].submit(run, i, p) for i, p in enumerate(prompts)]
    for f in as_completed(futures):
        idx, res = f.result()
        results[idx] = res if res is not None else _fallback

    curriculum.step(len(prompts))

    list_results = [r for r in results if r is not None]
    n = len(list_results)
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    wins     = sum(1 for r in list_results if r["final_score"] > 0.5)
    losses   = sum(1 for r in list_results if r["final_score"] < -0.5)
    avg_return = sum(r["reward"] for r in list_results) / n if n else 0
    win_rate = (wins / finished) if finished else 0.0
    avg_invalid = sum(r.get("invalid_count", 0) for r in list_results) / n if n else 0
    print(
        f"[BATCH] Finished:{finished}/{n} W:{wins} L:{losses} "
        f"WinRate:{win_rate:.2%} AvgReturn:{avg_return:.3f} AvgInv:{avg_invalid:.2f}"
    )

    component_keys = ["terminal", "fold_k_bonus", "raise_j_bonus",
                      "invalid_total", "clip_delta",
                      "belief_fold_k_count", "belief_raise_j_count", "final_belief_topP"]
    if n:
        avgs = {
            k: sum(r.get("components", {}).get(k, 0.0) for r in list_results) / n
            for k in component_keys
        }
        comp_str = " ".join(f"{k}:{v:+.3f}" for k, v in avgs.items())
        print(f"[SHAPING] {comp_str}")

    out = {
        "prompt_ids":     [r["prompt_ids"]     for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs":       [r["logprobs"]       for r in list_results],
        "env_rewards":    [r["reward"]         for r in list_results],
        "terminal_raw":   [float(r["final_score"])               for r in list_results],
        "shaping_sum":    [float(r["reward"] - r["final_score"]) for r in list_results],
        "invalid_count":  [int(r.get("invalid_count", 0))        for r in list_results],
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
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
