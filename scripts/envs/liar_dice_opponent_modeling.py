"""Liar's Dice with Bayesian-style opponent modeling.

Tracks opponent's bid history within an episode and surfaces inferred dice
composition + bluff rate to the LLM prompt. The MCTS opponent has no
cross-episode memory, so explicit per-episode inference gives the LLM an
asymmetric-information edge.

Algorithms used
---------------
- Rational-bid inference: each opp bid Q on face F implies
    implied_opp_count[F] >= max(0, Q - our_matching_count[F])
  (because total = our_count + opp_count and bid claims total >= Q).
  When implied_opp_count > num_dice_per_player, the bid is impossible
  → guaranteed bluff.
- Bluff-rate moving average over opp's bid history.
- True-probability re-evaluation via the existing scipy binomial helper
  imported from liar_dice_env.

Imports parser, action extractor, and core dataclasses from
``liar_dice_env`` so per-game logic stays in one place.
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
from envs.liar_dice_env import (
    Bid,
    Action,
    GameState,
    bid_probability,
    parse_game_state,
    extract_action_id,
    _reformat_observation,
    _SYSTEM_PROMPT,
    BLUFF_PROB_THRESHOLD,
    RISKY_LIAR_PROB_MIN,
    RISKY_LIAR_PROB_MAX,
    BLUFF_WIN_BONUS,
    RISKY_LIAR_WIN_BONUS,
    RISKY_BONUS_MAX_COUNT,
    SHUFFLE_PROB,
    NORMALIZE_REWARDS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME = "liars_dice"
_MAX_EPISODE_TOKENS = 16384
_MAX_PROMPT_LEN = 5000
_TIMEOUT = 2400

# Validator-aligned reward scale ([-1, 1]) — mirrors gin_rummy_opponent_modeling.
TERMINAL_WIN_REWARD   = 1.0
TERMINAL_LOSS_REWARD  = -1.0
INVALID_PENALTY       = -0.1
INVALID_TOTAL_CLIP    = -0.3
TERMINAL_REWARD_CLIP  = 1.0

# Opponent-modeling shaping bonuses (small, additive on top of terminal).
BLUFF_CAUGHT_BONUS    = 0.10   # we called Liar on opp's detected bluff and won
EXPLOIT_FOLD_BONUS    = 0.05   # reserved for future use; not fired by current logic
NUM_DICE_PER_PLAYER   = 5      # default for the project's liars_dice variant


# ---------------------------------------------------------------------------
# Bidding history tracker (per-episode)
# ---------------------------------------------------------------------------

class BiddingHistoryTracker:
    """Track opponent's bids and infer their dice composition.

    Each bid is interpreted under the rational-bid assumption: opp wouldn't
    bid Q on face F unless their expected matching count for F is high enough
    that the bid has at least some chance of being true. We compute the
    implied minimum opp matching count, flag impossible bids as bluffs, and
    expose a compact summary that fits in the LLM prompt.
    """

    def __init__(self, num_dice_per_player: int = NUM_DICE_PER_PLAYER) -> None:
        self.num_dice = num_dice_per_player
        self.opp_bid_history: list[Bid] = []
        self.detected_bluffs = 0
        # Per-face minimum implied opp matching count (rank 1..6)
        self.implied_min_matches: dict[int, int] = {}

    @staticmethod
    def _our_matching_count(face: int, our_dice: list[int]) -> int:
        """Count of our dice matching the bid face (6 is wild for non-6 faces)."""
        if face == 6:
            return sum(1 for d in our_dice if d == 6)
        return sum(1 for d in our_dice if d == face or d == 6)

    def record_opp_bid(self, opp_bid: "Bid | None", our_dice: list[int]) -> None:
        if opp_bid is None:
            return
        self.opp_bid_history.append(opp_bid)
        our_count = self._our_matching_count(opp_bid.face, our_dice)
        implied = max(0, opp_bid.quantity - our_count)
        if implied > self.num_dice:
            # Bid claims more F dice than opp could possibly hold → guaranteed bluff
            self.detected_bluffs += 1
        capped = min(implied, self.num_dice)
        cur = self.implied_min_matches.get(opp_bid.face, 0)
        self.implied_min_matches[opp_bid.face] = max(cur, capped)

    def is_likely_bluff(self, bid: "Bid | None", our_dice: list[int]) -> bool:
        """True when the given bid implies an opp matching count > num_dice."""
        if bid is None:
            return False
        our_count = self._our_matching_count(bid.face, our_dice)
        return (bid.quantity - our_count) > self.num_dice

    def bluff_rate(self) -> float:
        if not self.opp_bid_history:
            return 0.0
        return self.detected_bluffs / len(self.opp_bid_history)

    def style_label(self) -> str:
        if len(self.opp_bid_history) < 2:
            return "UNKNOWN"
        br = self.bluff_rate()
        if br >= 0.4:
            return "BLUFFER"
        if br <= 0.1:
            return "CONSERVATIVE"
        return "MIXED"

    def summary(self) -> str:
        if not self.opp_bid_history:
            return ""
        lines: list[str] = []
        if self.implied_min_matches:
            parts = [
                f"face{f}\u2265{c}"
                for f, c in sorted(self.implied_min_matches.items()) if c > 0
            ]
            if parts:
                lines.append(f"[Opp model] Implied opp dice (min): {' '.join(parts)}")
        lines.append(
            f"[Opp model] Style: {self.style_label()} "
            f"(detected bluffs: {self.detected_bluffs}/{len(self.opp_bid_history)})"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reward calculator (validator-aligned scale [-1, 1])
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Episode-level shaped reward for opponent-modeling Liar's Dice.

    Final reward is clipped to [-1, 1] to align with validator scoring.
    Components: terminal win/loss, invalid penalty, opponent-model bonuses.
    """

    def __init__(self) -> None:
        self.invalid_penalty = INVALID_PENALTY

    def calculate_step_reward(self, action: "Action | None", env_reward: float) -> float:
        # Per-step shaping is deferred to episode-level for this variant.
        return 0.0

    def calculate_episode_reward(
        self,
        won: bool,
        invalid_count: int,
        bluff_caught_count: int,
        components: Optional[dict] = None,
    ) -> float:
        terminal = TERMINAL_WIN_REWARD if won else TERMINAL_LOSS_REWARD
        bluff_caught_bonus = BLUFF_CAUGHT_BONUS * min(bluff_caught_count, 2) if won else 0.0
        invalid_total = max(invalid_count * INVALID_PENALTY, INVALID_TOTAL_CLIP)

        raw = terminal + bluff_caught_bonus + invalid_total
        clipped = max(min(raw, TERMINAL_REWARD_CLIP), -TERMINAL_REWARD_CLIP)

        if components is not None:
            components["terminal"]      = components.get("terminal", 0.0)      + terminal
            components["bluff_caught"]  = components.get("bluff_caught", 0.0)  + bluff_caught_bonus
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
        final_max_turn=15,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.5,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _current_mcts_sims(curriculum: CurriculumScheduler) -> int:
    turn_range = max(curriculum.final_max_turn - curriculum.initial_max_turn, 1)
    progress = (curriculum.get_max_turn() - curriculum.initial_max_turn) / turn_range
    progress = max(0.0, min(progress, 1.0))
    return int(10 + progress * (225 - 10))


def _ensure_initialized(trainer) -> None:
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": 42,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)
    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Initialized (opp_modeling): initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn=15, rollouts_per_stage={trainer.args.rollouts_per_stage}"
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
# Hint prompt (extends base with opponent-modeling notes)
# ---------------------------------------------------------------------------

_HINT_PROMPT = (
    "\n\n# Strategy Guide (with Opponent Model)\n\n"
    "BASE STRATEGY \u2014 use your dice to anchor bids:\n"
    "- your_match(F) = count(F) + count(6) for non-6 face\n"
    "- Conservative: claim your_match (provably supported)\n"
    "- Normal: claim your_match + 1 (~85% true with 5 hidden dice)\n"
    "- Aggressive: claim your_match + 2 (~54% true)\n\n"
    "USING THE OPP MODEL (in your observation):\n"
    "- 'Implied opp dice (min): faceF\u2265K' = opp's bids prove they have \u2265K of face F.\n"
    "  Adjust your matching estimate: total_F = your_match(F) + K (lower bound).\n"
    "- 'Style: BLUFFER' \u2192 challenge sooner; their high bids are often lies.\n"
    "  When opp bids near-impossible counts, call Liar even if math is borderline.\n"
    "- 'Style: CONSERVATIVE' \u2192 don't challenge their bids unless math strongly disagrees.\n"
    "  They likely DO have what they claim; outbid carefully.\n"
    "- 'Style: MIXED' or UNKNOWN \u2192 fall back to math-only thresholds.\n\n"
    "CHALLENGE MATH (needed = bid_quantity - your_match - implied_opp_min):\n"
    "- needed \u2264 0: bid is fully accounted for \u2192 do NOT challenge.\n"
    "- needed = 1 from 5 dice: ~87% true \u2192 do NOT challenge.\n"
    "- needed = 2: ~54% \u2192 use opp style to break tie.\n"
    "- needed \u2265 3: ~21% or worse \u2192 CHALLENGE (call Liar).\n\n"
    "EXPLOITING MCTS (1 random rollout per node, no cross-episode memory):\n"
    "- MCTS cannot track your bid history \u2014 a consistent 'always bid +2 above match' line is safe.\n"
    "- MCTS does not adapt to your bluff rate \u2014 mix in occasional clear bluffs to break its priors.\n"
    "- The opp model above is YOURS only; MCTS has no equivalent. Use it.\n"
)


# ---------------------------------------------------------------------------
# Observation augmentation
# ---------------------------------------------------------------------------

def _augment_observation(
    obs: str,
    gs: "GameState | None",
    tracker: BiddingHistoryTracker,
    use_hints: bool,
) -> str:
    """Reuse base reformatter (action shuffle + p_true/p_lie annotations + hint),
    then append the opp-model summary."""
    base_obs = _reformat_observation(obs, gs, use_hints) if gs is not None else obs
    opp_summary = tracker.summary()
    if opp_summary:
        return f"{base_obs}\n\n{opp_summary}"
    return base_obs


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
    current_mcts_sims: int,
) -> tuple[int, "dict | None"]:
    game_id = int(prompt)
    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:    list[int]   = []
    episode_completion_ids: list[int]  = []
    episode_logprobs:      list[float] = []
    episode_action_mask:   list[int]   = []
    prev_full_ids: "list[int] | None"  = None

    # Last-prompt fallback
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    invalid_count       = 0
    done                = False
    final_reward        = 0.0
    turn_number         = 0
    bluff_caught_count  = 0
    game_state_history: list[GameState] = []
    components: dict[str, float]        = {}

    use_hints  = random.random() < current_hint_prob
    tracker    = BiddingHistoryTracker(num_dice_per_player=NUM_DICE_PER_PLAYER)
    calculator = RewardCalculator()

    # --- Reset environment ---
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
        raw_observation = result_block.get("observation", "")
        initial_gs = parse_game_state(raw_observation)
        game_state_history.append(initial_gs)
        # Initial bid (if opp went first) — record before agent sees it
        if initial_gs.current_bid is not None:
            tracker.record_opp_bid(initial_gs.current_bid, initial_gs.our_dice)
        formatted_observation = _augment_observation(raw_observation, initial_gs, tracker, use_hints)
    except Exception as exc:
        import traceback; traceback.print_exc()
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

        # Detect "we called Liar on a known-bluff" pattern (for bonus shaping).
        prev_state_for_decision = game_state_history[-1] if game_state_history else None
        called_liar_on_bluff = False
        if prev_state_for_decision is not None:
            standing_bid = prev_state_for_decision.current_bid
            try:
                action_id = int(extract_action_id(completion_text) or "-1")
            except ValueError:
                action_id = -1
            taken_action = next(
                (a for a in prev_state_for_decision.actions if a.action_id == action_id), None
            )
            if (
                taken_action is not None
                and taken_action.is_liar
                and tracker.is_likely_bluff(standing_bid, prev_state_for_decision.our_dice)
            ):
                called_liar_on_bluff = True

        # --- Token accumulation ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {_MAX_PROMPT_LEN} tokens at turn {turn_number}, ending early")
                done = True
                break

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

            if completion_ids:
                episode_completion_ids.extend(completion_ids)
                episode_logprobs.extend(logprobs)
                episode_action_mask.extend([1] * len(completion_ids))
                if prev_full_ids is not None:
                    prev_full_ids = prev_full_ids + completion_ids

        messages.append({"role": "assistant", "content": completion_text})

        action_to_send = extract_action_id(completion_text)

        # --- Step env ---
        is_invalid = False
        try:
            raw_observation = ""
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            raw_observation = step_block.get("observation", "")
            step_reward     = step_block.get("reward", 0)
            done            = step_block.get("done", False)
        except Exception as exc:
            print(f"Step failed: {exc}")
            step_reward = -0.01
            done = False
            invalid_count += 1
            raw_observation = ""

        if "Nothing happens" in raw_observation or "Invalid" in raw_observation:
            invalid_count += 1
            is_invalid = True

        # Parse new state and record opp's bid (if any new bid appeared)
        new_gs: "GameState | None" = None
        if not done and not is_invalid and raw_observation:
            try:
                new_gs = parse_game_state(raw_observation)
                prev_bid = game_state_history[-1].current_bid if game_state_history else None
                new_bid  = new_gs.current_bid
                if new_bid is not None and new_bid != prev_bid:
                    tracker.record_opp_bid(new_bid, new_gs.our_dice)
                game_state_history.append(new_gs)
            except Exception as exc:
                print(f"Failed to parse game state: {exc}")
                is_invalid = True
                invalid_count += 1

        if done:
            final_reward = step_reward
            if called_liar_on_bluff and step_reward > 0.5:
                bluff_caught_count += 1
            messages.append({"role": "user", "content": raw_observation})
        else:
            augmented = _augment_observation(raw_observation, new_gs, tracker, use_hints)
            messages.append({"role": "user", "content": augmented})

        turn_number += 1

    # --- Episode reward ---
    won = final_reward > 0.5
    train_reward = calculator.calculate_episode_reward(
        won=won,
        invalid_count=invalid_count,
        bluff_caught_count=bluff_caught_count,
        components=components,
    )
    components["bluff_caught_count"] = float(bluff_caught_count)
    components["opp_bluff_rate"]     = float(tracker.bluff_rate())

    print(
        "[ID:{:<6} Hints:{} Done:{} T:{:>2d} | Reward:{:>6.3f} | "
        "EnvR:{:>5.1f} | OppBids:{:<2} Bluffs:{:<2} Caught:{:<2} Inv:{:<2}]".format(
            str(game_id)[:6], int(use_hints), int(done), turn_number,
            train_reward, final_reward,
            len(tracker.opp_bid_history), tracker.detected_bluffs,
            bluff_caught_count, invalid_count,
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
        current_max_turn=current_max_turn,
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

    component_keys = ["terminal", "bluff_caught", "invalid_total", "clip_delta",
                      "bluff_caught_count", "opp_bluff_rate"]
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
    max_turns: int = 15,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 15,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
