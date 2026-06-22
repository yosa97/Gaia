"""Othello (Reversi) environment training module.

Othello is a deterministic, perfect-information 8x8 board game served by the same
OpenSpiel-backed env server as the other PvP games (task_id range 400M-499M, see
``GAMES_TO_TASK_ID_RANGE``). Unlike gin_rummy / liars_dice / leduc_poker there is
no hidden state and no opponent card/dice modelling, so the rollout here is a
clean board-play loop: present the observation, let the policy pick a legal
action, step the server, and reward the terminal win/loss.

The policy is trained in the SAME tool-calling dialect the PvP evaluator expects
(``core/pvp/bot.py`` → ``game_action`` tool). The system prompt advertises the
tool via ``TOOL_GUIDANCE`` and ``trainer.tools`` (set in train_grpo_env.py for
every env in ``GAMES_TO_TASK_ID_RANGE``), and the completion is parsed with the
shared :func:`extract_action_id`, so a plain-text answer can never cause an
``InvalidActionForfeitError`` at eval.

Public names (imported by env_configs.py):
  * rollout_full_prompt_and_completion_parallelized_curriculum
  * rollout_last_prompt_and_completion_parallelized_curriculum
  * rollout_reward_func  (re-exported from shared_env)
  * _curriculum_factory
"""

import functools
from concurrent.futures import as_completed
from threading import Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

from envs.shared_env import (
    GAMES_TO_TASK_ID_RANGE,
    CurriculumScheduler,
    init_env_pool,
    rollout_reward_func,  # re-exported for callers  # noqa: F401
)
from envs.pvp_tool_format import (
    MINER_SEED,
    TOOL_GUIDANCE,
    extract_action_id as _pvp_extract_action_id,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SELECTED_GAME      = "othello"
_MAX_EPISODE_TOKENS = 16384   # cap on accumulated full-prompt completion tokens
_MAX_PROMPT_LEN     = 8000    # per-turn prompt token cap (board obs are compact)
_TIMEOUT            = 2400    # HTTP timeout (s) — covers slow MCTS opponent moves
_FINAL_MAX_TURN     = 64      # an 8x8 othello game lasts at most 60 placements


# ---------------------------------------------------------------------------
# System prompt (rules + tool-calling output format)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are playing othello.\n\n"
    "# Game Rules\n"
    "OTHELLO (REVERSI) RULES:\n\n"
    "Board: 8x8 grid. Two colours, Black and White. You control one colour.\n"
    "Start: the four centre squares hold two Black and two White discs diagonally.\n\n"
    "A legal move places one of your discs on an empty square so that it brackets "
    "one or more of the opponent's discs in a straight line (horizontal, vertical, "
    "or diagonal) between the newly placed disc and another of your discs. Every "
    "bracketed opponent disc is flipped to your colour.\n"
    "You MUST make a move that flips at least one opponent disc. If and only if you "
    "have no legal move, the only legal action is to pass.\n\n"
    "Action IDs: each legal action is shown in the observation as 'ID -> square' "
    "(e.g. a board square such as d3, or 'pass'). Choose by its action_id.\n\n"
    "Goal: when neither player can move, the player with MORE discs on the board "
    "wins. Maximise your final disc count.\n\n"
    "Strategy notes (not rules): corners cannot be flipped and are very strong; "
    "edges are stable; avoid squares adjacent to an empty corner early.\n\n\n\n"
    + TOOL_GUIDANCE
)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Discounted-return calculator for Othello.

    Othello is a sparse-reward game: the env returns 0 during play and a terminal
    win/loss signal at the end. We amplify that terminal signal and discount it
    back over the episode so earlier decisive moves receive credit.
    """

    def __init__(self, gamma: float = 0.97):
        self.gamma           = gamma   # near-1: othello outcomes hinge on late play
        self.terminal_weight = 3.0     # amplify the ±1 win/loss into ±3

    def calculate_discounted_return(self, rewards: list[float]) -> float:
        if not rewards:
            return 0.0
        T = len(rewards)
        return sum(self.gamma ** (T - 1 - i) * r for i, r in enumerate(rewards))


# ---------------------------------------------------------------------------
# Module-level state (shared between full and last rollout functions)
# ---------------------------------------------------------------------------

_state: dict = {}


def _curriculum_factory(args) -> CurriculumScheduler:
    """Construct this env's curriculum from training args. Referenced by env_configs registry."""
    return CurriculumScheduler(
        initial_max_turn=args.initial_max_turn,
        final_max_turn=_FINAL_MAX_TURN,
        rollouts_per_stage=args.rollouts_per_stage,
        initial_hint_prob=0.0,
        final_hint_prob=0.0,
        warmup_rollouts=args.rollouts_per_stage,
    )


def _ensure_initialized(trainer) -> None:
    """Set up the server pool and curriculum once per process (no-op afterwards)."""
    if _state.get("initialized"):
        return

    reset_payload = {
        "task_id": GAMES_TO_TASK_ID_RANGE[_SELECTED_GAME][0],
        "seed": MINER_SEED,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    rank, env_pool, num_servers, thread_pool, generation_semaphore = init_env_pool(reset_payload)

    curriculum = _curriculum_factory(trainer.args)
    print(
        f"[CURRICULUM] Othello initialized: initial_max_turn={trainer.args.initial_max_turn}, "
        f"final_max_turn={_FINAL_MAX_TURN}, rollouts_per_stage={trainer.args.rollouts_per_stage}"
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
# Core episode runner (shared between full-prompt and last-prompt variants)
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
) -> tuple[int, "dict | None"]:
    """Run one Othello episode.

    With ``use_full_prompt=True`` the per-turn token IDs are accumulated with an
    action mask (1 for the policy's own tokens, 0 for env-injected observation
    tokens); with ``use_full_prompt=False`` only the final turn is returned.
    """
    game_id = int(prompt)

    server_idx   = (index + rank) % num_servers
    env_endpoint = env_pool[server_idx]["base_url"]

    # Full-prompt accumulation state
    episode_prompt_ids:     list[int]   = []
    episode_completion_ids: list[int]   = []
    episode_logprobs:       list[float] = []
    episode_action_mask:    list[int]   = []
    prev_full_ids: "list[int] | None"   = None

    # Last-prompt fallback (overwritten each iteration when use_full_prompt=False)
    prompt_ids:     list[int]   = []
    completion_ids: list[int]   = []
    logprobs:       list[float] = []

    # Episode state
    invalid_count = 0
    done          = False
    final_reward  = 0.0
    turn_number   = 0
    rewards: list[float] = []
    calculator = RewardCalculator()

    # --- Reset environment ---
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        reset_res.raise_for_status()
        result_block          = reset_res.json()["result"]
        episode_id            = result_block.get("episode_id", "")
        formatted_observation = result_block.get("observation", "")
    except Exception as exc:
        print(f"Failed to reset othello environment (Game {game_id}): {exc}")
        return index, None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": formatted_observation},
    ]

    # --- Interaction loop ---
    while not done and turn_number < current_max_turn:
        with generation_semaphore:
            rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

        prompt_ids      = rollout_outputs.get("prompt_ids", [])
        completion_ids  = rollout_outputs.get("completion_ids", [])
        logprobs        = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # --- Token accumulation (full-prompt mode) ---
        if use_full_prompt:
            if len(prompt_ids) > _MAX_PROMPT_LEN:
                print(
                    f"Warning: othello prompt exceeded {_MAX_PROMPT_LEN} tokens "
                    f"({len(prompt_ids)}) at turn {turn_number}, ending episode early"
                )
                done = True
                break

            if turn_number == 0:
                episode_prompt_ids = prompt_ids
                prev_full_ids      = prompt_ids.copy()
            else:
                if prev_full_ids is None:
                    prev_full_ids = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                    # BPE re-tokenisation can shift earlier IDs; skip delta mask this turn.
                    print(
                        f"Warning: othello token shift at turn {turn_number} "
                        f"(expected prefix {len(prev_full_ids)}, got {len(prompt_ids)})."
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

        # --- Parse action (tool call preferred, legacy text as fallback) ---
        action_to_send = _pvp_extract_action_id(completion_text)

        # --- Step environment ---
        try:
            formatted_observation = ""
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action_to_send, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block            = step_res.json()["result"]
            formatted_observation = step_block.get("observation", "")
            step_reward           = step_block.get("reward", 0)
            done                  = step_block.get("done", False)
        except Exception as exc:
            print(f"Othello step failed: {exc}")
            step_reward = -0.01
            done = False
            invalid_count += 1

        if "Nothing happens" in formatted_observation or "Invalid" in formatted_observation:
            invalid_count += 1

        if done:
            final_reward = step_reward
            # Amplify the terminal win/loss; intermediate steps contribute 0.
            rewards.append(step_reward * calculator.terminal_weight)
        else:
            rewards.append(0.0)
            messages.append({"role": "user", "content": formatted_observation})

        turn_number += 1

    # --- Final reward ---
    train_reward = calculator.calculate_discounted_return(rewards)

    print(
        "[OTHELLO ID:{:<6} Done:{} T:{:>2d} | Reward:{:>7.2f} | EnvR:{:>5.1f} | Inv:{:<2}]".format(
            str(game_id)[:6], int(done), turn_number, train_reward, final_reward, invalid_count,
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

    curriculum       = _state["curriculum"]
    current_max_turn = curriculum.get_max_turn()
    print(f"[CURRICULUM] Othello rollout {curriculum.total_rollouts}: max_turn={current_max_turn}")

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
    print(f"[BATCH] Othello finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.2f}")

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
    max_turns: int = _FINAL_MAX_TURN,
) -> dict[str, list]:
    """Parallelised rollout — accumulates all turns with action masking."""
    return _dispatch(prompts, trainer, use_full_prompt=True)


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = _FINAL_MAX_TURN,
) -> dict[str, list]:
    """Parallelised rollout — returns only the last turn's token IDs."""
    return _dispatch(prompts, trainer, use_full_prompt=False)
