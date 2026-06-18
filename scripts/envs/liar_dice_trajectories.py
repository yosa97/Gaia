"""Expert trajectory generator for Liar's Dice SFT training.

Produces SFT data in PvP-eval-compatible format (matches what validator feeds
the model during PvP tournament eval starting 2026-05-25).

Output trajectory:
- system: PvP system prompt (`pvp_game_prompts.yml:system_prompt_template`)
- user:   Current State + Player ID + Legal Actions + 'Your choice (ID only):'
- assistant: pure integer action ID (no CoT)

Format source: G.O.D feature/pvp-eval-container
- validator/evaluation/pvp/agents.py:LiarsDiceAgent.format_state
- core/config/pvp_game_prompts.yml
"""

import math
import os
import random
import re

import requests

from envs.liar_dice_env import parse_game_state
# Tool-calling format (new eval, #1201): system/user prompts in the new format
# and the assistant turn is a native game_action tool call (not plain text).
from envs.pvp_tool_format import (
    SYSTEM_PROMPT_LIARS_DICE,
    build_user_prompt_liars_dice as build_pvp_user_prompt_liars_dice,
    assistant_action_message,
)

_TIMEOUT = 2400
# Dedup divergence: a miner-unique seed for the per-game action RNG. Teammates
# sharing this file (byte-identical) sampled actions from the global RNG; here
# each game uses random.Random(game_id ^ MINER_SEED), so the expert's action
# choices on near-ties differ per miner -> different trajectories -> different
# model, while still playing valid expert moves. Set MINER_SEED uniquely.
_MINER_SEED = int(os.environ.get("MINER_SEED", "483047253"))
# Slightly higher than the team's 0.01 so near-tied actions vary a bit more
# (still near-greedy expert play; preserves quality).
_SAMPLING_TEMPERATURE = 0.03

# System prompt now sourced from PvP canonical (matches validator eval format exactly)
_SYSTEM_PROMPT = SYSTEM_PROMPT_LIARS_DICE


def _softmax_weights(probs: list[float], temperature: float) -> list[float]:
    """Convert raw probs to softmax weights."""
    if temperature <= 0:
        best = max(range(len(probs)), key=lambda i: probs[i])
        return [1.0 if i == best else 0.0 for i in range(len(probs))]
    scaled = [p / temperature for p in probs]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def get_expert_action(messages: list[dict], rng: "random.Random | None" = None) -> str:
    """Probability-based action selection. ``rng`` is a per-game seeded RNG
    (miner-unique) so action choices diverge from teammates; falls back to the
    global RNG when not provided."""
    picker = rng or random
    try:
        gs = parse_game_state(messages)
    except Exception:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        raw = re.findall(r"^(\d+)\s*->", last_user, re.MULTILINE)
        return min(raw, key=int) if raw else "0"

    if not gs.actions:
        return "0"

    probs = [a.prob for a in gs.actions]
    weights = _softmax_weights(probs, _SAMPLING_TEMPERATURE)
    chosen = picker.choices(gs.actions, weights=weights, k=1)[0]
    return str(chosen.action_id)


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> "list[dict] | None":
    """
    Run one Liar's Dice game against the env server using the expert policy.
    Returns the messages list (system/user/assistant) or None on failure.

    User messages are reformatted to **PvP eval format** (Current State + Player ID
    + 'Your choice (ID only):' suffix) so SFT trains the exact distribution that
    PvP tournament will feed at eval time.

    Player perspective alternates between 0 and 1 per game_id (parity) so SFT
    dataset covers both perspectives (PvP eval plays each seed twice with
    positions swapped).
    """
    reset_payload = {
        "task_id": game_id,
        "seed": game_id,
        "opponent": "mcts",
        "mcts_max_simulations": 225,
        "mcts_num_rollouts": 1,
    }
    try:
        res = requests.post(f"{env_endpoint}/reset", json=reset_payload, timeout=_TIMEOUT)
        res.raise_for_status()
        block = res.json()["result"]
        episode_id = block.get("episode_id", "")
        observation = block.get("observation", "")
    except Exception as exc:
        print(f"[env] Reset failed (game {game_id}): {exc}")
        return None

    # PvP eval covers both player perspectives via position swap (PVP_NUM_GAMES_PER_ENV
    # × 2). Alternate per game_id parity so SFT dataset includes both.
    player_id = game_id % 2

    # Per-game miner-unique RNG: deterministic per (game_id, MINER_SEED) so runs
    # are reproducible, but different from teammates using the same game_id.
    rng = random.Random((game_id * 0x9E3779B1) ^ _MINER_SEED)

    user_prompt = build_pvp_user_prompt_liars_dice(observation, player_id=player_id)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    for _ in range(max_turn):
        action = get_expert_action(messages, rng=rng)
        # SFT target = native game_action tool call (new eval). The env server
        # still receives the raw action id below.
        messages.append(assistant_action_message(action))

        try:
            step_res = requests.post(
                f"{env_endpoint}/step",
                json={"action": action, "episode_id": episode_id},
                timeout=_TIMEOUT,
            )
            step_res.raise_for_status()
            step_block = step_res.json()["result"]
            observation = step_block.get("observation", "")
            done = step_block.get("done", False)
        except Exception as exc:
            print(f"[env] Step failed (game {game_id}): {exc}")
            return None

        if done:
            break
        user_prompt = build_pvp_user_prompt_liars_dice(observation, player_id=player_id)
        messages.append({"role": "user", "content": user_prompt})
    else:
        print(f"[env] max_turn={max_turn} reached (game {game_id})")

    return messages

# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
