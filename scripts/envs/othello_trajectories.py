"""Expert trajectory generator for Othello SFT training (tool-calling format).

Othello action_ids index the 8x8 board (cell = row*8 + col, 0-63), with a pass
action when no placement is legal. A strong, dependency-free expert is classic
*positional play* (Iago/Rosenbloom weights): grab corners, take stable edges,
and avoid the X-/C-squares next to empty corners that hand the opponent a
corner. We pick the legal action with the highest positional weight. This yields
legal, sensible moves so the SFT policy learns to commit a `game_action` tool
call (not forfeit) and plays a reasonable opening/midgame.

Output trajectory matches the other env generators: system / user / assistant
(native game_action tool call) per turn. Registered in
``envs/sft_env_configs.py`` under the env name "othello".
"""

import os
import random
import re

import requests

from envs.othello_env import _SYSTEM_PROMPT
from envs.pvp_tool_format import assistant_action_message
from envs.pvp_memory_tools import (
    default_memories,
    eval_system_prompt,
    build_assistant_turn,
    LONGTERM_NOTE,
)

_TIMEOUT = 2400
_MINER_SEED = int(os.environ.get("MINER_SEED", "483047253"))
_MEMORY_TRAINING = os.environ.get("PVP_MEMORY_TRAINING", "0") == "1"

# Standard Othello positional weights (Rosenbloom/Iago), indexed by cell = r*8+c.
# Corners are decisive; X-squares (diagonal-adjacent to a corner) and C-squares
# (edge-adjacent to a corner) are dangerous because they concede the corner.
_OTHELLO_WEIGHTS = [
    120, -20,  20,   5,   5,  20, -20, 120,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
     20,  -5,  15,   3,   3,  15,  -5,  20,
      5,  -5,   3,   3,   3,   3,  -5,   5,
      5,  -5,   3,   3,   3,   3,  -5,   5,
     20,  -5,  15,   3,   3,  15,  -5,  20,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
    120, -20,  20,   5,   5,  20, -20, 120,
]

_ACTION_LINE_RE = re.compile(r"^\s*(\d+)\s*->", re.MULTILINE)


def _legal_action_ids(observation: str) -> list:
    """Parse the 'id -> move' legal-action lines from the env observation."""
    return [int(m) for m in _ACTION_LINE_RE.findall(observation or "")]


def _expert_action_id(observation: str) -> "int | None":
    """Pick the legal action with the best positional weight (pass only if forced)."""
    ids = _legal_action_ids(observation)
    if not ids:
        return None
    # Board placements (0-63) ranked by positional weight; pass / out-of-board
    # actions get a sentinel low score so they are chosen only when alone.
    return max(ids, key=lambda a: _OTHELLO_WEIGHTS[a] if 0 <= a < 64 else -1000)


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 64,
) -> "list[dict] | None":
    """Play one Othello game vs the env server with the positional expert."""
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
        print(f"[env] Reset failed (othello game {game_id}): {exc}")
        return None

    memories = default_memories() if _MEMORY_TRAINING else None
    system_content = (
        eval_system_prompt("othello", memories) if _MEMORY_TRAINING else _SYSTEM_PROMPT
    )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": observation},
    ]

    for turn_idx in range(max_turn):
        aid = _expert_action_id(observation)
        if aid is None:
            break
        action = str(aid)

        if _MEMORY_TRAINING:
            mem_ops = (
                [("long_term_append", 1, LONGTERM_NOTE["othello"])] if turn_idx == 0 else None
            )
            messages.append(build_assistant_turn(action, memory_ops=mem_ops))
        else:
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
            print(f"[env] Step failed (othello game {game_id}): {exc}")
            return None

        if done:
            break
        messages.append({"role": "user", "content": observation})

    return messages
