"""Expert trajectory generator for Goofspiel SFT training (tool-calling format).

Goofspiel (Game of Pure Strategy): each turn a prize card is revealed and both
players simultaneously commit one bid card; the higher bid wins the prize. A
strong, simple expert is *proportional bidding* — commit the hand card whose
value is closest to the revealed prize value (spend high cards on high prizes,
save low cards for low prizes). This produces legal, sensible moves so the SFT
policy learns to play (and to emit the `game_action` tool call the evaluator
requires) instead of forfeiting.

Output trajectory matches the other env generators: system / user / assistant
(native game_action tool call), repeated per turn. Registered in
``envs/sft_env_configs.py`` under the tournament env name "goofspiel".
"""

import os
import random

import requests

from envs.goof_spiel_env import (
    extract_and_format_observation,
    extract_prize_card,
    get_hand_cards,
    _BASE_SYSTEM_PROMPT,
)
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


def _expert_action_id(prize_card: "int | None", hand: list) -> "int | None":
    """Proportional bidding: play the hand card closest to the prize value.

    action_id = card - 1 (the env lists each card ``c`` as ``c-1 -> Bid: c``).
    Ties break toward the higher card so a strong prize is not under-bid.
    """
    if not hand:
        return None
    target = prize_card if prize_card is not None else hand[len(hand) // 2]
    card = min(hand, key=lambda c: (abs(c - target), -c))
    return card - 1


def generate_expert_episode(
    game_id: int,
    env_endpoint: str,
    max_turn: int = 30,
) -> "list[dict] | None":
    """Play one Goofspiel game vs the env server with the proportional expert."""
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
        print(f"[env] Reset failed (goofspiel game {game_id}): {exc}")
        return None

    player_id = game_id % 2
    formatted = extract_and_format_observation(observation)

    memories = default_memories() if _MEMORY_TRAINING else None
    system_content = (
        eval_system_prompt("goofspiel", memories) if _MEMORY_TRAINING else _BASE_SYSTEM_PROMPT
    )

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": formatted},
    ]

    for turn_idx in range(max_turn):
        prize = extract_prize_card(formatted)
        if prize is None:
            prize = extract_prize_card(observation)
        hand = get_hand_cards(formatted, player_id) or get_hand_cards(observation, player_id)
        aid = _expert_action_id(prize, hand)
        if aid is None:
            break
        action = str(aid)

        if _MEMORY_TRAINING:
            mem_ops = (
                [("long_term_append", 1, LONGTERM_NOTE["goofspiel"])] if turn_idx == 0 else None
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
            print(f"[env] Step failed (goofspiel game {game_id}): {exc}")
            return None

        if done:
            break
        formatted = extract_and_format_observation(observation)
        messages.append({"role": "user", "content": formatted})

    return messages
