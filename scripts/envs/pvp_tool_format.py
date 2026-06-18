"""Tool-calling PvP format for SFT — matches the NEW env-tournament eval
(core/pvp/bot.py, released 2026-06-13 #1201).

The eval no longer reads a plain-text action. Each turn it:
  * builds system  = "You are playing {game}.\n# Game Rules {rules}" + memory
                      notes + tool guidance,
  * builds user    = "Current state:\n{state}\n\nYou are Player {pid}.\n"
                      "Legal actions:\n{id} -> {move}\n...",
  * passes the `game_action` tool (+ memory tools) with tool_choice="auto",
  * and reads the model's TOOL CALL: game_action(action_id=N). No tool call
    (e.g. plain text) => InvalidActionForfeitError => the model LOSES the turn.

So SFT must teach the model to emit a `game_action` tool call, not text. This
module builds the new system/user prompts and the assistant tool-call message.

State/legal-action reconstruction reuses pvp_format's per-game reformatters
(the agents' format_state output is unchanged); only the wrapper, the system
template and the OUTPUT (tool call) differ.

DESIGN — content-based tool call (intentional, low-risk):
The assistant turn carries the tool call as the model's NATIVE serialization in
plain CONTENT — `<tool_call>\n{"name":"game_action","arguments":{"action_id":N}}\n</tool_call>` —
NOT a structured `tool_calls` field. Reasons:
  * The tournament base model is Qwen3-4B-Instruct-2507; SGLang serves it with
    the 'qwen25' tool-call parser, which extracts exactly this <tool_call> JSON
    block from generated text — so producing that text trains the model to do
    precisely what the parser reads back.
  * Because it is plain content, the existing tokenizer/assistant-masking and
    the whole text pipeline (_clean / dedup / windowing / save) work UNCHANGED.
    No reliance on how a given tokenizer renders a structured tool_calls field
    (which could not be checked without the real tokenizer).
GAME_ACTION_TOOL below mirrors the validator's tool schema for reference / the
optional smoke test; the SFT pipeline itself does not need to pass it anywhere.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from envs.pvp_format import (
    reformat_liars_dice_observation,
    reformat_leduc_poker_observation,
    reformat_gin_rummy_observation,
)

_ASSETS_DIR = Path(__file__).resolve().parent / "pvp_assets"
_PROMPTS_PATH = _ASSETS_DIR / "pvp_game_prompts.yml"

# Verbatim from core/pvp/bot.py:_TOOL_GUIDANCE (the eval appends this to the
# system prompt). Keeping it identical makes SFT match the eval's instruction.
_TOOL_GUIDANCE = (
    "You get ONE response this turn. In it, optionally edit your memory notes, and "
    "then call game_action with a legal action id to commit your move. If you do not "
    "call game_action, you forfeit the turn — so always include it."
)

GAME_ACTION_TOOL_NAME = "game_action"

# Tool schema, shape-matched to core/pvp/tools.build_game_action_tool ->
# FunctionSchema.to_openai(). Reference only (mirrors what the validator passes
# at eval); the content-based SFT pipeline does NOT pass this to the tokenizer.
GAME_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": GAME_ACTION_TOOL_NAME,
        "description": "Commit your move and end your turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "integer",
                    "description": "The id of the legal action to play.",
                }
            },
            "required": ["action_id"],
        },
    },
}


def _load_prompts() -> dict:
    with open(_PROMPTS_PATH) as handle:
        return yaml.safe_load(handle)


def build_system_prompt(game_name: str) -> str:
    """NEW system prompt: "You are playing {game}.\n# Game Rules {rules}" + tool
    guidance. (The eval also injects a dynamic memory-notes block; we omit it for
    SFT since it is empty/stateful — the model still learns to call game_action.)
    """
    prompts = _load_prompts()
    rules_key = f"{game_name}_rules"
    if rules_key not in prompts:
        raise ValueError(f"Unknown game: {game_name} (no {rules_key})")
    base = prompts["system_prompt_template"].format(game_name=game_name, rules=prompts[rules_key])
    return f"{base}\n\n{_TOOL_GUIDANCE}"


def build_user_prompt(state_desc: str, player_id: int, legal_actions_block: str) -> str:
    """NEW user prompt — matches core/pvp/bot.LLMBot._user_prompt EXACTLY
    (lowercase 'Current state' / 'Legal actions'; NO 'Your choice' suffix)."""
    return (
        f"Current state:\n{state_desc}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal actions:\n{legal_actions_block}"
    )


import json as _json


def assistant_action_message(action_id) -> dict:
    """Assistant turn that COMMITS the move via a game_action tool call.

    We put the tool call in the assistant CONTENT as the model's NATIVE
    serialization (Qwen / Hermes `<tool_call>{...}</tool_call>`), NOT in a
    `tool_calls` field. Why:
      * The tournament base model is Qwen3-4B-Instruct-2507; SGLang serves it
        with the 'qwen25' tool-call parser, which extracts exactly this
        <tool_call> JSON block from the generated text.
      * Keeping it as plain content means the existing assistant-masking and the
        whole text pipeline (_clean / dedup / windowing / save) work unchanged —
        no dependency on how a tokenizer renders a structured `tool_calls` field
        (which cannot be verified without the real tokenizer).
    At eval, the model emits this same block, SGLang parses action_id -> a legal
    move is committed instead of a forfeit.
    """
    try:
        aid = int(action_id)
    except (TypeError, ValueError):
        aid = action_id
    payload = _json.dumps({"name": GAME_ACTION_TOOL_NAME, "arguments": {"action_id": aid}})
    return {
        "role": "assistant",
        "content": f"<tool_call>\n{payload}\n</tool_call>",
    }


def extract_action_id(assistant_content: str):
    """Pull the action_id back out of a native tool-call assistant content
    (used by data-balancing). Returns str(action_id) or '' if not parseable."""
    import re as _re
    m = _re.search(r'"action_id"\s*:\s*(-?\d+)', assistant_content or "")
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Per-game user-prompt builders (reuse pvp_format reformatters, new assembly)
# ---------------------------------------------------------------------------

def build_user_prompt_liars_dice(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_liars_dice_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


def build_user_prompt_leduc_poker(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_leduc_poker_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


def build_user_prompt_gin_rummy(env_obs: str, player_id: int = 0) -> str:
    state_desc, actions = reformat_gin_rummy_observation(env_obs, player_id)
    return build_user_prompt(state_desc, player_id, actions)


# System prompts (cached at import)
SYSTEM_PROMPT_LIARS_DICE = build_system_prompt("liars_dice") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_LEDUC_POKER = build_system_prompt("leduc_poker") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_GIN_RUMMY = build_system_prompt("gin_rummy") if _PROMPTS_PATH.exists() else None
SYSTEM_PROMPT_OTHELLO = build_system_prompt("othello") if _PROMPTS_PATH.exists() else None
