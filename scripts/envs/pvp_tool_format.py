"""Tool-calling format helpers for PvP environment training.

The SN56 PvP evaluation bot (``core/pvp/bot.py``) serves the policy through
SGLang with the ``qwen25`` tool-call parser. At inference time the bot exposes a
single ``game_action`` tool and reads the model's reply as a *native* Qwen tool
call of the form::

    <tool_call>
    {"name": "game_action", "arguments": {"action_id": 3}}
    </tool_call>

If the model instead emits a bare action ID (the legacy "respond with ONLY the
action ID" behaviour) the bot raises ``InvalidActionForfeitError`` and the game
is forfeited -> score 0. To avoid that, the GRPO rollouts must train the policy
to speak the *same* tool-call dialect that the evaluator expects.

This module centralises that dialect so every environment shares one source of
truth:

* :data:`GAME_ACTION_TOOL` - the JSON-schema tool definition handed to
  ``apply_chat_template(tools=...)`` so the chat template primes Qwen's
  tool-calling mode exactly as the evaluator does.
* :data:`TOOL_GUIDANCE` - a system-prompt block, with a literal example, that
  instructs the model to answer via the tool call. This is kept independent of
  the template's auto-injected tool block so the instruction reaches the model
  even on tokenizers whose chat template does not render the tool schema.
* :func:`extract_action_id` - a tolerant parser that recovers the chosen
  ``action_id`` from a tool call (tagged JSON, bare JSON, or ``game_action(...)``
  call syntax) and falls back to the legacy plain-text forms so training never
  collapses to all-forfeit while the policy is still learning the format.
* :func:`assistant_action_message` - builds the assistant turn in tool-call
  form when an environment needs to replay an expert/teacher action into the
  dialogue history.
"""

from __future__ import annotations

import json
import re
from typing import Optional

# A fixed, miner-specific seed. Kept here so every environment that wants a
# reproducible-but-distinct RNG stream can import the same value. The concrete
# number is arbitrary; what matters is that it is *ours* and not the upstream
# default of 42, so trajectory sampling diverges from any repo sharing the
# baseline generators.
MINER_SEED = 716293041


# ---------------------------------------------------------------------------
# Tool schema handed to apply_chat_template(tools=[GAME_ACTION_TOOL])
# ---------------------------------------------------------------------------
GAME_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "game_action",
        "description": (
            "Submit the action you choose for the current game state. Call this "
            "exactly once per turn, passing the integer action_id of one of the "
            "legal actions listed in the observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": {
                    "type": "integer",
                    "description": (
                        "The integer ID of the legal action you choose to play "
                        "this turn."
                    ),
                }
            },
            "required": ["action_id"],
        },
    },
}

# Convenience: the list form most callers want to assign to ``trainer.tools``.
GAME_ACTION_TOOLS = [GAME_ACTION_TOOL]


# ---------------------------------------------------------------------------
# System-prompt guidance block (replaces the legacy "ONLY the action ID" text)
# ---------------------------------------------------------------------------
TOOL_GUIDANCE = (
    "# Output Format\n"
    "You MUST respond by calling the `game_action` tool with the action_id of a "
    "legal action. Emit the call as a native tool call in exactly this form:\n"
    "<tool_call>\n"
    '{"name": "game_action", "arguments": {"action_id": N}}\n'
    "</tool_call>\n"
    "Replace N with the integer ID of your chosen legal action. Do NOT print the "
    "action ID as bare text, and do NOT add any prose, explanation, or card "
    "names outside the tool call.\n"
    "Example - to choose the legal action whose ID is 2, respond with exactly:\n"
    "<tool_call>\n"
    '{"name": "game_action", "arguments": {"action_id": 2}}\n'
    "</tool_call>"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ACTION_ID_JSON_RE = re.compile(r'"action_id"\s*:\s*(-?\d+)')
_GAME_ACTION_CALL_RE = re.compile(r"game_action\s*\(\s*action_id\s*=\s*(-?\d+)\s*\)")
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def _action_id_from_tool_call(text: str) -> Optional[str]:
    """Return the action_id encoded as a tool call, or ``None`` if absent.

    Handles, in order of preference:
      1. ``<tool_call>{...}</tool_call>`` wrapped JSON (the canonical Qwen form).
      2. Bare ``{"name": "game_action", "arguments": {"action_id": N}}`` JSON.
      3. ``game_action(action_id=N)`` Python-call syntax.
    """
    # 1) Canonical tagged tool call. Parse the JSON body when well-formed; fall
    #    back to a regex on the body when the model emitted slightly malformed
    #    JSON (trailing commas, single quotes) so we still recover the id.
    for body in _TOOL_CALL_BLOCK_RE.findall(text):
        try:
            obj = json.loads(body)
            args = obj.get("arguments", obj)
            if isinstance(args, str):
                args = json.loads(args)
            if "action_id" in args:
                return str(int(args["action_id"]))
        except (ValueError, TypeError):
            pass
        m = _ACTION_ID_JSON_RE.search(body)
        if m:
            return m.group(1)

    # 2) Bare JSON object naming the tool, without the <tool_call> wrapper.
    if "game_action" in text:
        m = _ACTION_ID_JSON_RE.search(text)
        if m:
            return m.group(1)
        m = _GAME_ACTION_CALL_RE.search(text)
        if m:
            return m.group(1)

    return None


def extract_action_id(completion_text: str) -> str:
    """Recover the chosen action ID from a model completion.

    Preference order: a valid ``game_action`` tool call first (matching the
    evaluator), then the legacy ``Action: N`` marker, then the trailing integer.
    Always returns a string (possibly the cleaned text) so callers behave like
    the original plain-text parsers.
    """
    if completion_text is None:
        return ""

    tool_id = _action_id_from_tool_call(completion_text)
    if tool_id is not None:
        return tool_id

    # --- legacy fallbacks (kept so training is robust while the policy is still
    # learning the tool-call format) ---
    cleaned = _THOUGHT_RE.sub("", completion_text).strip()
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-4].strip()
    if "Action:" in cleaned:
        tail = cleaned.split("Action:")[-1].strip()
        m = re.match(r"\s*(-?\d+)", tail)
        if m:
            return m.group(1)
    matches = re.findall(r"-?\d+", cleaned)
    return matches[-1] if matches else cleaned.strip()


def is_valid_tool_call(completion_text: str) -> bool:
    """True iff the completion contains a parseable ``game_action`` tool call.

    Useful for format-rate metrics / reward shaping: the evaluator only accepts
    completions for which this is True.
    """
    return _action_id_from_tool_call(completion_text or "") is not None


def assistant_action_message(action_id) -> dict:
    """Build an assistant turn that submits ``action_id`` as a tool call.

    Used by environments that replay an expert/teacher move into the message
    history so the recorded assistant turns match the inference-time dialect.
    """
    payload = {"name": "game_action", "arguments": {"action_id": int(action_id)}}
    return {
        "role": "assistant",
        "content": "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>",
    }
