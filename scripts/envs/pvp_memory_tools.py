"""Memory-tool schemas + slot memory mirroring the live PvP eval harness.

The evaluator (gradients-ai/G.O.D `core/pvp/bot.py` + `tools.py` + `memory.py`)
gives the policy, in addition to ``game_action``, a set of memory tools:

    working_rewrite / working_append      -> notes for THIS game (reset each game)
    long_term_rewrite / long_term_append  -> notes on THIS opponent (persist
                                             across games vs the same opponent)

A turn is a single model response that may edit memory and then MUST call
``game_action``; after each game a reflection step lets the model consolidate
long-term notes. Long-term memory persisting across a match series is the new
axis almost no miner trains for — a policy that externalises opponent reads into
persistent slots and recalls them can systematically out-adapt opponents.

This module mirrors the eval's tool surface, slot sizes, and rendering so the
policy is trained on EXACTLY what it meets at eval. It is dependency-free and
unit-testable (no trl/torch import), and additive: nothing here changes the
existing game_action path.

Cross-checked against the eval on 2026-06-22 (core/pvp/{tools,memory,constants}.py).
"""

from __future__ import annotations

import json
import re
from typing import Literal, Optional

from envs.pvp_tool_format import GAME_ACTION_TOOL


# ---------------------------------------------------------------------------
# Constants — mirror core/pvp/constants.py
# ---------------------------------------------------------------------------
WORKING_MEM_SLOTS    = 4
WORKING_SLOT_TOKENS  = 128
LONGTERM_MEM_SLOTS   = 8
LONGTERM_SLOT_TOKENS = 128

# Area / op identifiers — must match memory_tool_name(area, op) = f"{area}_{op}"
_AREAS = {
    "working":   ("notes for THIS game, reset each game",        WORKING_MEM_SLOTS),
    "long_term": ("notes on THIS opponent, persist across games", LONGTERM_MEM_SLOTS),
}
_OPS = {
    "rewrite": ("Overwrite", "replaces the slot's previous content"),
    "append":  ("Append to", "oldest text drops if the slot is full"),
}


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible function form, same shape as GAME_ACTION_TOOL)
# ---------------------------------------------------------------------------
def _memory_tool(area: str, op: str) -> dict:
    purpose, n_slots = _AREAS[area]
    verb, effect = _OPS[op]
    return {
        "type": "function",
        "function": {
            "name": f"{area}_{op}",
            "description": (
                f"{verb} a {area} memory slot (slots 1-{n_slots}; {purpose}); {effect}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot": {
                        "type": "integer",
                        "description": f"Slot number to edit (1-{n_slots}).",
                        "minimum": 1,
                        "maximum": n_slots,
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to store in the slot.",
                    },
                },
                "required": ["slot", "content"],
            },
        },
    }


MEMORY_TOOLS = [
    _memory_tool("working", "rewrite"),
    _memory_tool("working", "append"),
    _memory_tool("long_term", "rewrite"),
    _memory_tool("long_term", "append"),
]

# Full tool surface the eval exposes per turn: memory tools + the move tool.
# Assign this to ``trainer.tools`` so the chat template renders the same tools
# the evaluator provides.
MEMORY_AND_GAME_TOOLS = MEMORY_TOOLS + [GAME_ACTION_TOOL]

_MEMORY_TOOL_NAMES = {t["function"]["name"] for t in MEMORY_TOOLS}


# ---------------------------------------------------------------------------
# Eval-aligned prompt builders (verbatim format from core/pvp/bot.py)
# ---------------------------------------------------------------------------
# The evaluator's per-turn system prompt is:
#   agent.generate_system_prompt()  +  memory_block  +  tool guidance
# and the user prompt uses lowercase "Current state:" / "Legal actions:".
# Training on this exact surface maximises transfer to eval.
EVAL_TOOL_GUIDANCE = (
    "You get ONE response this turn. In it, optionally edit your memory notes, and "
    "then call game_action with a legal action id to commit your move. If you do not "
    "call game_action, you forfeit the turn — so always include it."
)


def build_eval_system_prompt(game_rules_prompt: str, memories: "dict[str, SlotMemory]") -> str:
    """System prompt exactly as the eval assembles it: rules + memory + guidance."""
    return "\n\n".join([game_rules_prompt.strip(), render_memory_block(memories), EVAL_TOOL_GUIDANCE])


def build_eval_user_prompt(state_desc: str, player_id: int, legal_action_lines: str) -> str:
    """User prompt verbatim from LLMBot._user_prompt (lowercase headers)."""
    return (
        f"Current state:\n{state_desc.strip()}\n\n"
        f"You are Player {player_id}.\n"
        f"Legal actions:\n{legal_action_lines.strip()}"
    )


def eval_system_prompt(game_name: str, memories: "dict[str, SlotMemory]") -> str:
    """Convenience: rules (from the eval YAML) + memory block + tool guidance.

    Used by the SFT trajectory generators so the training system prompt matches
    what LLMBot assembles at eval. Falls back to a minimal rules line if the
    prompt YAML is unavailable, so it never raises at import/runtime.
    """
    try:
        from envs.pvp_format import build_system_prompt as _rules_prompt
        rules = _rules_prompt(game_name)
    except Exception:
        rules = f"You are playing {game_name}."
    return build_eval_system_prompt(rules, memories)


# Grounded, generic-but-valid long-term opponent reads per game. Injected once
# per episode in memory-augmented SFT so the policy learns the WRITE format of
# long_term notes (the content is sound strategy, not filler). At eval the model
# generalises this to writing real per-opponent reads.
LONGTERM_NOTE = {
    "liars_dice": (
        "Opponent read: weigh each bid against the implied dice count; call Liar "
        "when the claimed quantity is statistically improbable, and bluff-raise "
        "only when behind."
    ),
    "leduc_poker": (
        "Opponent read: track how often they raise vs call per round; fold weak "
        "holdings into sustained aggression, value-raise pairs and high cards."
    ),
    "gin_rummy": (
        "Opponent read: watch their discards and pickups to infer melds; hold "
        "cards that block their runs and knock early when your deadwood is low."
    ),
    "othello": (
        "Opponent read: note whether they grab edges/corners early; prioritise "
        "corners, avoid squares adjacent to empty corners, and value mobility."
    ),
}


# ---------------------------------------------------------------------------
# SlotMemory — faithful (simplified) port of core/pvp/memory.py
# ---------------------------------------------------------------------------
def _count(text: str) -> int:
    return len(text.split())


def _truncate(text: str, max_tokens: int, keep: Literal["head", "tail"]) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    kept = words[:max_tokens] if keep == "head" else words[len(words) - max_tokens:]
    return " ".join(kept)


class SlotMemory:
    """A fixed set of fixed-size, independently addressable memory slots.

    Total operations: a bad slot or malformed input returns an error string and
    NEVER raises (matches the eval — only game_action can forfeit). ``rewrite``
    keeps the head; ``append`` is FIFO (drops the oldest text).
    """

    def __init__(self, n_slots: int, slot_token_budget: int, separator: str = "\n"):
        self.n_slots = n_slots
        self.slot_token_budget = slot_token_budget
        self._sep = separator
        self.slots: dict[int, str] = {i: "" for i in range(1, n_slots + 1)}

    def _valid(self, slot: int) -> bool:
        return isinstance(slot, int) and not isinstance(slot, bool) and 1 <= slot <= self.n_slots

    def _fit(self, text: str, keep: Literal["head", "tail"]) -> str:
        if _count(text) <= self.slot_token_budget:
            return text
        return _truncate(text, self.slot_token_budget, keep)

    def rewrite(self, slot: int, content: str) -> str:
        if not self._valid(slot):
            return f"error: slot {slot} out of range (1-{self.n_slots})"
        self.slots[slot] = self._fit(content, keep="head")
        return f"ok: slot {slot} rewritten"

    def append(self, slot: int, content: str) -> str:
        if not self._valid(slot):
            return f"error: slot {slot} out of range (1-{self.n_slots})"
        existing = self.slots[slot]
        combined = f"{existing}{self._sep}{content}" if existing else content
        self.slots[slot] = self._fit(combined, keep="tail")
        return f"ok: slot {slot} appended"

    def render(self, title: Optional[str] = None) -> str:
        lines = [f"  [{i}] {self.slots[i] or '(empty)'}" for i in range(1, self.n_slots + 1)]
        body = "\n".join(lines)
        return f"{title}\n{body}" if title else body

    def reset(self) -> None:
        self.slots = {i: "" for i in range(1, self.n_slots + 1)}


def default_memories() -> dict[str, SlotMemory]:
    """Build the standard working + long-term areas (eval default sizes)."""
    return {
        "working":   SlotMemory(WORKING_MEM_SLOTS, WORKING_SLOT_TOKENS),
        "long_term": SlotMemory(LONGTERM_MEM_SLOTS, LONGTERM_SLOT_TOKENS),
    }


def render_memory_block(memories: dict[str, SlotMemory]) -> str:
    """Render the memory block exactly as the eval injects it into the prompt.

    Mirrors LLMBot._memory_block: each area titled ``"<AREA> (your notes):"``.
    """
    return "\n\n".join(
        mem.render(title=f"{area.upper()} (your notes):") for area, mem in memories.items()
    )


# ---------------------------------------------------------------------------
# Assistant-turn builder (memory edits + game_action, as native tool calls)
# ---------------------------------------------------------------------------
def _tool_call_block(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def build_assistant_turn(action_id, memory_ops: Optional[list] = None) -> dict:
    """Assistant message that (optionally) edits memory then commits the move.

    ``memory_ops`` is a list of ``(tool_name, slot, content)`` tuples emitted as
    tool calls BEFORE the game_action call — matching the eval, which applies
    every memory edit in the response and takes the first legal game_action.
    """
    blocks = []
    for op in memory_ops or []:
        name, slot, content = op
        if name in _MEMORY_TOOL_NAMES:
            blocks.append(_tool_call_block(name, {"slot": int(slot), "content": str(content)}))
    blocks.append(_tool_call_block("game_action", {"action_id": int(action_id)}))
    return {"role": "assistant", "content": "\n".join(blocks)}


# ---------------------------------------------------------------------------
# Parsing memory tool calls from a completion (for GRPO rollout application)
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_memory_ops(completion_text: str) -> list:
    """Return [(tool_name, slot, content), ...] for memory tool calls in the text.

    game_action calls are ignored here (handled by the game parser). Malformed
    blocks are skipped silently — never raises.
    """
    ops = []
    for body in _TOOL_CALL_RE.findall(completion_text or ""):
        try:
            obj = json.loads(body)
        except (ValueError, TypeError):
            continue
        name = obj.get("name")
        if name in _MEMORY_TOOL_NAMES:
            args = obj.get("arguments", {}) or {}
            if "slot" in args and "content" in args:
                ops.append((name, args["slot"], args["content"]))
    return ops


def apply_memory_ops(memories: dict[str, SlotMemory], ops: list) -> None:
    """Apply parsed memory ops to the given memory areas (best-effort, no raise)."""
    for name, slot, content in ops:
        try:
            area, op = name.rsplit("_", 1)
            mem = memories.get(area)
            if mem is None:
                continue
            try:
                slot_i = int(slot)
            except (ValueError, TypeError):
                continue
            getattr(mem, op, lambda *_: None)(slot_i, str(content))
        except Exception:
            continue
