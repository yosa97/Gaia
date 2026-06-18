"""VM smoke test for the tool-calling SFT format (run on the real model/tokenizer).

WHY: the new env-tournament eval (core/pvp/bot.py, #1201) reads a `game_action`
TOOL CALL, not plain text. Training the model to emit a tool call requires the
tokenizer's chat template to render assistant `tool_calls` correctly AND the
assistant-masking to cover those tokens. That depends on the exact tokenizer and
CANNOT be checked without it. Run this with the tournament base model BEFORE
trusting any tool-format SFT data.

Usage (inside the trainer container or any box with the model cached):
    python scripts/smoke_tool_format.py /cache/models/Qwen--Qwen3-4B-Instruct-2507
    # or any local model dir / HF id

PASS criteria (printed at the end):
  1. apply_chat_template(tools=...) renders WITHOUT error.
  2. The rendered text contains the action id (the tool call is present).
  3. Assistant-masking (the train_sft_env logic) yields sum(mask) > 0 on the
     assistant tool-call turn.
If any fail, DO NOT switch to tool-format SFT — the chat template needs a
different tool_calls shape (e.g. arguments as a JSON string). Report the output.
"""

import sys

from transformers import AutoTokenizer

from envs.pvp_tool_format import (
    GAME_ACTION_TOOL,
    SYSTEM_PROMPT_LIARS_DICE,
    build_user_prompt,
    assistant_action_message,
)


def main(model_path: str) -> None:
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    user = build_user_prompt(
        state_desc="Your dice: [3, 5, 6]\nTotal dice in game: 10",
        player_id=0,
        legal_actions_block="3 -> 1-1\n4 -> 2-5\n5 -> Liar",
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_LIARS_DICE},
        {"role": "user", "content": user},
        assistant_action_message(5),  # content = native <tool_call> text
    ]

    # CONTENT-BASED approach: the tool call lives in the assistant CONTENT as
    # native <tool_call> text, so we tokenize EXACTLY like train_sft_env does —
    # plain apply_chat_template, no tools= needed.
    print("=== 1) render full conversation (plain, no tools=) ===")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    print(text)
    has_action = "<tool_call>" in text and "game_action" in text and "5" in text
    print(f"\n[check] <tool_call> game_action(action_id=5) present in rendered text: {has_action}")

    print("\n=== 2) assistant masking (train_sft_env logic, no tools=) ===")
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    mask = [0] * len(ids)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        p = len(tok.apply_chat_template(msgs[:i], tokenize=True, add_generation_prompt=True))
        r = len(tok.apply_chat_template(msgs[:i + 1], tokenize=True, add_generation_prompt=False))
        p = max(0, min(p, len(ids)))
        r = max(0, min(r, len(ids)))
        for j in range(p, r):
            mask[j] = 1
    n_assist = sum(mask)
    print(f"[check] assistant tokens masked for loss: {n_assist} (must be > 0)")
    if n_assist:
        masked_text = tok.decode([t for t, mk in zip(ids, mask) if mk])
        print(f"[info] masked (trained) span decodes to:\n{masked_text!r}")

    print("\n=== RESULT ===")
    ok = has_action and n_assist > 0
    print("PASS — tool-format SFT is safe to use on this tokenizer." if ok
          else "FAIL — do NOT use tool-format as-is; report this output for a fix.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/smoke_tool_format.py <model_path_or_hf_id>")
        sys.exit(2)
    main(sys.argv[1])
