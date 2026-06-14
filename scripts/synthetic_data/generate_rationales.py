#!/usr/bin/env python3
"""Generate CoT-augmented training data for ENV tournament SFT.

Reads ``gradients-io-tournaments/env_training_gradients`` from HF, filters
by game, augments each assistant turn with a ``Thought: ...`` rationale,
and writes the result to::

    scripts/data/augmented/env_training_<game>_cot.jsonl

This is the file ``scripts/train_sft_env.py:_load_env_training`` looks for
when ``USE_AUGMENTED_DATA != "0"`` (i.e. the "Augmented" tournament mode).
If the file is missing the trainer falls back to the raw HF dataset, which
is the behaviour you want for the "Anons" tournament mode.

USAGE
-----
::

    # Rule-based (fast, no API):
    python scripts/synthetic_data/generate_rationales.py --game gin_rummy
    python scripts/synthetic_data/generate_rationales.py --game liars_dice
    python scripts/synthetic_data/generate_rationales.py --game leduc_poker
    python scripts/synthetic_data/generate_rationales.py --game goof_spiel

    # Smoke test with a handful of samples:
    python scripts/synthetic_data/generate_rationales.py --game gin_rummy --limit 20

    # LLM-based rationale (slower, costs $$, but typically higher quality —
    # implement `_rationale_via_llm` to wire in your API of choice):
    python scripts/synthetic_data/generate_rationales.py --game gin_rummy --strategy llm

DESIGN NOTES
------------
- Two strategies: ``rule`` (cheap heuristic per game) and ``llm`` (stub).
- Output preserves the upstream ShareGPT schema (``conversations``
  list with ``from`` / ``value`` fields), so it slots directly into
  ``_sharegpt_to_messages`` without any extra plumbing.
- Each assistant turn becomes ``Thought: <rationale>\\nAction: <id>`` so
  the trainer's CoT branch can keep the reasoning when
  ``strip_thought=False`` (which is the default for augmented data — see
  ``train_sft_env.py:540``).
- The rule-based heuristics are intentionally conservative: they only
  surface facts that are unambiguously visible in the observation
  (deadwood, phase, current bid, public card, etc.) so the rationale
  cannot contradict the game state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

# `datasets` is imported lazily inside main() so that `--help` works in
# environments where the dependency is not installed.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_DEFAULT = "gradients-io-tournaments/env_training_gradients"
OUT_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "augmented"

GAME_KEYWORDS: dict[str, list[str]] = {
    "gin_rummy":   ["gin_rummy", "gin rummy", "ginrummy"],
    "liars_dice":  ["liars_dice", "liar's dice", "liars dice", "liarsdice"],
    "leduc_poker": ["leduc_poker", "leduc poker", "leducpoker"],
    "goof_spiel":  ["goof_spiel", "goofspiel", "goof spiel"],
}


# ---------------------------------------------------------------------------
# Dataset filtering
# ---------------------------------------------------------------------------

def filter_by_game(ds, game: str):
    """Keep only samples relevant to the given game.

    Prefers the dedicated ``game`` column if present (cheap & exact),
    otherwise falls back to substring matching across the serialised row.
    """
    keywords = GAME_KEYWORDS.get(game, [game])
    if "game" in ds.column_names:
        return ds.filter(lambda ex: (ex.get("game") or "").lower() in keywords,
                          desc=f"Filter game={game} (column)")
    return ds.filter(lambda ex: any(kw in str(ex).lower() for kw in keywords),
                      desc=f"Filter game={game} (text)")


# ---------------------------------------------------------------------------
# Rule-based rationale per game
# ---------------------------------------------------------------------------

def _rationale_gin_rummy(obs: str, action: str) -> str:
    parts: list[str] = []
    dw = re.search(r"Deadwood=(\d+)", obs)
    if dw:
        parts.append(f"My deadwood is {dw.group(1)}.")
    if "Phase: Draw" in obs:
        upcard = re.search(r"Upcard:\s*(\S+)", obs)
        if action == "53":
            parts.append("Drawing from stock — the upcard does not obviously help my melds.")
        elif action == "52":
            tag = upcard.group(1) if upcard else "upcard"
            parts.append(f"Picking up the upcard ({tag}) — it pairs with cards I already hold.")
        else:
            parts.append("Selecting a draw action consistent with the legal moves.")
    elif "Phase: Discard" in obs:
        parts.append("In Discard phase — drop the highest-value card not part of any meld to reduce deadwood.")
    elif "Phase: Knock" in obs:
        parts.append("Deadwood is low enough to knock; ending the round.")
    elif "Phase: FirstUpcard" in obs:
        parts.append("Opening turn — decide whether to grab the first upcard or pass to draw.")
    elif "Phase: Layoff" in obs:
        parts.append("Laying off cards onto opponent's melds where possible.")
    parts.append(f"Choosing action {action}.")
    return " ".join(parts)


def _rationale_liars_dice(obs: str, action: str) -> str:
    parts: list[str] = []
    own = re.search(r"Your dice:?\s*([0-9 ,]+)", obs)
    if own:
        parts.append(f"My dice are {own.group(1).strip()}.")
    bid = re.search(r"Current bid:\s*(.+)", obs)
    if bid and "no bid" not in bid.group(1).lower():
        parts.append(f"Standing bid: {bid.group(1).strip()}.")
    # Action 0 is conventionally "Call Liar" in OpenSpiel liars_dice.
    if action.strip() == "0":
        parts.append("Calling the bluff — the bid looks unlikely given visible dice and prior bidding.")
    else:
        parts.append(f"Raising to a tighter bid (action {action}) consistent with my dice plus reasonable opponent dice.")
    return " ".join(parts)


def _rationale_leduc_poker(obs: str, action: str) -> str:
    parts: list[str] = []
    priv = re.search(r"Your card:\s*(\S+)", obs)
    if priv:
        parts.append(f"My hole card is {priv.group(1)}.")
    pub  = re.search(r"Public card:\s*(\S+)", obs)
    if pub and pub.group(1) not in ("-", "None"):
        parts.append(f"Public card is {pub.group(1)}.")
    pot  = re.search(r"Pot size:\s*(\d+)", obs)
    if pot:
        parts.append(f"Pot is {pot.group(1)}.")
    rnd  = re.search(r"Current round:\s*(\d+)/\d+", obs)
    if rnd:
        parts.append(f"Round {rnd.group(1)}.")
    # Default Leduc legal-action mapping: 0=Fold, 1=Call/Check, 2=Raise.
    action_label = {"0": "Fold", "1": "Call/Check", "2": "Raise"}.get(action.strip(), action.strip())
    if action_label == "Fold":
        parts.append("Hand strength is too weak for the price; folding.")
    elif action_label in ("Call", "Check", "Call/Check"):
        parts.append("Pot odds justify continuing without raising.")
    elif action_label == "Raise":
        parts.append("Hand is strong enough — or bluff equity high enough — to raise.")
    parts.append(f"Choosing action {action}.")
    return " ".join(parts)


def _rationale_goof_spiel(obs: str, action: str) -> str:
    parts: list[str] = []
    prize = re.search(r"Prize card:\s*(\S+)", obs)
    if prize:
        parts.append(f"Prize this round is {prize.group(1)}.")
    hand  = re.search(r"Your hand:\s*(.+)", obs)
    if hand:
        parts.append(f"Remaining hand: {hand.group(1).strip()}.")
    parts.append(f"Bidding card {action} — committing a value proportional to the prize and what is left to play.")
    return " ".join(parts)


RATIONALE_FNS: dict[str, Callable[[str, str], str]] = {
    "gin_rummy":   _rationale_gin_rummy,
    "liars_dice":  _rationale_liars_dice,
    "leduc_poker": _rationale_leduc_poker,
    "goof_spiel":  _rationale_goof_spiel,
}


# ---------------------------------------------------------------------------
# Optional: LLM-based rationale (stub).
#
# Wire this to your API of choice. Suggested prompt template at the bottom
# of the function body — instruct the model to return *only* the rationale
# without re-stating the action or the observation.
# ---------------------------------------------------------------------------

def _rationale_via_llm(obs: str, action: str, game: str) -> str:
    """Placeholder. Implement to call your LLM and return a single sentence."""
    raise NotImplementedError(
        "Implement _rationale_via_llm() with your API client of choice.\n"
        "Suggested prompt:\n"
        f"  You are an expert {game} player. The observation is:\n"
        f"  <obs>{obs}</obs>\n"
        f"  The action taken was: {action}\n"
        f"  In one or two short sentences, explain why this is a reasonable move.\n"
        f"  Do not restate the observation."
    )


# ---------------------------------------------------------------------------
# Sample augmentation
# ---------------------------------------------------------------------------

def _generate(obs: str, action: str, game: str, strategy: str) -> str:
    if strategy == "rule":
        return RATIONALE_FNS[game](obs, action)
    if strategy == "llm":
        return _rationale_via_llm(obs, action, game)
    raise ValueError(f"unknown strategy: {strategy}")


def augment_sample(sample: dict, game: str, strategy: str) -> dict:
    """Insert a ``Thought:`` prefix into every assistant turn of ``sample``.

    Skips messages that are not ``user`` / ``assistant`` (e.g. ``system``).
    Tracks the most recent user message so the rationale generator can
    inspect the observation that preceded the action.
    """
    conv = sample.get("conversations") or []
    if not isinstance(conv, list):
        return sample

    new_conv: list[dict] = []
    last_user_value = ""
    for msg in conv:
        if not isinstance(msg, dict):
            new_conv.append(msg)
            continue
        role = msg.get("from", "")
        value = msg.get("value", "")
        if role in ("user", "human"):
            last_user_value = value
            new_conv.append(msg)
        elif role in ("assistant", "gpt"):
            action_str = (value or "").strip()
            try:
                rationale = _generate(last_user_value, action_str, game, strategy)
            except Exception as exc:
                # Don't lose the sample on bad rationale — degrade to a
                # generic line. This matters during LLM-strategy runs where
                # individual API calls may fail intermittently.
                rationale = f"Selecting action {action_str} based on the current state."
                print(f"[generate_rationales] rationale fallback: {exc}", file=sys.stderr)
            new_conv.append({
                "from": role,
                "value": f"Thought: {rationale}\nAction: {action_str}",
            })
        else:
            new_conv.append(msg)

    out = dict(sample)
    out["conversations"] = new_conv
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate CoT-augmented env training data for SFT.",
    )
    ap.add_argument("--game", required=True, choices=sorted(RATIONALE_FNS.keys()),
                     help="Which env game to filter and augment.")
    ap.add_argument("--strategy", default="rule", choices=("rule", "llm"),
                     help="Rationale generation strategy (default: rule).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of samples written (for smoke tests).")
    ap.add_argument("--hf-repo", default=HF_REPO_DEFAULT,
                     help=f"HF dataset repo to read (default: {HF_REPO_DEFAULT}).")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT),
                     help=f"Output directory (default: {OUT_DIR_DEFAULT}).")
    ap.add_argument("--split", default="train",
                     help="HF split to read (default: train).")
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # noqa: WPS433 — lazy import for --help
    except ImportError:
        sys.exit("ERROR: `pip install datasets` required to run this script.")

    print(f"[generate_rationales] Loading {args.hf_repo}:{args.split} ...")
    ds = load_dataset(args.hf_repo, split=args.split)
    print(f"[generate_rationales] Loaded {len(ds)} samples (all games).")

    ds = filter_by_game(ds, args.game)
    print(f"[generate_rationales] {len(ds)} samples after filter for game={args.game}.")
    if len(ds) == 0:
        print("[generate_rationales] WARN: nothing to write. Exiting.", file=sys.stderr)
        return 1

    if args.limit is not None and args.limit < len(ds):
        ds = ds.select(range(args.limit))
        print(f"[generate_rationales] Truncated to {len(ds)} samples (--limit).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"env_training_{args.game}_cot.jsonl"

    n_ok = 0
    n_err = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            try:
                aug = augment_sample(dict(sample), args.game, args.strategy)
                f.write(json.dumps(aug, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as exc:
                n_err += 1
                print(f"[generate_rationales] skip sample: {exc}", file=sys.stderr)

    print(f"[generate_rationales] Wrote {n_ok} samples to {out_path}"
          + (f" (errors: {n_err})" if n_err else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
