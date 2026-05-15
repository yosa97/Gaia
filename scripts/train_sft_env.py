"""Full SFT trainer — pure SFT mode (no GRPO), 2-phase sequential.

Replaces train_grpo_env.py for branches dedicated to full SFT. Reads two
miner-requested whitelisted datasets and trains in TWO sequential phases:

  Phase 1 — Boardgame-QA warm-up (gentle, brief):
    - 500 randomly sampled rows (from 15000)
    - LR 5e-6, 1 epoch, no neftune
    - Goal: instill general logical reasoning patterns into LoRA adapter
    - ~16 steps, ~3 min on 4 GPU 7B

  Phase 2 — env_training specialize (DOMINANT):
    - single-game filter (training_args.environment_name) + OUTCOME-WEIGHTED
      oversampling (win=5, draw=3, loss=1) instead of hard win/draw filter
    - For LD: 1100 -> 400 -> ~1500 effective samples (after weighting)
    - LR 1e-5, 10 epoch (chess SFT paper recipe), neftune_noise_alpha=5
    - Goal: specialize adapter to game action format
    - ~120-160 steps, ~25-35 min on 2 GPU 3B

Same model+LoRA adapter across both phases (gradient accumulation). Phase 2
update magnitude (steps × LR) is ~4.5× Phase 1, so final adapter direction
is ~80% Phase 2 + 20% Phase 1 reasoning residual.

Hard time budget enforced via TimeBudgetCallback (default 165 min = 2h45m,
leaves 15 min buffer pre-3h cap). Triggered via SFT_ONLY=1 env var routed
through scripts/text_trainer.py. Direct download fallback if
MINER_DATASETS_DIR not mounted (production-ready before PR #1082 merges).
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import PeftModelForCausalLM, get_peft_model
from transformers import AutoTokenizer, TrainerCallback
from transformers.modeling_utils import is_deepspeed_zero3_enabled
from transformers.trainer_utils import is_main_process
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from utility import log_info


LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

ENV_TRAINING_DATASET_REPO = "gradients-io-tournaments/env_training_gradients"
ENV_TRAINING_DATASET_DIR = "gradients-io-tournaments__env_training_gradients"
BOARDGAME_QA_REPO = "tasksource/Boardgame-QA"
BOARDGAME_QA_DIR = "tasksource__Boardgame-QA"
FALLBACK_CACHE_DIR = "/tmp/sft_dataset_cache"

DEFAULT_BUDGET_MIN = 165
TARGET_GAMES = ("liars_dice", "leduc_poker", "gin_rummy")
TARGET_OUTCOMES = ("win", "draw")
# PvP: raise win weight 5 -> 8. Model must learn from winning trajectories
# more aggressively against diverse human/AI opponent styles.
OUTCOME_WEIGHTS = {"win": 8, "draw": 3, "loss": 1}

# Gin-Rummy specialist dataset (whitelisted) — only loaded for gin_rummy game.
GIN_RUMMY_TRAJ_REPO = "GoodStartLabs/gin-rummy-trajectories-32k"
GIN_RUMMY_TRAJ_DIR  = "GoodStartLabs__gin-rummy-trajectories-32k"


# === Game-specific system prompts (verbatim from scripts/envs/<game>_opponent_modeling.py) ===
# CRITICAL: validator routes via _VARIANT_OVERRIDES in scripts/envs/env_configs.py
#   "liars_dice" -> "liars_dice_opponent_modeling"
#   "leduc_poker" -> "leduc_poker_opponent_modeling"
#   "gin_rummy" -> "gin_rummy_opponent_modeling"
# So inference uses *_opponent_modeling.py prompts, NOT the base *_env.py prompts.
# Differences (NOT cosmetic):
#   - LD: opponent_modeling has NO leading/trailing '"' wrapping (base does)
#   - LP: examples differ ("Fold"/"Raise" vs generic "roll"/"a3")
#   - GR: opponent_modeling has "EACH TURN" section + card notation example
# === Game-specific HINT prompts (verbatim from scripts/envs/<game>_opponent_modeling.py) ===
# Validator's env may add these conditionally via use_hints flag at inference.
# For SFT, randomly include 50% of training samples → model robust to both
# conditions (hint/no-hint), mirrors GRPO's curriculum hint_prob mechanism.
LIARS_DICE_HINT_PROMPT = (
    '\n# Strategy Tips\n'
    '- Count your dice that match the bid (including 6s as wild)\n'
    '- Call "Liar" when the bid is more likely false than any available bid is true.\n'
    '- Make conservative bids early, aggressive when opponent seems weak\n'
)

LEDUC_POKER_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "Round 1:\n"
    "- Hold K or Q → call a raise; raise first if unchallenged.\n"
    "- Hold J → fold against a raise; check if unchallenged.\n\n"
    "Round 2 (public card revealed):\n"
    "- You have a PAIR → raise; never fold.\n"
    "- You have K (no pair) → raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is K → raise first; call if opponent raises.\n"
    "- You have Q (no pair), public card is J → check; fold if opponent raises.\n"
    "- You have J (no pair) → check; fold if opponent raises.\n"
)

GIN_RUMMY_HINT_PROMPT = (
    "\n\n# Strategy Tips\n"
    "- Early game: Draw from deck to see more cards\n"
    "- Build runs and sets to reduce deadwood\n"
    "- Track opponent's discards to guess their hand\n"
    "- Knock when you have ≤10 deadwood points and think you're ahead\n"
    "- Go for Gin (0 deadwood) when close for bonus points\n"
    "- In Layoff phase: use 'Dead cards' hint to find extension opportunities\n"
    "- IMPORTANT: YOU MUST PICK THE ACTION ID FROM THE LEGAL ACTIONS."
)

GAME_HINT_PROMPTS = {
    "liars_dice": LIARS_DICE_HINT_PROMPT,
    "leduc_poker": LEDUC_POKER_HINT_PROMPT,
    "gin_rummy": GIN_RUMMY_HINT_PROMPT,
}


LIARS_DICE_SYSTEM_PROMPT = (
    'You are playing liars_dice.\n\n# Game Rules\nLIAR\'S DICE RULES:\n\n'
    'Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.\n\n'
    'Goal: Make bids about total dice across ALL players, or call "Liar" on opponent\'s bid.\n\n'
    'Actions:\n- Bid (quantity, face): Claim there are at least \'quantity\' dice showing \'face\' among all dice.\n'
    '- Call Liar: Challenge the previous bid.\n\n'
    'Bidding rules: Each bid must be higher than the previous bid. "Higher" means:\n'
    '  - Same face value but higher quantity (e.g., "2 fours" beats "1 four")\n'
    '  - Same quantity but higher face value (e.g., "2 fives" beats "2 fours")\n\n'
    'Wild dice: 6s are WILD and count as ANY face value.\n'
    '- When counting dice for a bid, include 6s in the count\n'
    '- Example: Bid "3 fours" means at least 3 dice showing EITHER 4 OR 6\n\n'
    'Winning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.\n\n\n\n'
    '# Output Format\nYou must respond with ONLY the action ID (a single number).\n'
    'Do NOT include descriptions or explanations.\n\n'
    'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
)

LEDUC_POKER_SYSTEM_PROMPT = (
    "You are playing leduc_poker.\n\n"
    "# Game Rules\n"
    "LEDUC POKER RULES:\n\n"
    "Deck: 2 suits × (num_players + 1) ranks. For 2 players: 6 cards (J♠ J♥ Q♠ Q♥ K♠ K♥).\n\n"
    "Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.\n\n"
    "Round 1: Each player receives one private card. Actions: Fold (lose ante), Call/Check "
    "(match current bet), Raise (add 2 chips to bet). Maximum 2 raises per round.\n"
    "Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.\n\n"
    "Winning: Player with best hand wins pot (or last remaining if others fold).\n"
    "Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).\n\n\n\n"
    "# Output Format\n"
    "You must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    "Examples:\n"
    '- For action "0 -> Fold": respond "0"\n'
    '- For action "2 -> Raise": respond "2"'
)

GIN_RUMMY_SYSTEM_PROMPT = (
    "You are playing gin_rummy.\n\n# Game Rules\nGIN RUMMY RULES:\n\n"
    "SETUP:\n- 52-card deck, each player receives 7-10 cards (variant dependent)\n"
    "- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)\n\n"
    "MELDS (Valid Combinations):\n"
    "1. SET: 3+ cards of SAME RANK (e.g., 7♠ 7♥ 7♣)\n"
    "2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5♦ 6♦ 7♦)\n"
    "Examples:\n- Valid runs: A♠-2♠-3♠, 9♥-10♥-J♥-Q♥, 10♣-J♣-Q♣-K♣\n"
    "- Invalid: K♠-A♠-2♠ (Ace is LOW only, not wraparound)\n\n"
    "CARD NOTATION:\n- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)\n"
    "- Suits: s(♠), h(♥), d(♦), c(♣)\n"
    "- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades\n\n"
    "GAME PHASES:\n"
    "1. FirstUpcard: 52=Draw upcard, 54=Pass\n"
    "2. Draw: 52=Draw upcard, 53=Draw stock\n"
    "3. Discard: action ID = card index (shown in Legal Actions)\n"
    "4. Layoff: card indices or 54=Pass\n"
    "5. Knock: declare end when deadwood ≤ knock_card\n\n"
    "EACH TURN:\n1. DRAW: stock (53) or upcard (52)\n"
    "2. DISCARD: choose a card by action ID\n\n"
    "KNOCKING:\n- Gin: 0 deadwood = 25-point bonus\n\n"
    "SCORING: Winner scores difference in deadwood.\n"
    "Card Values: A=1, 2-10=face value, J=11, Q=12, K=13\n\n"
    "IMPORTANT: Always respond with the action ID number ONLY, never card names.\n\n"
    "# Output Format\nYou must respond with ONLY the action ID (a single number).\n"
    "Do NOT include descriptions or explanations.\n\n"
    'Examples:\n- For action "0 -> roll": respond "0"\n- For action "89 -> a3": respond "89"'
)

# All 3 prompts verbatim from scripts/envs/<game>_env.py base prompt (no hints).
# At inference, env may add _HINT_PROMPT but base is the safe minimum overlap.
GAME_SYSTEM_PROMPTS = {
    "liars_dice": LIARS_DICE_SYSTEM_PROMPT,
    "leduc_poker": LEDUC_POKER_SYSTEM_PROMPT,
    "gin_rummy": GIN_RUMMY_SYSTEM_PROMPT,
}


# === Chat templates with {% generation %} markers for assistant_only_loss ===
# Default templates per model family lack generation markers; TRL silently
# disables AOL. We detect the family and replace with a minimal compatible
# template that includes markers so AOL works correctly.
#
# Coverage of 9-model tournament lineup:
#   - Qwen2.5/Qwen3 (ChatML): 5 of 9 models
#   - Mistral-7B / CodeLlama-7B ([INST]): 2 of 9 models
#   - Llama-3-8B (special header tags): 1 of 9 models
#   - Other 7B (varies)

CHATML_TEMPLATE_WITH_GENERATION = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# Mistral [INST]/[/INST] family (Mistral-7B-Instruct, CodeLlama-Instruct).
# System message prepended as a separate [INST] turn with "Understood." reply
# to maintain pair structure. Not Mistral's native convention but works for
# SFT loss computation since generation markers wrap assistant content.
MISTRAL_INST_TEMPLATE_WITH_GENERATION = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "[INST] {{ message['content'] }} [/INST] Understood.{{ eos_token }}"
    "{% elif message['role'] == 'user' %}"
    "[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}"
    " {% generation %}{{ message['content'] }}{% endgeneration %}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

# Llama-3 special header tags (Llama-3-8B-Instruct, Llama-3.1, Llama-3.2).
LLAMA3_TEMPLATE_WITH_GENERATION = (
    "<|begin_of_text|>"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% generation %}{{ message['content'] }}{% endgeneration %}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def _try_patch_chat_template(tokenizer) -> tuple[bool, str]:
    """Inject {% generation %} markers based on detected template family.

    Returns (supports_aol, family) tuple.
    family ∈ {"chatml", "mistral", "llama3", "unknown", "preexisting"}
    """
    tmpl = tokenizer.chat_template or ""

    # Already has markers? Use existing template as-is.
    if "{% generation %}" in tmpl or "{%- generation %}" in tmpl:
        return True, "preexisting"

    # ChatML (Qwen2/Qwen2.5/Qwen3/Yi/etc)
    if "<|im_start|>" in tmpl and "<|im_end|>" in tmpl:
        tokenizer.chat_template = CHATML_TEMPLATE_WITH_GENERATION
        return True, "chatml"

    # Llama-3 (header_id tags) — check before Mistral since Llama-3 may include [INST] strings in some variants
    if "<|start_header_id|>" in tmpl and "<|end_header_id|>" in tmpl:
        tokenizer.chat_template = LLAMA3_TEMPLATE_WITH_GENERATION
        return True, "llama3"

    # Mistral / CodeLlama [INST]/[/INST]
    if "[INST]" in tmpl and "[/INST]" in tmpl:
        tokenizer.chat_template = MISTRAL_INST_TEMPLATE_WITH_GENERATION
        return True, "mistral"

    return False, "unknown"


def _strip_thought_prefix(content: str) -> str:
    """Extract action ID from 'Thought: ... Action: N' format.

    Mirrors env parser at scripts/envs/liar_dice_env.py:455-456:
    if 'Action:' in completion: completion.split('Action:')[-1].strip()

    Training data format: 'Thought:\\n<reasoning>\\n\\nAction:\\n<id>'
    Env system prompt: 'respond with ONLY the action ID (a single number)'
    Strip the Thought prefix so SFT teaches the format the validator expects.
    """
    if "Action:" in content:
        return content.split("Action:")[-1].strip()
    return content.strip()


@dataclass
class TrainingArguments(SFTConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    environment_name: Optional[str] = field(default=None)


class TimeBudgetCallback(TrainerCallback):
    """Hard-stop training when wall-clock budget exceeded."""

    def __init__(self, budget_seconds: float):
        self.budget_seconds = budget_seconds
        self._start: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._start is None:
            return control
        if time.time() - self._start > self.budget_seconds:
            log_info(
                f"[full_sft] time budget {self.budget_seconds:.0f}s reached at "
                f"step {state.global_step}; stopping"
            )
            control.should_training_stop = True
        return control


def _direct_download(repo: str, target_dir: str, target_root: str) -> bool:
    target = Path(target_root) / target_dir
    if target.exists() and any(target.iterdir()):
        log_info(f"[full_sft] dataset cached at {target}")
        return True
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(target),
            token=token,
        )
        log_info(f"[full_sft] direct-downloaded {repo} -> {target}")
        return True
    except Exception as exc:
        log_info(f"[full_sft] download {repo} failed: {exc}")
        return False


def _resolve_dataset_root() -> str:
    miner_dir = os.getenv("MINER_DATASETS_DIR")
    miner_list = os.getenv("MINER_DATASETS", "")
    if miner_dir and ENV_TRAINING_DATASET_DIR in miner_list:
        return miner_dir
    log_info("[full_sft] MINER_DATASETS_DIR not set; using fallback download")
    Path(FALLBACK_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return FALLBACK_CACHE_DIR


def _load_env_training(root: str, target_game: str) -> Optional[Dataset]:
    """Load env_training data. Prefer pre-generated CoT-augmented JSONL if present.

    Augmented data path: scripts/data/augmented/env_training_<game>_cot.jsonl
    Generated by scripts/synthetic_data/generate_rationales.py (offline).

    If augmented file exists AND env var USE_AUGMENTED_DATA != "0", uses that
    (Phase 1: CoT rationale augmentation per Distilling Step-by-Step). Else
    falls back to standard HF dataset.
    """
    augmented_path = Path(__file__).parent / "data" / "augmented" / f"env_training_{target_game}_cot.jsonl"
    use_augmented = augmented_path.exists() and os.getenv("USE_AUGMENTED_DATA", "1") != "0"

    if use_augmented:
        log_info(f"[full_sft] CoT-augmented data found at {augmented_path}, using it (USE_AUGMENTED_DATA=1)")
        try:
            ds = load_dataset("json", data_files=str(augmented_path), split="train")
            log_info(f"[full_sft] augmented dataset loaded: {len(ds)} samples")
            strip_thought = False
        except Exception as exc:
            log_info(f"[full_sft] augmented load failed: {exc}; falling back to standard HF dataset")
            use_augmented = False

    if not use_augmented:
        if not _direct_download(ENV_TRAINING_DATASET_REPO, ENV_TRAINING_DATASET_DIR, root):
            return None
        try:
            ds = load_dataset(str(Path(root) / ENV_TRAINING_DATASET_DIR), split="train")
        except Exception as exc:
            log_info(f"[full_sft] env_training load failed: {exc}")
            return None
        n0 = len(ds)
        ds = ds.filter(lambda r: r.get("game") == target_game)
        n1 = len(ds)

        # Point #1 — Outcome-weighted oversampling instead of hard filter.
        # Previously: filter to win+draw only (drops loss samples entirely).
        # Now: keep ALL outcomes, oversample by weight to bias training toward
        # high-quality demonstrations while still extracting state-coverage
        # signal from loss trajectories.
        outcome_counts = {}
        for row in ds:
            oc = row.get("outcome", "unknown")
            outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
        log_info(f"[full_sft] outcome distribution (pre-weighting): {outcome_counts}")

        weighted_parts = []
        for outcome, weight in OUTCOME_WEIGHTS.items():
            subset = ds.filter(lambda r, o=outcome: r.get("outcome") == o)
            if len(subset) == 0:
                continue
            for _ in range(weight):
                weighted_parts.append(subset)

        if not weighted_parts:
            log_info(f"[full_sft] no samples matched any of {list(OUTCOME_WEIGHTS)}; aborting")
            return None
        ds = concatenate_datasets(weighted_parts)
        n2 = len(ds)
        log_info(
            f"[full_sft] env_training: {n0} -> {n1} (game={target_game}) -> "
            f"{n2} (outcome-weighted oversampling: {OUTCOME_WEIGHTS})"
        )
        strip_thought = True

    # Tier 1 — Curriculum data ordering: sort by num_turns ascending (easy first).
    # Mirrors GRPO's max_turn ↑ scheduler, but as static dataset ordering.
    if "num_turns" in ds.column_names:
        ds = ds.sort("num_turns")
        log_info(f"[full_sft] sorted samples by num_turns ascending (curriculum)")

    # Tier 1 — Hint randomization (50%) + multi-prompt diversity (25% concise suffix).
    # Deterministic per-index for reproducibility. Combined: 4 system-prompt
    # variants distributed roughly 25% each across the dataset.
    columns_to_remove = ds.column_names

    def _map_fn(row, idx):
        add_hint = (idx % 2 == 0)              # 50% with hint
        add_concise = (idx % 4 < 2)            # 50% with concise suffix
        return _sharegpt_to_messages(
            row,
            target_game=target_game,
            add_hint=add_hint,
            add_concise_suffix=add_concise,
            strip_thought=strip_thought,
        )

    log_info(
        f"[full_sft] applying Tier 1 augmentations: hint randomization (50%), "
        f"concise-suffix variant (50%) — 4 system-prompt variants total "
        f"(strip_thought={strip_thought})"
    )
    return ds.map(_map_fn, with_indices=True, remove_columns=columns_to_remove)


def _load_gin_rummy_trajectories(root: str) -> Optional[Dataset]:
    """Load GoodStartLabs/gin-rummy-trajectories-32k for gin_rummy specialist training.

    This dataset is only used as Phase 2.5 for gin_rummy game — it contains
    32k expert game traces that sharpen meld/knock/discard decision making.
    Schema: {observation, action, outcome} or ShareGPT {conversations} format.
    """
    miner_list = os.getenv("MINER_DATASETS", "")
    if GIN_RUMMY_TRAJ_DIR in miner_list:
        # Pre-mounted by run_anon.sh/run_augm.sh via MINER_DATASETS_3
        dataset_root = os.getenv("MINER_DATASETS_DIR", root)
        dataset_path = str(Path(dataset_root) / GIN_RUMMY_TRAJ_DIR)
    else:
        # Fallback: direct download
        if not _direct_download(GIN_RUMMY_TRAJ_REPO, GIN_RUMMY_TRAJ_DIR, root):
            return None
        dataset_path = str(Path(root) / GIN_RUMMY_TRAJ_DIR)

    try:
        ds = load_dataset(dataset_path, split="train")
    except Exception:
        try:
            ds = load_dataset(GIN_RUMMY_TRAJ_REPO, split="train")
        except Exception as exc:
            log_info(f"[full_sft] gin-rummy-trajectories load failed: {exc}")
            return None

    log_info(f"[full_sft] gin-rummy-trajectories loaded: {len(ds)} samples; columns={ds.column_names}")

    # Convert to messages format — handle both ShareGPT and flat schemas.
    def _traj_to_messages(row):
        # ShareGPT format: {conversations: [{from, value}]}
        if "conversations" in row:
            return _sharegpt_to_messages(row, target_game="gin_rummy",
                                         add_hint=False, strip_thought=True)
        # Flat format: {observation, action} or {prompt, response}
        obs = row.get("observation") or row.get("prompt") or row.get("input") or ""
        act = row.get("action") or row.get("response") or row.get("output") or ""
        return {
            "messages": [
                {"role": "system", "content": GIN_RUMMY_SYSTEM_PROMPT},
                {"role": "user",   "content": str(obs)},
                {"role": "assistant", "content": str(act)},
            ]
        }

    return ds.map(_traj_to_messages, remove_columns=ds.column_names)


def _load_boardgame_qa(root: str) -> Optional[Dataset]:
    if not _direct_download(BOARDGAME_QA_REPO, BOARDGAME_QA_DIR, root):
        return None
    try:
        ds = load_dataset(str(Path(root) / BOARDGAME_QA_DIR), split="train")
    except Exception as exc:
        log_info(f"[full_sft] Boardgame-QA load failed: {exc}")
        return None
    log_info(f"[full_sft] Boardgame-QA loaded: {len(ds)} samples; columns={ds.column_names}")
    return ds.map(_qa_to_messages, remove_columns=ds.column_names)


def _sharegpt_to_messages(
    row: dict,
    target_game: str = None,
    add_hint: bool = False,
    add_concise_suffix: bool = False,
    strip_thought: bool = True,
) -> dict:
    """Convert ShareGPT format to messages, with alignment + diversity fixes:

    Fix 2: prepend env's system prompt to match validator inference distribution.
    Fix 3 (conditional): strip 'Thought:' reasoning prefix from assistant turns.
        - strip_thought=True for original env_training_gradients (default).
        - strip_thought=False for CoT-augmented data (we WANT the rationale).

    Tier 1 enhancements (port GRPO mechanisms):
    - add_hint: append _HINT_PROMPT to system (mirrors GRPO use_hints toggle)
    - add_concise_suffix: append "Be precise..." reminder (multi-prompt diversity)
    """
    role_map = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant", "system": "system"}
    msgs = []

    # Fix 2 + Tier 1 hint randomization: build system prompt with optional hint
    if target_game and target_game in GAME_SYSTEM_PROMPTS:
        sys_content = GAME_SYSTEM_PROMPTS[target_game]
        if add_hint and target_game in GAME_HINT_PROMPTS:
            sys_content = sys_content + GAME_HINT_PROMPTS[target_game]
        if add_concise_suffix:
            sys_content = sys_content + "\n\nBe precise and concise. Output only the action ID."
        msgs.append({"role": "system", "content": sys_content})

    for turn in row.get("conversations") or []:
        role = role_map.get(str(turn.get("from", "")).lower(), "user")
        content = turn.get("value", "")
        # Fix 3: strip Thought reasoning from assistant — keep action ID only.
        # Skipped for CoT-augmented data (rationale is the point).
        if role == "assistant" and strip_thought:
            content = _strip_thought_prefix(content)
        msgs.append({"role": role, "content": content})

    return {"messages": msgs}


def _qa_to_messages(row: dict) -> dict:
    """Adapt Boardgame-QA rows to messages format.

    Boardgame-QA schema (verified 2026-05-03):
      theory, facts, rules, preferences, goal -> prompt
      proof, label -> response
    """
    if "goal" in row and "label" in row:
        prompt_parts = []
        if row.get("theory"):
            prompt_parts.append(f"Theory:\n{row['theory']}")
        if row.get("facts"):
            prompt_parts.append(f"Facts:\n{row['facts']}")
        if row.get("rules"):
            prompt_parts.append(f"Rules:\n{row['rules']}")
        if row.get("preferences"):
            prompt_parts.append(f"Preferences:\n{row['preferences']}")
        prompt_parts.append(f"Question: {row['goal']}")
        user_content = "\n\n".join(prompt_parts)

        response_parts = []
        if row.get("proof"):
            response_parts.append(f"Proof:\n{row['proof']}")
        response_parts.append(f"Answer: {row['label']}")
        assistant_content = "\n\n".join(response_parts)

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    # Generic fallback for other Q/A schemas
    question = (
        row.get("question")
        or row.get("input")
        or row.get("prompt")
        or row.get("instruction")
        or ""
    )
    answer = (
        row.get("answer")
        or row.get("output")
        or row.get("response")
        or row.get("completion")
        or ""
    )
    return {
        "messages": [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)},
        ]
    }


def _interleave_5050(env_ds: Dataset, qa_ds: Dataset, seed: int = 42) -> Dataset:
    """Interleave env_training and Boardgame-QA roughly 50/50 per batch.

    Uses datasets.interleave_datasets with stopping_strategy='all_exhausted'
    so model sees full coverage of both. env_training (smaller) gets resampled.
    """
    from datasets import interleave_datasets

    return interleave_datasets(
        [env_ds, qa_ds],
        probabilities=[0.5, 0.5],
        seed=seed,
        stopping_strategy="all_exhausted",
    )


def main():
    print("--------------------------------")
    print("FULL SFT TRAINING (no GRPO)")
    print("--------------------------------")
    try:
        argument_parser = transformers.HfArgumentParser((TrainingArguments, ModelConfig))
        training_args, model_args = argument_parser.parse_args_into_dataclasses()

        train_info = json.load(open(training_args.request_path, "r"))
        train_request = train_info["train_request"]
        budget_min = float(train_request.get("budget_min", DEFAULT_BUDGET_MIN))

        tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Fix 1: patch chat template with {% generation %} markers per-family.
        # Now supports ChatML (Qwen), Mistral [INST], Llama-3 header_id formats.
        supports_aol, tmpl_family = _try_patch_chat_template(tokenizer)
        if not supports_aol and training_args.assistant_only_loss:
            if is_main_process(LOCAL_RANK):
                log_info(
                    f"[full_sft] WARNING: chat_template family unknown ({tmpl_family}); "
                    "disabling assistant_only_loss to avoid TRL crash. Model will train "
                    "on user tokens too (suboptimal but functional)."
                )
            training_args.assistant_only_loss = False
        elif supports_aol and is_main_process(LOCAL_RANK):
            log_info(
                f"[full_sft] chat_template patched with generation markers (family={tmpl_family}) — "
                "assistant_only_loss=True will mask user tokens correctly"
            )

        quantization_config = get_quantization_config(model_args)
        device_string = "cuda:" + str(LOCAL_RANK)
        device_map = (
            get_kbit_device_map()
            if quantization_config is not None
            else {"": device_string}
        )
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            device_map = None

        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=device_map,
            quantization_config=quantization_config,
        )

        log_info(f"[full_sft] training_args: {training_args}")

        if training_args.use_liger:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            model_class = AutoLigerKernelForCausalLM
        else:
            model_class = transformers.AutoModelForCausalLM

        model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)

        peft_config = get_peft_config(model_args)
        if "lora_model" in train_request:
            model = PeftModelForCausalLM.from_pretrained(
                model, train_request["lora_model"], is_trainable=True, **model_kwargs
            )
        elif peft_config is not None:
            # CRITICAL: pre-wrap model so adapter persists across Phase 1 → Phase 2.
            # Otherwise SFTTrainer wraps internally but outer `model` variable stays
            # unwrapped, making Phase 2 train the BASE model (full FT) instead of LoRA.
            model = get_peft_model(model, peft_config)
            if is_main_process(LOCAL_RANK):
                log_info("[full_sft] Pre-wrapped model with PEFT (LoRA r=64) — "
                         "adapter shared across Phase 1 → Phase 2")

        if is_main_process(LOCAL_RANK):
            os.makedirs(training_args.output_dir, exist_ok=True)
            log_info(f"[full_sft] Created output directory: {training_args.output_dir}")

        # Load datasets — single-game env (282 LD/LP/GR win-draw) + QA subset (500 of 15K)
        root = _resolve_dataset_root()
        target_game = training_args.environment_name or "liars_dice"
        env_ds = _load_env_training(root, target_game=target_game)
        if env_ds is None:
            raise RuntimeError(f"[full_sft] env_training unavailable for game={target_game}; aborting")

        qa_full = _load_boardgame_qa(root)
        qa_subset = None
        if qa_full is not None:
            phase1_n = min(500, len(qa_full))
            qa_subset = qa_full.shuffle(seed=42).select(range(phase1_n))
            log_info(f"[full_sft] Phase 1 QA subset: {phase1_n} from {len(qa_full)}")
        else:
            log_info("[full_sft] Boardgame-QA unavailable — skipping Phase 1 warm-up")

        # === PHASE 1: Boardgame-QA warm-up (gentle, brief) ===
        if qa_subset is not None:
            phase1_args = copy.deepcopy(training_args)
            phase1_args.learning_rate = 5e-6
            phase1_args.num_train_epochs = 1
            phase1_args.warmup_ratio = 0.1
            phase1_args.warmup_steps = 0
            phase1_args.neftune_noise_alpha = None
            phase1_args.output_dir = os.path.join(training_args.output_dir, "phase1")
            phase1_args.report_to = []  # avoid duplicate wandb run init
            # phase1_args.assistant_only_loss inherits from training_args (already detected)
            if is_main_process(LOCAL_RANK):
                os.makedirs(phase1_args.output_dir, exist_ok=True)

            log_info(
                f"[full_sft] Phase 1 START — n={len(qa_subset)} epoch=1 lr=5e-6 "
                f"(QA warm-up, gentle)"
            )
            trainer1 = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                args=phase1_args,
                train_dataset=qa_subset,
                peft_config=None,  # model already wrapped with PEFT above
                callbacks=[TimeBudgetCallback(budget_min * 60.0)],
            )
            trainer1.train()
            log_info("[full_sft] Phase 1 DONE — adapter has QA reasoning warm-up signal")

        # === PHASE 2: env specialize (DOMINANT — same model+adapter, accumulate) ===
        # training_args.assistant_only_loss already set based on chat template detection
        log_info(
            f"[full_sft] Phase 2 START — env={target_game} n={len(env_ds)} "
            f"epoch={training_args.num_train_epochs} lr={training_args.learning_rate} "
            f"budget={budget_min}min HARD"
        )
        trainer2 = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=env_ds,
            peft_config=None,  # model already wrapped with PEFT above
            callbacks=[TimeBudgetCallback(budget_min * 60.0)],
        )
        trainer2.train()

        # === PHASE 2.5: Gin Rummy specialist (only if target_game == gin_rummy) ===
        # GoodStartLabs/gin-rummy-trajectories-32k — 32k expert traces.
        # Applied AFTER Phase 2 so main env data provides the base,
        # and specialist data sharpens meld/knock/discard strategy on top.
        # Inherits same adapter (no PEFT re-wrap needed).
        if target_game == "gin_rummy":
            gin_traj_ds = _load_gin_rummy_trajectories(root)
            if gin_traj_ds is not None:
                # Sample up to 3000 rows to keep training time bounded.
                n_traj = min(3000, len(gin_traj_ds))
                gin_traj_ds = gin_traj_ds.shuffle(seed=42).select(range(n_traj))
                phase25_args = copy.deepcopy(training_args)
                phase25_args.num_train_epochs = 3
                phase25_args.learning_rate = 5e-6   # gentle — specialist fine-tune
                phase25_args.output_dir = os.path.join(training_args.output_dir, "phase25")
                phase25_args.report_to = []
                if is_main_process(LOCAL_RANK):
                    os.makedirs(phase25_args.output_dir, exist_ok=True)
                log_info(
                    f"[full_sft] Phase 2.5 START — gin_rummy specialist "
                    f"n={n_traj} epoch=3 lr=5e-6"
                )
                trainer25 = SFTTrainer(
                    model=model,
                    processing_class=tokenizer,
                    args=phase25_args,
                    train_dataset=gin_traj_ds,
                    peft_config=None,
                    callbacks=[TimeBudgetCallback(budget_min * 60.0)],
                )
                trainer25.train()
                log_info("[full_sft] Phase 2.5 DONE — gin_rummy specialist complete")
            else:
                log_info("[full_sft] Phase 2.5 SKIPPED — gin-rummy-trajectories unavailable")

        if is_main_process(LOCAL_RANK):
            final_dir = train_request.get("submission_dir", training_args.output_dir)
            # Save LoRA adapter (NOT merged). Validator eval pipeline detects
            # adapter_config.json → downloads base from original_model + loads
            # our adapter via SGLang --enable-lora. Saving merged breaks the
            # eval volume mount (snapshot symlinks point outside mount root).
            trainer2.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
            log_info(f"[full_sft] Phase 2 DONE — final LoRA adapter saved to {final_dir}")

            with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
                f.write("Success")

    except Exception as e:
        import traceback
        print(f"Error training: {e}")
        print(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()