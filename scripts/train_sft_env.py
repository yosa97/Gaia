"""
train_sft_env.py — SFT warm-start trainer untuk environment tasks (G.O.D tournament).

Pipeline:
  SFT warm-start (file ini) → checkpoint → GRPO (train_grpo_env.py)

Flow:
  1. Load dataset dari HuggingFace (pakai whitelisted dataset ID)
  2. Game-aware field mapping (gin_rummy / poker / textarena / generic)
  3. Tokenize dengan apply_chat_template (format per model)
  4. Train dengan TRL SFTTrainer (lebih robust dari bare HF Trainer)
  5. Simpan checkpoint → path diteruskan ke GRPO sebagai base model

Jika dataset tidak tersedia → raise DatasetNotAvailableError
agar text_trainer.py bisa skip SFT dan langsung GRPO.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import is_main_process
from peft import LoraConfig

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


# ──────────────────────────────────────────────────────────────────────────────
# Local path helper — fix HFValidationError di HuggingFace Hub versi baru
# ──────────────────────────────────────────────────────────────────────────────
def _is_local_path(path: str) -> bool:
    """Cek apakah path adalah local filesystem path (bukan HF repo ID)."""
    return path.startswith("/") or path.startswith("./") or os.path.isdir(path)


def _resolve_model_path(path: str) -> str:
    """Resolve symlink dan kembalikan absolute path untuk local models.

    HuggingFace Hub baru (>=0.24) strict validate repo ID — path lokal
    dengan banyak '/' akan ditolak. Solusi: deteksi local path dan
    gunakan os.path.realpath() untuk normalisasi.
    """
    if _is_local_path(path):
        real = os.path.realpath(path)
        if os.path.isdir(real):
            return real
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Custom exception — ditangkap text_trainer.py untuk skip SFT → langsung GRPO
# ──────────────────────────────────────────────────────────────────────────────
class DatasetNotAvailableError(RuntimeError):
    """Raised when SFT dataset tidak bisa diload atau kosong setelah filtering."""


# ──────────────────────────────────────────────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SftEnvArguments(transformers.TrainingArguments):
    request_path:       Optional[str]  = field(default=None)
    use_lora:           Optional[bool] = field(default=True)
    disable_fa:         Optional[bool] = field(default=False)
    sft_dataset_id:     Optional[str]  = field(default=None,
                            metadata={"help": "HuggingFace dataset repo ID untuk SFT warm-start"})
    sft_dataset_split:  Optional[str]  = field(default="train",
                            metadata={"help": "Split yang dipakai (train/test/validation)"})
    max_sft_samples:    Optional[int]  = field(default=5000,
                            metadata={"help": "Max jumlah sample SFT (cukup kecil utk warm-start)"})


# ──────────────────────────────────────────────────────────────────────────────
# Game-aware field mapping
# Format: dataset_id → fungsi yang mengkonversi satu row ke dict {system, user, assistant}
# ──────────────────────────────────────────────────────────────────────────────

def _map_gin_rummy(row: dict) -> dict | None:
    """Gin Rummy: GoodStartLabs/gin-rummy-trajectories-32k, ArkadiumGinrummy."""
    # Coba beberapa kemungkinan field schema dari dataset HF
    obs   = row.get("observation") or row.get("state") or row.get("prompt") or row.get("input") or ""
    act   = row.get("action") or row.get("response") or row.get("output") or row.get("completion") or ""
    reason = row.get("reasoning") or row.get("rationale") or ""

    if not obs or not act:
        return None

    user_text = f"You are playing Gin Rummy.\n\nGame state:\n{obs}"
    assistant_text = f"{reason}\n\nAction: {act}".strip() if reason else f"Action: {act}"
    return {
        "system":    "You are an expert Gin Rummy player. Analyze the game state carefully and choose the best action.",
        "user":      user_text,
        "assistant": assistant_text,
    }


def _map_poker(row: dict) -> dict | None:
    """Poker: SoelMgd/Poker_Dataset, RZ412/PokerBench."""
    obs   = row.get("prompt") or row.get("question") or row.get("observation") or row.get("input") or ""
    act   = row.get("completion") or row.get("answer") or row.get("action") or row.get("output") or ""

    if not obs or not act:
        return None

    return {
        "system":    "You are an expert poker player. Analyze the hand and betting situation, then decide the optimal action.",
        "user":      obs,
        "assistant": act,
    }


def _map_textarena(row: dict) -> dict | None:
    """TextArena multi-game: the-acorn-ai/textarena-player-game-traces."""
    game   = row.get("game_name") or row.get("game") or "unknown game"
    obs    = row.get("observation") or row.get("state") or row.get("prompt") or ""
    act    = row.get("action") or row.get("response") or row.get("output") or ""

    if not obs or not act:
        return None

    return {
        "system":    f"You are an expert {game} player. Choose actions that maximize your chance of winning.",
        "user":      obs,
        "assistant": act,
    }


def _map_boardgame_qa(row: dict) -> dict | None:
    """Boardgame QA: tasksource/Boardgame-QA."""
    q = row.get("question") or row.get("input") or ""
    a = row.get("answer") or row.get("output") or ""
    if not q or not a:
        return None
    return {
        "system":    "You are an expert in board games. Answer the question accurately.",
        "user":      q,
        "assistant": a,
    }


def _map_generic(row: dict) -> dict | None:
    """Generic fallback — coba semua field umum."""
    # Prioritas: messages (chat format) → instruction/output → prompt/completion → question/answer
    if "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        asst_msg = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        sys_msg  = next((m.get("content", "") for m in msgs if m.get("role") == "system"),
                        "You are a helpful strategic game-playing assistant.")
        if user_msg and asst_msg:
            return {"system": sys_msg, "user": user_msg, "assistant": asst_msg}

    pairs = [
        ("instruction", "output"),
        ("prompt",       "completion"),
        ("question",     "answer"),
        ("input",        "output"),
        ("observation",  "action"),
    ]
    for user_key, asst_key in pairs:
        u = row.get(user_key, "")
        a = row.get(asst_key, "")
        if u and a:
            return {
                "system":    "You are a helpful strategic game-playing assistant.",
                "user":      u,
                "assistant": a,
            }
    return None


GAME_MAPPERS = {
    "gin_rummy":    _map_gin_rummy,
    "poker":        _map_poker,
    "textarena":    _map_textarena,
    "boardgame_qa": _map_boardgame_qa,
    "generic":      _map_generic,
    "env_generic":  _map_generic,
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading & tokenization
# ──────────────────────────────────────────────────────────────────────────────

def _load_hf_dataset(dataset_id: str, split: str, max_samples: int):
    """Load dataset dari HuggingFace. Raise DatasetNotAvailableError jika gagal."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetNotAvailableError(f"Package 'datasets' tidak tersedia: {e}") from e

    print(f"[SFT] Loading dataset: {dataset_id} (split={split})", flush=True)
    try:
        ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception as exc:
        raise DatasetNotAvailableError(
            f"Gagal load dataset {dataset_id!r}: {exc}"
        ) from exc

    if ds is None or len(ds) == 0:
        raise DatasetNotAvailableError(f"Dataset {dataset_id!r} kosong (0 rows).")

    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    print(f"[SFT] Dataset loaded: {len(ds)} rows.", flush=True)
    return ds


def _build_text_field(row: dict, game: str, tokenizer) -> str | None:
    """Konversi satu row ke string teks siap tokenize (dengan chat template)."""
    mapper = GAME_MAPPERS.get(game, _map_generic)
    mapped = mapper(row)
    if mapped is None:
        return None

    messages = [
        {"role": "system",    "content": mapped["system"]},
        {"role": "user",      "content": mapped["user"]},
        {"role": "assistant", "content": mapped["assistant"]},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text
    except Exception:
        # Fallback: manual format kalau model tidak punya chat template
        text = (
            f"<|system|>{mapped['system']}<|end|>\n"
            f"<|user|>{mapped['user']}<|end|>\n"
            f"<|assistant|>{mapped['assistant']}<|end|>"
        )
        return text


def prepare_sft_dataset(dataset_id: str, game: str, tokenizer, split: str, max_samples: int):
    """Load + map + tokenize dataset, return HuggingFace Dataset dengan field 'text'."""
    from datasets import Dataset as HFDataset

    raw_ds = _load_hf_dataset(dataset_id, split, max_samples)

    texts = []
    skipped = 0
    for row in raw_ds:
        text = _build_text_field(dict(row), game, tokenizer)
        if text:
            texts.append({"text": text})
        else:
            skipped += 1

    print(f"[SFT] Mapped: {len(texts)} valid, {skipped} skipped.", flush=True)

    if len(texts) == 0:
        raise DatasetNotAvailableError(
            f"Semua {len(raw_ds)} row dari {dataset_id!r} gagal di-map (field tidak dikenali)."
        )

    # Split train/dev (90/10)
    n_dev   = min(200, max(10, len(texts) // 10))
    dev_ds  = HFDataset.from_list(texts[:n_dev])
    train_ds = HFDataset.from_list(texts[n_dev:])
    print(f"[SFT] train={len(train_ds)}, dev={len(dev_ds)}", flush=True)
    return train_ds, dev_ds


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_model(model_path: str, use_lora: bool, disable_fa: bool):
    attn_impl  = "eager" if disable_fa else "flash_attention_2"
    local      = _is_local_path(model_path)
    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        local_files_only=local,
    )
    base.config.use_cache = False

    if use_lora:
        from peft import get_peft_model
        lora_cfg = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
        return model

    return base


# ──────────────────────────────────────────────────────────────────────────────
# Entry point (dipanggil oleh text_trainer.py via subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Run SFT warm-start training."""
    argument_parser = transformers.HfArgumentParser((SftEnvArguments,))
    (args,) = argument_parser.parse_args_into_dataclasses()

    # ── Load train_request ──
    if not args.request_path or not os.path.exists(args.request_path):
        raise FileNotFoundError(f"request_path tidak ditemukan: {args.request_path!r}")

    with open(args.request_path) as f:
        train_info_full = json.load(f)
    train_request = train_info_full.get("train_request", train_info_full)

    model_path     = _resolve_model_path(train_request["model_path"])
    dataset_id     = args.sft_dataset_id or train_request.get("sft_dataset_id", "")
    game           = train_request.get("sft_game", "generic")

    if not dataset_id:
        raise DatasetNotAvailableError("Tidak ada sft_dataset_id di args maupun train_request.")

    # ── Tokenizer ──
    _local = _is_local_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[SFT] Tokenizer: {model_path} (local={_local}), pad={tokenizer.pad_token!r}", flush=True)

    # ── Dataset (akan raise DatasetNotAvailableError jika gagal) ──
    train_ds, dev_ds = prepare_sft_dataset(
        dataset_id=dataset_id,
        game=game,
        tokenizer=tokenizer,
        split=args.sft_dataset_split,
        max_samples=args.max_sft_samples,
    )

    # ── Model ──
    model = _load_model(model_path, args.use_lora, args.disable_fa)

    # ── SFTTrainer (TRL) ──
    try:
        from trl import SFTTrainer, SFTConfig
        print("[SFT] Using TRL SFTTrainer.", flush=True)

        sft_config = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=20,
            weight_decay=0.01,
            bf16=True,
            tf32=True,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            gradient_checkpointing=args.gradient_checkpointing,
            optim=args.optim if args.optim else "adamw_torch",
            report_to=args.report_to if args.report_to else "none",
            dataset_text_field="text",
            max_seq_length=2048,
            packing=True,         # SFTTrainer mendukung packing langsung
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
        )

    except ImportError:
        # Fallback ke bare HF Trainer kalau TRL belum terinstall
        print("[SFT] TRL tidak tersedia, pakai HF Trainer fallback.", flush=True)
        from transformers import Trainer
        from torch.nn.utils.rnn import pad_sequence
        import functools

        def _collate(batch, pad_id):
            from transformers import DataCollatorForSeq2Seq
            # Tokenize on-the-fly
            enc_list = [tokenizer(b["text"], truncation=True, max_length=2048) for b in batch]
            ids = pad_sequence([torch.tensor(e["input_ids"]) for e in enc_list], True, pad_id)
            mask = pad_sequence([torch.tensor(e["attention_mask"]) for e in enc_list], True, 0)
            return {"input_ids": ids, "attention_mask": mask, "labels": ids.clone()}

        collate_fn = functools.partial(_collate, pad_id=tokenizer.pad_token_id)

        if is_main_process(LOCAL_RANK):
            os.makedirs(args.output_dir, exist_ok=True)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=collate_fn,
        )

    # ── Train ──
    print(f"[SFT] Starting SFT warm-start: dataset={dataset_id}, game={game}", flush=True)
    trainer.train()

    # ── Simpan checkpoint ──
    if is_main_process(LOCAL_RANK):
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        success_file = os.path.join(args.output_dir, "sft_success.txt")
        with open(success_file, "w") as f:
            f.write(f"SFT warm-start completed. dataset={dataset_id}, game={game}")
        print(f"[SFT] Checkpoint saved to: {args.output_dir}", flush=True)

    print("[SFT] Warm-start training selesai.", flush=True)


if __name__ == "__main__":
    main()
