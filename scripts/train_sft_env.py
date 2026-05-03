"""
train_sft_env.py — SFT trainer for environment tasks (G.O.D tournament).

Flow:
  1. Baca train_request dari request_path JSON.
  2. Download & parse dataset dari URL yang disediakan G.O.D server.
  3. Tokenize dataset dengan format chat (system + user + assistant).
  4. Train dengan HuggingFace Trainer (LoRA kalau model besar, full sinon).
  5. Tulis success.txt ke output_dir kalau berhasil.

Jika dataset tidak tersedia / kosong → raise DatasetNotAvailableError
agar text_trainer.py bisa fallback ke GRPO.
"""

from __future__ import annotations

import json
import os
import sys
import requests
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import is_main_process
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utility import log_info
from customized_trainer import (
    resize_if_needed,
    set_generation_config,
    CustomEvalSaveCallback,
    WhenToEvalHandler,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

# ──────────────────────────────────────────────────────────────
# Custom exception — ditangkap oleh text_trainer.py untuk fallback
# ──────────────────────────────────────────────────────────────
class DatasetNotAvailableError(RuntimeError):
    """Raised when the SFT dataset cannot be fetched or is empty."""


# ──────────────────────────────────────────────────────────────
# CLI arguments (diparsing oleh text_trainer.py → train_sft_env)
# ──────────────────────────────────────────────────────────────
@dataclass
class SftEnvTrainingArguments(transformers.TrainingArguments):
    request_path: Optional[str] = field(default=None)
    use_lora:     Optional[bool] = field(default=True)
    disable_fa:   Optional[bool] = field(default=False)


# ──────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────
_SFT_FIELDS = ("instruction", "input", "output")   # format G.O.D
_MAX_LENGTH  = 2048


def _fetch_dataset(dataset_url: str) -> list[dict]:
    """Download dataset JSON dari URL. Return list of dicts.

    Raises DatasetNotAvailableError jika URL tidak bisa diakses atau kosong.
    """
    if not dataset_url or dataset_url == "dummy":
        raise DatasetNotAvailableError("Dataset URL is empty or 'dummy' — tidak ada SFT dataset.")

    # Kalau sudah berupa path lokal (sudah didownload oleh trainer_downloader)
    if os.path.exists(dataset_url):
        log_info(f"[SFT] Loading dataset from local path: {dataset_url}")
        with open(dataset_url, "r") as f:
            data = json.load(f)
    else:
        log_info(f"[SFT] Downloading dataset from: {dataset_url}")
        try:
            resp = requests.get(dataset_url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise DatasetNotAvailableError(
                f"Gagal download SFT dataset dari {dataset_url!r}: {exc}"
            ) from exc

    if not data:
        raise DatasetNotAvailableError("Dataset berhasil didownload tapi kosong (0 rows).")

    log_info(f"[SFT] Dataset loaded: {len(data)} rows.")
    return data


def _format_sample(item: dict, tokenizer: AutoTokenizer) -> dict:
    """Konversi satu row dataset ke format chat message lalu tokenize."""
    instruction = item.get("instruction") or item.get("instruct") or ""
    inp         = item.get("input", "")
    output      = item.get("output", "")

    if inp:
        user_text = f"{instruction}\n\n{inp}"
    else:
        user_text = instruction

    messages = [
        {"role": "system",    "content": "You are a strategic game-playing assistant."},
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": output},
    ]

    # apply_chat_template mengikuti format model (Qwen, Llama, Mistral, dst.)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        text,
        max_length=_MAX_LENGTH,
        truncation=True,
        padding=False,
    )

    # Labels: mask prompt, hanya hitung loss pada assistant turn.
    # Cari posisi "assistant" reply di token ids.
    input_ids = encoded["input_ids"]
    labels    = [-100] * len(input_ids)

    # Encode hanya bagian assistant untuk menentukan offset awal reply
    assistant_encoded = tokenizer(output, add_special_tokens=False)
    assistant_ids     = assistant_encoded["input_ids"]
    n_assist          = len(assistant_ids)

    if n_assist > 0 and n_assist <= len(input_ids):
        # Geser label ke posisi akhir — assistant tokens berada di akhir sequence
        for j in range(n_assist):
            labels[len(input_ids) - n_assist + j] = input_ids[len(input_ids) - n_assist + j]

    encoded["labels"] = labels
    return encoded


class SftEnvDataset(Dataset):
    """Torch Dataset yang sudah di-tokenize untuk SFT."""

    def __init__(self, items: list[dict], tokenizer: AutoTokenizer):
        self.samples = []
        skipped = 0
        for item in items:
            # Lewati item dengan output kosong
            if not item.get("output") and not item.get("assistant"):
                skipped += 1
                continue
            try:
                enc = _format_sample(item, tokenizer)
                # Lewati jika semua label -100 (tidak ada loss)
                if all(l == -100 for l in enc["labels"]):
                    skipped += 1
                    continue
                self.samples.append(enc)
            except Exception:
                skipped += 1
        log_info(f"[SFT] Dataset setelah filtering: {len(self.samples)} valid, {skipped} skipped.")
        if len(self.samples) == 0:
            raise DatasetNotAvailableError("Semua sampel dataset tidak valid setelah filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


# ──────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────
def _load_model_for_sft(training_args: SftEnvTrainingArguments, model_path: str):
    """Load model — pakai LoRA untuk model ≥4B, full weight untuk kecil."""
    attn_impl = "eager" if training_args.disable_fa else "flash_attention_2"

    base = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    base.config.use_cache = False

    if training_args.use_lora:
        if training_args.gradient_checkpointing:
            base.enable_input_require_grads()

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


# ──────────────────────────────────────────────────────────────
# Data collator (padding dynamic batch)
# ──────────────────────────────────────────────────────────────
def _collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ──────────────────────────────────────────────────────────────
# Entry point dipanggil oleh text_trainer.py
# ──────────────────────────────────────────────────────────────
def main():
    """Dipanggil via: torchrun train_sft_env.py --request_path ... [huggingface TrainingArguments]"""
    argument_parser = transformers.HfArgumentParser((SftEnvTrainingArguments,))
    (training_args,) = argument_parser.parse_args_into_dataclasses()

    train_info_full = json.load(open(training_args.request_path))
    train_request   = train_info_full["train_request"]

    task_id    = train_request["task_id"]
    model_path = train_request["model_path"]
    dataset_url = train_request["dataset"]

    # ── 1. Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_info(f"[SFT] Tokenizer loaded. pad_token={tokenizer.pad_token!r}")

    # ── 2. Dataset — akan raise DatasetNotAvailableError jika tidak ada ──
    raw_data = _fetch_dataset(dataset_url)

    import random
    random.seed(42)
    random.shuffle(raw_data)
    dev_size  = min(200, max(10, len(raw_data) // 10))
    dev_raw   = raw_data[:dev_size]
    train_raw = raw_data[dev_size:]

    train_ds = SftEnvDataset(train_raw, tokenizer)
    dev_ds   = SftEnvDataset(dev_raw,   tokenizer)
    log_info(f"[SFT] train={len(train_ds)}, dev={len(dev_ds)}")

    # ── 3. Model ──
    set_generation_config(train_request["model_name"], None)
    model = _load_model_for_sft(training_args, model_path)

    # ── 4. Callback (save/eval sesuai waktu tournament) ──
    max_steps = train_request.get("max_steps", -1)
    total_steps_per_epoch = max(1, len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    ))

    import functools
    collate = functools.partial(_collate_fn, pad_token_id=tokenizer.pad_token_id)

    if is_main_process(LOCAL_RANK):
        os.makedirs(training_args.output_dir, exist_ok=True)

    training_args.save_only_model = True

    callback = CustomEvalSaveCallback(
        WhenToEvalHandler(
            train_request["end_time"],
            train_request.get("save_before_remaining_time", 10),
            periodic_save_steps=train_request.get("periodic_save_steps", 75),
            steps_per_epoch=total_steps_per_epoch,
            max_steps=max_steps,
        ),
        train_request["submission_dir"],
        training_args.output_dir,
        train_request["model_name"],
        max_steps,
    )

    # ── 5. Train ──
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate,
        callbacks=[callback],
    )

    log_info("[SFT] Starting SFT training...")
    trainer.train()

    if is_main_process(LOCAL_RANK):
        success_file = os.path.join(training_args.output_dir, "success.txt")
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("[SFT] Training successfully completed.")


if __name__ == "__main__":
    main()
