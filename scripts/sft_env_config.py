"""Full SFT training config — pure SFT mode (no GRPO).

Builds the torchrun command for `train_full_sft.py`. Mirrors the structure of
`grpo_env_config.py` but skips GRPO/vLLM machinery entirely. Triggered via
SFT_ONLY=1 env var routed through `scripts/text_trainer.py`.

Two-phase training (Plan B per project_full_sft_plan_b_escalation.md):
  - Phase 1: Boardgame-QA 500-sample warm-up (LR 5e-6, 1 epoch, gentle)
  - Phase 2: env_training single-game (LD/LP/GR) specialize (per-bucket lr/epoch)
  Same LoRA adapter accumulates; Phase 2 dominates final direction.

  Per-bucket schedule (domain-aware: bounded action prediction is closer to
  PokerBench arxiv 2501.08328 than to instruction tuning, so we DO NOT lift
  LR to LoRA-generic 1e-4–2e-4):
    - 0_1_b / 1_2_b / 2_4_b: LR 1e-5, epoch 10  (chess paper recipe analog)
    - 4_5_b: LR 1e-5, epoch 8  (mild taper)
    - 5_6_b: LR 8e-6, epoch 7
    - 6_9_b: LR 5e-6, epoch 6  (chess paper 8B uses 5e-6; PokerBench shows
      higher LR destabilizes 7B+ on game-action SFT)
  Biderman et al. "LoRA Learns Less and Forgets Less" supports narrower stable
  LR band for larger LoRA models even when generic recipes are flat.

Critical config (per liars_dice SFT research):
  - assistant_only_loss=True (mask user tokens, train on assistant only)
  - neftune_noise_alpha=5 (regularizer for tiny single-game dataset)
  - max_length=2048 (game traces are short)
"""

from copy import deepcopy

from model_utility import (
    get_gpu_count,
    get_model_architecture,
    get_model_num_params,
    get_use_liger,
)


SFT_CONFIG = {
    "0_1_b": {
        "lr": 1e-5,
        "epoch_num": 10,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "use_lora": True,
    },
    "1_2_b": {
        "lr": 1e-5,
        "epoch_num": 10,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "use_lora": True,
    },
    "2_4_b": {
        "lr": 1e-5,
        "epoch_num": 10,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "use_lora": True,
    },
    "4_5_b": {
        "lr": 1e-5,
        "epoch_num": 8,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_lora": True,
    },
    "5_6_b": {
        "lr": 8e-6,
        "epoch_num": 7,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_lora": True,
    },
    "6_9_b": {
        "lr": 5e-6,
        "epoch_num": 6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_lora": True,
    },
}


# Per-game epoch overrides on top of bucket defaults. Used when a specific game
# has different optimization needs than the bucket default suggests.
#
# leduc_poker: epoch=10 produced eval 0.5388 with HEALTHY training profile
# (entropy 0.76 stable, grad_norm 2.02 stable, no overfit signal). Loss still
# decreasing at epoch 10 (0.085 → 0.080 → 0.068). Hypothesis: LP underperform
# vs LD (0.6301) is state-space coverage limit, not capacity limit. Conservative
# LIMA-style bump to 12 only for small buckets that already have epoch=10
# baseline; large buckets keep their lower epoch counts to avoid overfit.
PER_GAME_EPOCH_OVERRIDE = {
    "leduc_poker": {
        "0_1_b": 12,
        "1_2_b": 12,
        "2_4_b": 12,
    },
    # gin_rummy: more epochs needed to converge on meld/knock decision boundaries.
    # LIMA-style bump — only safe for small buckets where overfit risk is low.
    "gin_rummy": {
        "0_1_b": 12,
        "1_2_b": 12,
        "2_4_b": 12,
    },
}


def _select_size_bucket(num_params, model_name: str = "") -> str:
    """Pick the matching size bucket key from SFT_CONFIG.

    Bucket thresholds are in **billions** of parameters. `num_params` may
    arrive in two forms because of how `get_model_num_params` works:
      - Raw count (e.g. 3_085_938_688 for Qwen2.5-3B from safetensors or
        from the model_id regex `int(...) * 1_000_000_000`).
      - Billions (e.g. 3.086) when the inner regex fallback below produces it.
    Normalize anything ≥ 1_000_000 to billions so thresholds compare correctly.
    Without this normalization, a Qwen2.5-3B (3.086e9 raw count) compares
    > 6.0 and falls through to "6_9_b" — a silent misrouting that flat configs
    used to mask but per-bucket lr/epoch now exposes.
    """
    if num_params is None:
        import re
        match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", model_name)
        if match:
            num_params = float(match.group(1))
        else:
            print(f"[full_sft_config] num_params and model_name regex both failed for {model_name!r}; defaulting to 2_4_b bucket")
            return "2_4_b"
    # Normalize raw param count → billions.
    if num_params >= 1_000_000:
        num_params = num_params / 1_000_000_000.0
    if num_params < 1.0:
        return "0_1_b"
    if num_params < 2.0:
        return "1_2_b"
    if num_params < 4.0:
        return "2_4_b"
    if num_params < 5.0:
        return "4_5_b"
    if num_params < 6.0:
        return "5_6_b"
    return "6_9_b"


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    required_keys = [
        "epoch_num",
        "batch_size",
        "learning_rate",
        "min_lr_rate",
        "use_liger",
        "optimizer",
        "environment_name",
        "max_seq_len",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required key {key} not found in config")

    gpu_nums = get_gpu_count()
    start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    if config.get("distributed") == "ds":
        start_cmd = "deepspeed"

    template = (
        start_cmd
        + """ train_sft_env.py \
    --request_path {request_path} \
    --environment_name {environment_name} \
    --bf16 True \
    --report_to wandb \
    --output_dir /workspace/data/trained_model \
    --num_train_epochs {epoch_num} \
    --per_device_train_batch_size {batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 10 \
    --learning_rate {learning_rate} \
    --weight_decay {weight_decay} \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs "{\\"min_lr_rate\\": {min_lr_rate}}" \
    --tf32 True \
    --gradient_checkpointing True \
    --optim {optimizer} \
    --max_length {max_seq_len} \
    --assistant_only_loss True \
    --neftune_noise_alpha 5 \
    --use_liger {use_liger}"""
    )

    if config.get("use_lora", False):
        # LoRA R=128 for PvP tournament — larger adapter capacity needed to model
        # diverse opponent styles. Alpha=256 keeps alpha/r ratio=2 (standard).
        template += " --use_peft --lora_r 128 --lora_alpha 256 --lora_target_modules all-linear"

    if config.get("distributed") == "ds":
        template += " --deepspeed ds_config/zero3.json"

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    print(f"template: {template}", flush=True)
    return template


def get_training_json(train_info: dict) -> dict:
    """Build the train_request + run_cmd for full SFT mode.

    Mirrors get_training_json signature in grpo_env_config to drop into
    text_trainer.py dispatch in place of env training when SFT_ONLY=1.
    """
    model_name = train_info["model_name"]
    model_path = train_info.get("model_path", model_name)
    architecture = get_model_architecture(model_path)
    num_params = get_model_num_params(model_name, model_path)
    bucket = _select_size_bucket(num_params, model_name=model_name)

    base_config = deepcopy(SFT_CONFIG[bucket])
    base_config["gpu_nums"] = base_config["gpu_count"]
    # Disable liger for SFTTrainer: liger fused kernel returns logits=None,
    # SFTTrainer.compute_loss needs logits for entropy_from_logits → AttributeError.
    base_config["use_liger"] = False
    base_config["optimizer"] = "paged_adamw_8bit"
    base_config["learning_rate"] = base_config["lr"]
    base_config["min_lr_rate"] = 0.1
    base_config["max_seq_len"] = 2048
    # PvP: slightly higher weight_decay (0.05 vs 0.01) to regularize against
    # opponent-style overfitting — model must generalize, not memorize one style.
    base_config["weight_decay"] = 0.05

    dataset_type = train_info.get("dataset_type") or {}
    base_config["environment_name"] = dataset_type.get("environment_name", "liars_dice")

    # epoch_num is per-bucket by default (smaller models tolerate more epochs
    # on tiny data; larger models overfit faster). Small buckets (≤4B) keep
    # 10 epoch per Strategic Reasoning Chess paper (arxiv 2507.00726). Larger
    # buckets taper: 4_5_b=8, 5_6_b=7, 6_9_b=6 per Biderman et al. and
    # PokerBench (arxiv 2501.08328) which found game-action SFT destabilizes
    # at higher LR / more epochs for 7B+ models.
    #
    # Per-game override (PER_GAME_EPOCH_OVERRIDE) applies on top — currently
    # bumps leduc_poker to 12 epochs on small buckets based on observed
    # healthy training profile + state-space coverage hypothesis.
    env_name = base_config["environment_name"]
    override = PER_GAME_EPOCH_OVERRIDE.get(env_name, {}).get(bucket)
    if override is not None:
        print(
            f"[full_sft_config] per-game epoch override for env={env_name} "
            f"bucket={bucket}: {base_config['epoch_num']} -> {override}",
            flush=True,
        )
        base_config["epoch_num"] = override

    base_config["request_path"] = train_info["request_path"]

    train_request = deepcopy(train_info)
    train_request["lora_r"] = 64 if base_config.get("use_lora") else None
    train_request["effective_batch_size"] = (
        base_config["batch_size"]
        * base_config["gradient_accumulation_steps"]
        * base_config["gpu_count"]
    )
    train_request["budget_min"] = 165
    train_request["sft_mode"] = "full"

    run_cmd = get_run_cmd(base_config, base_config["gpu_count"])
    return {"train_request": train_request, "run_cmd": run_cmd}