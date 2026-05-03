"""
sft_env_config.py — Config builder untuk SFT environment tasks.

Mirip dengan instruct_config.py tapi:
- Dipakai khusus untuk EnvTask dengan SFT dataset
- Command menjalankan train_sft_env.py (bukan train_instruct.py)
- Pakai LoRA lebih kecil karena dataset kecil (sesuai info G.O.D)
"""

from copy import deepcopy
from model_utility import (
    get_model_architecture,
    get_model_num_params,
    disable_flash_attention,
    get_gradient_checkpointing,
    get_gpu_count,
)

# ── Config per ukuran model ────────────────────────────────────────────────
# Batch size kecil karena dataset SFT env kecil (few small SFT datasets).
SFT_ENV_CONFIG = {
    "0_1_b": {
        "lr": 5e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "use_lora": False,
        "epoch_num": 3,
    },
    "1_2_b": {
        "lr": 3e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "use_lora": False,
        "epoch_num": 3,
    },
    "2_4_b": {
        "lr": 2e-5,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_lora": True,
        "epoch_num": 3,
    },
    "4_5_b": {
        "lr": 1e-5,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_lora": True,
        "epoch_num": 3,
    },
    "5_9_b": {
        "lr": 8e-6,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "use_lora": True,
        "epoch_num": 3,
    },
    "9_12_b": {
        "lr": 6e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "use_lora": True,
        "epoch_num": 2,
    },
    "12_40_b": {
        "lr": 5e-6,
        "distributed": "ddp",
        "gpu_count": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "use_lora": True,
        "epoch_num": 2,
    },
}

for key in SFT_ENV_CONFIG:
    SFT_ENV_CONFIG[key]["label"] = key


def _get_sft_env_config(param_nums: int) -> dict:
    if param_nums < 1_000_000_000:
        return SFT_ENV_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return SFT_ENV_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return SFT_ENV_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return SFT_ENV_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        return SFT_ENV_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        return SFT_ENV_CONFIG["9_12_b"]
    else:
        return SFT_ENV_CONFIG["12_40_b"]


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    """Build the torchrun command string untuk train_sft_env.py."""
    gpu_nums = get_gpu_count()
    run_type  = config.get("distributed", "ddp")

    if gpu_nums > 1 and run_type == "ddp":
        start_cmd = f"torchrun --nproc_per_node={gpu_nums}"
    elif run_type == "ds":
        start_cmd = "deepspeed"
    else:
        start_cmd = "python"

    template = (
        start_cmd
        + """ train_sft_env.py \\
    --request_path {request_path} \\
    --bf16 True \\
    --report_to wandb \\
    --output_dir {output_dir} \\
    --num_train_epochs {epoch_num} \\
    --per_device_train_batch_size {batch_size} \\
    --per_device_eval_batch_size 1 \\
    --gradient_accumulation_steps {gradient_accumulation_steps} \\
    --eval_accumulation_steps 1 \\
    --eval_strategy no \\
    --save_strategy no \\
    --logging_steps 5 \\
    --learning_rate {learning_rate} \\
    --weight_decay 0.01 \\
    --warmup_steps 20 \\
    --lr_scheduler_type cosine \\
    --tf32 True \\
    --gradient_checkpointing {gradient_checkpointing} \\
    --optim paged_adamw_8bit \\
    --disable_fa {disable_fa} \\
    --use_lora {use_lora}"""
    )

    if run_type == "ds":
        template += " --deepspeed ds_config/zero3.json"

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    return template


def get_training_json(train_info: dict) -> dict:
    """Build full SFT env training config + command string."""
    model_name        = train_info["model_name"]
    model_path        = train_info["model_path"]
    model_architecture = get_model_architecture(model_path)
    param_nums        = get_model_num_params(model_name, model_path)
    config            = deepcopy(_get_sft_env_config(param_nums))

    run_config = {
        "epoch_num":                   config["epoch_num"],
        "batch_size":                  config["batch_size"],
        "learning_rate":               config["lr"] * train_info.get("reg_ratio", 1.0),
        "use_lora":                    config.get("use_lora", True),
        "disable_fa":                  disable_flash_attention(model_architecture, model_name),
        "gradient_checkpointing":      get_gradient_checkpointing(model_name),
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "distributed":                 config.get("distributed", "ddp"),
        "gpu_nums":                    config["gpu_count"],
        "output_dir":                  train_info["output_dir"],
        "request_path":                train_info["request_path"],
    }

    train_request = dict(train_info)
    train_request["save_before_remaining_time"] = 10
    train_request["min_steps"]                  = 50
    train_request["adjust_batch_size"]          = False
    train_request["periodic_save_steps"]        = 50

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])

    max_steps = train_info.get("max_steps", -1)
    if max_steps and max_steps > 0:
        run_cmd += f" --max_steps {max_steps}"
        print(f"[sft_env_config] max_steps={max_steps} → ditambahkan ke SFT command")

    print(f"[sft_env_config] SFT run_cmd: {run_cmd}", flush=True)
    return {"train_request": train_request, "run_cmd": run_cmd}
