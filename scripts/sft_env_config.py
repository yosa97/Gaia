"""
sft_env_config.py — Config builder untuk SFT warm-start environment tasks.

Dipanggil oleh text_trainer.py sebelum GRPO.
Output: run_cmd string untuk train_sft_env.py + path checkpoint untuk GRPO.
"""

from copy import deepcopy
from whitelisted_sft_datasets import validate_requested_datasets, get_game_for_dataset
from model_utility import (
    get_model_architecture,
    get_model_num_params,
    disable_flash_attention,
    get_gradient_checkpointing,
    get_gpu_count,
    resolve_model_path,
)

# ── Config per ukuran model ────────────────────────────────────────────────────
# Batch size kecil karena SFT dataset env kecil (few small datasets).
# SFT warm-start tidak perlu banyak step — hanya butuh model bisa format game.
SFT_ENV_CONFIG = {
    "0_2_b": {
        "lr": 5e-5, "distributed": "ddp", "gpu_count": 1,
        "batch_size": 4, "grad_accum": 4, "use_lora": False, "epochs": 2,
    },
    "2_5_b": {
        "lr": 2e-5, "distributed": "ddp", "gpu_count": 1,
        "batch_size": 2, "grad_accum": 8, "use_lora": True,  "epochs": 2,
    },
    "5_9_b": {
        "lr": 1e-5, "distributed": "ddp", "gpu_count": 2,
        "batch_size": 2, "grad_accum": 8, "use_lora": True,  "epochs": 2,
    },
    "9_b_plus": {
        "lr": 8e-6, "distributed": "ddp", "gpu_count": 4,
        "batch_size": 1, "grad_accum": 4, "use_lora": True,  "epochs": 1,
    },
}


def _pick_config(param_nums: int) -> dict:
    if param_nums < 2_000_000_000:
        return SFT_ENV_CONFIG["0_2_b"]
    elif param_nums < 5_000_000_000:
        return SFT_ENV_CONFIG["2_5_b"]
    elif param_nums < 9_000_000_000:
        return SFT_ENV_CONFIG["5_9_b"]
    else:
        return SFT_ENV_CONFIG["9_b_plus"]


def get_sft_output_dir(train_info: dict) -> str:
    """Path checkpoint SFT yang akan di-pass ke GRPO sebagai starting model."""
    output_dir = train_info.get("output_dir", "/workspace/scripts/soutputs")
    task_id    = train_info.get("task_id", "unknown")
    return f"{output_dir}_sft_{task_id}"


def get_run_cmd(config: dict, gpu_nums: int) -> str:
    """Build run command untuk train_sft_env.py."""
    gpu_nums = get_gpu_count()
    run_type  = config.get("distributed", "ddp")

    if gpu_nums > 1 and run_type == "ddp":
        # --error_file agar traceback dari setiap rank terlihat jelas
        start_cmd = f"torchrun --nproc_per_node={gpu_nums} --error_file /tmp/sft_error.txt"
    else:
        start_cmd = "python"

    template = (
        start_cmd
        + """ train_sft_env.py \\
    --request_path {request_path} \\
    --sft_dataset_id {sft_dataset_id} \\
    --sft_dataset_split train \\
    --max_sft_samples {max_sft_samples} \\
    --bf16 True \\
    --output_dir {sft_output_dir} \\
    --num_train_epochs {epochs} \\
    --per_device_train_batch_size {batch_size} \\
    --gradient_accumulation_steps {grad_accum} \\
    --learning_rate {learning_rate} \\
    --weight_decay 0.01 \\
    --warmup_steps 20 \\
    --lr_scheduler_type cosine \\
    --tf32 True \\
    --gradient_checkpointing {gradient_checkpointing} \\
    --optim adamw_torch \\
    --report_to none \\
    --disable_fa {disable_fa} \\
    --use_lora {use_lora}"""
    )

    for key, value in config.items():
        template = template.replace("{" + key + "}", str(value))

    return template


def get_training_json(train_info: dict, requested_datasets: list[str]) -> dict | None:
    """Build SFT config.

    Returns:
        dict dengan 'run_cmd' dan 'sft_output_dir', atau None jika tidak ada dataset valid.
    """
    # Validasi whitelist
    valid_datasets = validate_requested_datasets(requested_datasets)
    if not valid_datasets:
        print("[sft_env_config] Tidak ada dataset valid di whitelist. Skip SFT.", flush=True)
        return None

    # Ambil dataset pertama yang valid
    dataset_id = valid_datasets[0]
    game       = get_game_for_dataset(dataset_id)

    model_name         = train_info["model_name"]
    model_path         = train_info["model_path"]
    model_path         = resolve_model_path(model_path, model_name)
    model_arch         = get_model_architecture(model_path)
    param_nums         = get_model_num_params(model_name, model_path)
    config             = deepcopy(_pick_config(param_nums))
    sft_output_dir     = get_sft_output_dir(train_info)

    run_config = {
        "request_path":         train_info["request_path"],
        "sft_dataset_id":       dataset_id,
        "sft_output_dir":       sft_output_dir,
        "max_sft_samples":      5000,           # cukup untuk warm-start
        "epochs":               config["epochs"],
        "batch_size":           config["batch_size"],
        "grad_accum":           config["grad_accum"],
        "learning_rate":        config["lr"] * train_info.get("reg_ratio", 1.0),
        "use_lora":             config.get("use_lora", True),
        "disable_fa":           disable_flash_attention(model_arch, model_name),
        "gradient_checkpointing": get_gradient_checkpointing(model_name),
        "distributed":          config.get("distributed", "ddp"),
        "gpu_nums":             config["gpu_count"],
    }

    train_request = dict(train_info)
    train_request["sft_dataset_id"] = dataset_id
    train_request["sft_game"]       = game
    train_request["sft_output_dir"] = sft_output_dir

    run_cmd = get_run_cmd(run_config, run_config["gpu_nums"])
    print(f"[sft_env_config] Dataset={dataset_id}, Game={game}, SFT dir={sft_output_dir}", flush=True)

    return {
        "train_request":  train_request,
        "run_cmd":        run_cmd,
        "sft_output_dir": sft_output_dir,
        "dataset_id":     dataset_id,
        "game":           game,
    }
