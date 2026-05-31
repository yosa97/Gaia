import argparse
import asyncio
import os
import shutil
import tempfile
import json
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

import trainer.utils.training_paths as train_paths
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from trainer import constants as cst


hf_api = HfApi()


def download_sft_datasets(datasets_csv: str, datasets_dir: str) -> None:
    """Download whitelisted SFT datasets from HuggingFace Hub.

    Reads a comma-separated list of HF dataset repo IDs from ``datasets_csv``
    and saves each one to ``{datasets_dir}/{repo_id}/`` using save_to_disk.

    Path convention matches ``_load_sft_datasets`` in train_grpo_env.py which
    does ``os.path.join(datasets_dir, name)`` — so the folder is the full
    repo_id including the slash, e.g.
    ``/cache/miner_datasets/gradients-io-tournaments/env_training_gradients``.

    Args:
        datasets_csv: Comma-separated HF dataset repo IDs, e.g.
                      \"gradients-io-tournaments/env_training_gradients\".
        datasets_dir: Local directory to download datasets into.
    """
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError:
        print("[SFT-Downloader] 'datasets' package not found, skipping SFT download.", flush=True)
        return

    dataset_list = [d.strip() for d in datasets_csv.split(",") if d.strip()]
    if not dataset_list:
        print("[SFT-Downloader] No MINER_DATASETS specified, skipping SFT dataset download.", flush=True)
        return

    print(f"[SFT-Downloader] Downloading {len(dataset_list)} SFT dataset(s) to: {datasets_dir}", flush=True)

    for repo_id in dataset_list:
        # Path must match train_grpo_env.py: os.path.join(datasets_dir, name)
        # e.g. /cache/miner_datasets/gradients-io-tournaments/env_training_gradients
        local_dir = os.path.join(datasets_dir, repo_id)
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"[SFT-Downloader] Dataset already cached, skipping: {repo_id}", flush=True)
            continue
        try:
            print(f"[SFT-Downloader] Downloading dataset: {repo_id}", flush=True)
            os.makedirs(local_dir, exist_ok=True)
            ds = _load_dataset(repo_id, split="train", trust_remote_code=True)
            ds.save_to_disk(local_dir)
            print(f"[SFT-Downloader] ✅ Saved {len(ds)} examples → {local_dir}", flush=True)
        except Exception as exc:
            # Non-fatal: log and continue — SFT will be skipped gracefully if folder is empty
            print(f"[SFT-Downloader] ⚠️ Failed to download {repo_id}: {exc}", flush=True)


async def download_text_dataset(task_id, dataset_url, file_format, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

    if file_format == FileFormat.S3.value:
        input_data_path = train_paths.get_text_dataset_path(task_id)

        if not os.path.exists(input_data_path):
            local_path = await download_s3_file(dataset_url)
            shutil.copy(local_path, input_data_path)

    elif file_format == FileFormat.HF.value:
        repo_name = dataset_url.replace("/", "--")
        input_data_path = os.path.join(dataset_dir, repo_name)

        if not os.path.exists(input_data_path):
            snapshot_download(repo_id=dataset_url, repo_type="dataset", local_dir=input_data_path, local_dir_use_symlinks=False)

    return input_data_path, file_format


def write_environment_task_proxy_dataset(
    out_path: str,
    dataset_size: int = 1000,
    prompt_text: str = "Interact with this environment.",
    prompt_field: str = "prompt",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [{prompt_field: prompt_text} for _ in range(dataset_size)]

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {out_path} with field '{prompt_field}'")


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    try:
        local_dir = os.path.expanduser(local_dir)
        local_filename = f"{repo_id.replace('/', '_')}.safetensors"
        final_path = os.path.join(local_dir, local_filename)
        os.makedirs(local_dir, exist_ok=True)
        if os.path.exists(final_path):
            print(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            print(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        raise e


async def download_axolotl_base_model(repo_id: str, save_dir: str) -> str:
    model_dir = os.path.join(save_dir, repo_id.replace("/", "--"))
    if os.path.exists(model_dir):
        print(f"Model {repo_id} already exists at {model_dir}. Skipping download.")
        return model_dir
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=model_dir, local_dir_use_symlinks=False)
    return model_dir


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--task-type",
        required=True,
        choices=[TaskType.INSTRUCTTEXTTASK.value, TaskType.DPOTASK.value, TaskType.GRPOTASK.value, TaskType.CHATTASK.value, TaskType.ENVIRONMENTTASK.value],
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--file-format")
    
    # Optional args that might technically be passed but ignored for text tasks
    # (Leaving out model-type completely as it causes issues if enum values are missing)
    
    args, unknown = parser.parse_known_args()

    dataset_dir = cst.CACHE_DATASETS_DIR
    model_dir = cst.CACHE_MODELS_DIR
    adapters_dir = cst.HUGGINGFACE_CACHE_PATH
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(adapters_dir, exist_ok=True)

    print(f"Downloading datasets to: {dataset_dir}", flush=True)
    print(f"Downloading models to: {model_dir}", flush=True)

    if args.task_type == TaskType.ENVIRONMENTTASK.value:
        model_path = await download_axolotl_base_model(args.model, model_dir)
        input_data_path = train_paths.get_text_dataset_path(args.task_id)
        write_environment_task_proxy_dataset(
            out_path=input_data_path,
            dataset_size=1000,
            prompt_text="Interact with this environment.",
            prompt_field="prompt",
        )

        # ── Download SFT cold-start datasets ──────────────────────────────
        # MINER_DATASETS and MINER_DATASETS_DIR are injected by run_environment.sh.
        # If not set, this step is silently skipped and SFT won't run.
        miner_datasets_csv = os.environ.get("MINER_DATASETS", "")
        miner_datasets_dir = os.environ.get("MINER_DATASETS_DIR", "")
        if miner_datasets_csv and miner_datasets_dir:
            download_sft_datasets(miner_datasets_csv, miner_datasets_dir)
        else:
            print("[SFT-Downloader] MINER_DATASETS / MINER_DATASETS_DIR not set — skipping SFT dataset download.", flush=True)
    else:
        dataset_path, _ = await download_text_dataset(args.task_id, args.dataset, args.file_format, dataset_dir)
        model_path = await download_axolotl_base_model(args.model, model_dir)

    print(f"Model path: {model_path}", flush=True)
    print(f"Dataset path: {dataset_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
