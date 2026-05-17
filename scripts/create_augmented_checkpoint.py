"""
Create your own augmented checkpoint for SN56 Bittensor Environment Tournament.

Applies one of three augmentation techniques to a base model and uploads
the result to your HuggingFace account.

Augmentation methods (same as validator's approved technique list):
  1. gaussian_noise   — Tambahkan noise kecil ke semua weight (paling cepat)
  2. weight_scaling   — Skala weight di layer tertentu
  3. magnitude_pruning — Set weight terkecil ke nol (sparsity ringan)

Usage:
    python3 scripts/create_augmented_checkpoint.py \
        --model_path NousResearch/Hermes-3-Llama-3.2-3B \
        --method gaussian_noise \
        --hf_username yosa722 \
        --hf_token hf_xxx... \
        --output_repo_name yosa-augmented-hermes3b
"""

import argparse
import hashlib
import os
import time

import torch
from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_gaussian_noise(model: AutoModelForCausalLM, noise_std: float = 0.005):
    """Tambahkan Gaussian noise kecil ke semua parameter model."""
    print(f"[augment] Applying gaussian_noise (std={noise_std}) to all layers...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.dtype in (torch.float32, torch.bfloat16, torch.float16):
                noise = torch.randn_like(param.float()) * noise_std
                param.add_(noise.to(param.dtype))
    print("[augment] gaussian_noise done.")


def apply_weight_scaling(model: AutoModelForCausalLM, scale_factor: float = 1.001):
    """Skala ringan pada weight di semua linear layer."""
    print(f"[augment] Applying weight_scaling (factor={scale_factor}) to linear layers...")
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                if "embed" not in name and "norm" not in name:
                    module.weight.data.mul_(scale_factor)
    print("[augment] weight_scaling done.")


def apply_magnitude_pruning(model: AutoModelForCausalLM, pruning_ratio: float = 0.001):
    """Set weight dengan magnitude terkecil ke nol (sparsity sangat ringan)."""
    print(f"[augment] Applying magnitude_pruning (ratio={pruning_ratio}) to linear layers...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                threshold = torch.quantile(param.abs().float(), pruning_ratio)
                mask = param.abs() > threshold.to(param.dtype)
                param.mul_(mask.to(param.dtype))
    print("[augment] magnitude_pruning done.")


def main():
    parser = argparse.ArgumentParser(description="Create augmented checkpoint for SN56")
    parser.add_argument("--model_path", type=str,
                        default="NousResearch/Hermes-3-Llama-3.2-3B",
                        help="Base model HuggingFace repo atau path lokal")
    parser.add_argument("--method", type=str,
                        choices=["gaussian_noise", "weight_scaling", "magnitude_pruning"],
                        default="gaussian_noise",
                        help="Metode augmentasi")
    parser.add_argument("--hf_username", type=str, required=True,
                        help="Username HuggingFace Anda")
    parser.add_argument("--hf_token", type=str, required=True,
                        help="Token HuggingFace Anda")
    parser.add_argument("--output_repo_name", type=str, default=None,
                        help="Nama repo output (default: auto-generated)")
    parser.add_argument("--save_dir", type=str, default="/tmp/augmented_model",
                        help="Direktori lokal sementara untuk menyimpan model")
    args = parser.parse_args()

    # Auto-generate repo name jika tidak diisi
    if args.output_repo_name is None:
        short_hash = hashlib.md5(
            f"{args.model_path}{args.method}{time.time()}".encode()
        ).hexdigest()[:8]
        args.output_repo_name = f"augmented-{short_hash}"

    full_repo_id = f"{args.hf_username}/{args.output_repo_name}"
    print(f"\n{'='*55}")
    print(f"[augment] Base model  : {args.model_path}")
    print(f"[augment] Method      : {args.method}")
    print(f"[augment] Output repo : {full_repo_id}")
    print(f"{'='*55}\n")

    # Load base model
    print(f"[augment] Loading base model {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # CPU agar tidak butuh GPU besar
        token=args.hf_token,
    )
    print(f"[augment] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply augmentation
    if args.method == "gaussian_noise":
        apply_gaussian_noise(model)
    elif args.method == "weight_scaling":
        apply_weight_scaling(model)
    elif args.method == "magnitude_pruning":
        apply_magnitude_pruning(model)

    # Save locally
    print(f"\n[augment] Saving to {args.save_dir} ...")
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("[augment] Saved locally.")

    # Upload to HuggingFace
    print(f"\n[augment] Uploading to HuggingFace: {full_repo_id} ...")
    api = HfApi(token=args.hf_token)
    create_repo(full_repo_id, token=args.hf_token, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=args.save_dir,
        repo_id=full_repo_id,
        repo_type="model",
    )
    print(f"\n✅ Upload selesai!")
    print(f"   Augmented repo: https://huggingface.co/{full_repo_id}")
    print(f"\n   Gunakan di run_enviroment.sh:")
    print(f'   MODEL="{full_repo_id}"')


if __name__ == "__main__":
    main()
