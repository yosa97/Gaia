import glob
import json
import os
import re
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers import AutoTokenizer

DPO = "dpo"
GRPO = "grpo"
INSTRUCT = "instruct"

MODEL_CONFIG = {
    "facebook/opt-1.3b": {"model_size": 1_300_000_000},
    "facebook/opt-3b": {"model_size": 3_000_000_000},
    "facebook/opt-6.7b": {"model_size": 6_700_000_000},
    "facebook/opt-13b": {"model_size": 13_000_000_000},
    "EleutherAI/gpt-neo-1.3B": {"model_size": 1_300_000_000},
    "EleutherAI/gpt-neo-125m": {"model_size": 125_000_000},
    "bigscience/bloom-560m": {"model_size": 560_000_000},
    "TinyLlama/TinyLlama_v1.1": {"model_size": 1_100_000_000},
}

# Architecture / model membership tables, hoisted to module level so each
# helper is a simple lookup. Values and semantics are unchanged.
_REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<reflection>", "</reflection>"),
)
_LIGER_ARCHS = frozenset({
    "qwen2forcausallm",
    "llamaforcausallm",
    "gemma2forcausallm",
    "mixtralforcausallm",
    "mistralforcausallm",
    "qwen3forcausallm",
    "phi3forcausallm",
    "gemmaforcausallm",
})
_NO_FLASH_ARCHS = frozenset({"gptneoforcausallm", "bloomforcausallm", "gptossforcausallm"})
_NO_VLLM_ARCHS = frozenset({"gptneoforcausallm", "bloomforcausallm"})
_NO_VLLM_MODELS = frozenset({
    "Eurdem/Defne_llama3_2x8B",
    "heegyu/WizardVicuna-open-llama-3b-v2",
    "openlm-research/open_llama_3b",
    "TitanML/tiny-mixtral",
    "dunzhang/stella_en_1.5B_v5",
    "oopsung/llama2-7b-n-ox-test-v1",
    "microsoft/phi-2",
    "databricks/dolly-v2-3b",
})
_BPE_NO_ACTION_MASK_MODELS = frozenset({
    "codellama/CodeLlama-7b-Instruct-hf",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
})

hf_api = HfApi()


def is_reasoning_tokenizer(tokenizer: AutoTokenizer) -> bool:
    try:
        vocab = tokenizer.get_vocab()
        return any(
            open_tag in vocab and close_tag in vocab
            for open_tag, close_tag in _REASONING_TAG_PAIRS
        )
    except Exception:
        return False


def get_model_architecture(model_path: str) -> str:
    # Tournament containers have no internet; the base model is pre-cached at a
    # local path, so resolve offline to avoid a doomed huggingface.co lookup.
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception:
        return "Unknown"


def get_use_liger(architecture: str) -> str:
    return "True" if architecture.lower() in _LIGER_ARCHS else "False"


def count_params_from_safetensors(model_dir):
    shards = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if not shards:
        return None
    total_params = 0
    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        tensors = load_file(shard_path)
        total_params += sum(v.numel() for v in tensors.values())
    return total_params


def count_params_from_bin(model_dir):
    shards = glob.glob(os.path.join(model_dir, "*.bin"))
    if not shards:
        return None
    total_params = 0
    for shard_path in shards:
        print(f"Loading shard: {shard_path}")
        try:
            state_dict = torch.load(shard_path, map_location="cpu")
            total_params += sum(v.numel() for v in state_dict.values())
        except Exception as e:
            print(f"cannot load {shard_path}: {e}")
            continue
    return total_params


def get_model_size_from_local_path(model_path: str) -> int:
    for loader, label in ((count_params_from_safetensors, "safetensors"),
                          (count_params_from_bin, "bin")):
        size = loader(model_path)
        if size is not None and size > 1000:
            print(f"Model size from {label}: {size}")
            return size
    return None


def get_gpu_count():
    return torch.cuda.device_count()


def get_model_num_params(model_id: str, model_path: str) -> int:
    if model_id in MODEL_CONFIG:
        return MODEL_CONFIG[model_id]["model_size"]
    try:
        size = get_model_size_from_local_path(model_path)
        if size is not None:
            return size
        raise Exception(f"Cannot get model size from {model_path}")
    except Exception as e:
        print(f"Error getting model size from safetensors: {e}")
        try:
            matched = re.search(r"(\d+)(?=[bB])", model_id)
            model_size = int(matched.group(1)) * 1_000_000_000 if matched else None
            print(f"Model size from regex: {model_size}")
            return model_size
        except Exception as e:
            print(f"Error getting model size from regex: {e}")
            return None


def disable_flash_attention(architecture: str, model: str) -> str:
    if model == "microsoft/phi-2":
        return "True"
    if "falcon-rw" in model.lower():  # ex, tiiuae/falcon-rw-1b
        return "True"
    return architecture.strip().lower() in _NO_FLASH_ARCHS


def disable_action_mask(model: str) -> str:
    return "True" if model in _BPE_NO_ACTION_MASK_MODELS else "False"


def get_use_vllm(architecture: str, model: str) -> str:
    if model in _NO_VLLM_MODELS:
        return False
    if "falcon-rw" in model.lower():
        return False
    return architecture not in _NO_VLLM_ARCHS


def get_gradient_checkpointing(model: str) -> str:
    if "falcon-rw" in model.lower():
        return "False"
    return "True"


def get_data_size(data_path: str) -> int:
    with open(data_path, "r") as f:
        data = json.load(f)
    return len(data)
