#!/bin/bash
set -e

redis-server --daemonize yes
sleep 10

# Tournament containers run with NO public internet. The base model + datasets
# are pre-downloaded into /cache by the validator, so every HuggingFace lookup
# must resolve from the local cache. Without these, transformers tries to reach
# huggingface.co (e.g. AutoConfig on the anonymized model id), hits a DNS
# failure, and wastes the time budget on retry backoff. Force offline mode so
# from_pretrained uses the local cache and fails fast if something is missing.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "*****Running text trainer"
source /workspace/.grpo_env/bin/activate
python3 /workspace/scripts/text_trainer.py "$@"
deactivate
# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line; does not change behavior.
