#!/bin/bash
# Environment training runner — pure GRPO for SN56 tournament (Goofspiel).
#
# Tournament rules (as of 2026):
#   1. You may NOT bring your own dataset in the docker image.
#   2. You may NOT bring a pretrained model in the docker image.
#   3. You may NOT do any SFT for environment tasks.
#
# This script runs GRPO-only training on the Goofspiel environment.
# Only one supported environment in tournament: goof_spiel (Task IDs 0-99999999).
#
# Usage:
#   bash examples/run_environment.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash examples/run_environment.sh

# Always run from the repo root so relative paths (dockerfiles/, etc.) resolve correctly.
cd "$(dirname "$0")/.." || exit 1

TASK_ID="1"
MODEL="${MODEL:-NousResearch/Hermes-3-Llama-3.2-3B}"
DATASET="dummy"
DATASET_TYPE='{
  "environment_name": "goof_spiel"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=3

# ── Wandb ──────────────────────────────────────────────────────────────────
WANDB_TOKEN=""
WANDB_PROJECT="environment"

# ── HuggingFace upload ─────────────────────────────────────────────────────
HUGGINGFACE_USERNAME=""
HUGGINGFACE_TOKEN=""
EXPECTED_REPO_NAME=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
DOCKER_BUILDKIT=1

# ── Auto-detect wandb mode ─────────────────────────────────────────────────
WANDB_RUN_NAME="${TASK_ID}_${EXPECTED_REPO_NAME}_grpo_env"

if [ -n "$WANDB_TOKEN" ]; then
  WANDB_MODE="online"
  echo "[wandb] Token detected — ONLINE mode (project: $WANDB_PROJECT, run: $WANDB_RUN_NAME)"
else
  WANDB_MODE="offline"
  echo "[wandb] No token set — OFFLINE mode"
fi

# ── Directory setup ────────────────────────────────────────────────────────
CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR"

NETWORK_NAME="env_training_net"
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Build images
docker build -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .
docker build -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .
docker build -t hf-uploader -f dockerfiles/hf-uploader.dockerfile .

# Download model
echo "Downloading model..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --file-format "$FILE_FORMAT" \
  --task-type "EnvTask"

# ── Start environment server (Goofspiel via Affinetes/OpenSpiel) ───────────
# The validator provisions one server per GPU during tournament.
# Locally: start the environment server before running this script,
# then set ENVIRONMENT_SERVER_URLS to point to it.
# Example: ENVIRONMENT_SERVER_URLS="http://localhost:8001" bash examples/run_environment.sh
ENV_SERVER_URLS="${ENVIRONMENT_SERVER_URLS:-http://localhost:8001}"
echo "[env] Using environment servers: $ENV_SERVER_URLS"

# ── Pure GRPO training (no SFT — tournament Rule #3) ──────────────────────
echo "Starting GRPO environment trainer..."

TIMEOUT_SECONDS=$(echo "$HOURS_TO_COMPLETE * 3600" | bc | cut -d. -f1)
(sleep $TIMEOUT_SECONDS && echo "[WATCHDOG] TIMEOUT — stopping container..." && docker stop grpo-env-trainer 2>/dev/null) &
TIMER_PID=$!

docker run --rm --gpus all \
  --shm-size=100gb \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network "$NETWORK_NAME" \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env WANDB_API_KEY="$WANDB_TOKEN" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env WANDB_INIT_TIMEOUT=300 \
  --env ENVIRONMENT_SERVER_URLS="$ENV_SERVER_URLS" \
  --env HF_TOKEN="$HUGGINGFACE_TOKEN" \
  --name grpo-env-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "EnvTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-project "$WANDB_PROJECT" || true

kill $TIMER_PID 2>/dev/null || true

# Cleanup
docker network rm "$NETWORK_NAME" 2>/dev/null || true

# Upload to HF
echo "Uploading outputs..."
docker run --rm --gpus all \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --env HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  --env HUGGINGFACE_USERNAME="$HUGGINGFACE_USERNAME" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env WANDB_API_KEY="$WANDB_TOKEN" \
  --env WANDB_MODE="$WANDB_MODE" \
  --env TASK_ID="$TASK_ID" \
  --env EXPECTED_REPO_NAME="$EXPECTED_REPO_NAME" \
  --env LOCAL_FOLDER="$LOCAL_FOLDER" \
  --env MODEL="$MODEL" \
  --name hf-uploader \
  hf-uploader
