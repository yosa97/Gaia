#!/bin/bash
# Environment training runner — SFT multi-env path for SN56 tournament.
#
# Tournament rules (verified vs gradients-ai/G.O.D, 2026-06-08):
#   1. You may NOT bundle your own dataset in the docker image.
#   2. You may NOT bundle a pretrained model in the docker image.
#   3. SFT IS allowed, but only via whitelisted requested datasets
#      (core/whitelisted_sft_datasets.json — max 2, mounted read-only).
#
# Active tournament envs: gin_rummy, liars_dice, leduc_poker, intercode.
# Composite tasks: R1 = 2 envs/task, R2 = 4, R3 = 6. This script defaults to a
# 2-env composite task (R1 shape) and exercises the SFT expert-trajectory path:
#   text_trainer.py → sft_env_config.get_training_json_multi_env
#     → envs/generate_trajectories.py (per env, vs MCTS opponent)
#     → envs/merge_datasets.py → train_sft_env.py
# intercode (dataset-builder env) needs MINER_DATASETS_DIR mounted — see below.
#
# Usage:
#   bash examples/run_environment.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash examples/run_environment.sh
#   ENVIRONMENTS='["liars_dice","gin_rummy","leduc_poker","intercode"]' bash examples/run_environment.sh
#   SINGLE_ENV=liars_dice bash examples/run_environment.sh   # legacy single-env shape

# Always run from the repo root so relative paths (dockerfiles/, etc.) resolve correctly.
cd "$(dirname "$0")/.." || exit 1

TASK_ID="1"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DATASET="dummy"

# ── Environment selection ──────────────────────────────────────────────────
# Default: 2-env composite (R1 shape). Override via ENVIRONMENTS (JSON list)
# or SINGLE_ENV (legacy environment_name payload).
ENVIRONMENTS="${ENVIRONMENTS:-[\"liars_dice\",\"gin_rummy\"]}"
if [ -n "$SINGLE_ENV" ]; then
  DATASET_TYPE="{\"environment_name\": \"$SINGLE_ENV\"}"
  echo "[env] Single-env (legacy) payload: $SINGLE_ENV"
else
  DATASET_TYPE="{\"environment_names\": $ENVIRONMENTS}"
  echo "[env] Multi-env composite payload: $ENVIRONMENTS"
fi

FILE_FORMAT="s3"
HOURS_TO_COMPLETE=1.5   # group-task budget (boss from-scratch task uses 3.0)

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
WANDB_RUN_NAME="${TASK_ID}_${EXPECTED_REPO_NAME}_sft_env"

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
MINER_DATASETS_HOST_DIR="${MINER_DATASETS_HOST_DIR:-$(pwd)/miner_datasets}"
# Trainer working dir (/workspace/scripts/datasets) holds tokenize_*.log,
# train_*.log, add_noise_*.log AND the generated SFT datasets. Mount it to the
# host so debugging survives `docker run --rm`.
TRAIN_LOGS_DIR="$(pwd)/train_logs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$MINER_DATASETS_HOST_DIR" "$TRAIN_LOGS_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$TRAIN_LOGS_DIR"

# ── Miner datasets (whitelisted SFT path — Rule 3) ────────────────────────
# In tournament the validator mounts these from your requested_datasets.
# Locally: pre-download into $MINER_DATASETS_HOST_DIR, dir name = id with "/"→"--".
# intercode REQUIRES gradients-io-tournaments--intercode_bigcode_combined_12k.
MINER_DATASETS_LIST=""
for d in "$MINER_DATASETS_HOST_DIR"/*/; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  MINER_DATASETS_LIST="${MINER_DATASETS_LIST:+$MINER_DATASETS_LIST,}$name"
done
echo "[miner-datasets] $MINER_DATASETS_HOST_DIR -> ${MINER_DATASETS_LIST:-<none>}"
if echo "$DATASET_TYPE" | grep -q intercode && ! echo "$MINER_DATASETS_LIST" | grep -q intercode; then
  echo "[miner-datasets] WARNING: intercode requested but intercode_bigcode_combined_12k not present —"
  echo "                 intercode will be DROPPED from the multi-env task (graceful-drop path)."
fi

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

# ── Environment servers (OpenSpiel games need them; intercode does NOT) ────
# The validator provisions one server per GPU during tournament (the MCTS API
# image). Locally: start them first (bash run_environment_env.sh — they publish
# host ports 8001+). NOTE: inside the trainer container `localhost` is the
# container itself, NOT the VM — so the default uses host.docker.internal
# (mapped to the host gateway via --add-host below). Override if needed:
#   ENVIRONMENT_SERVER_URLS="http://host.docker.internal:8001,http://host.docker.internal:8002" bash examples/run_environment.sh
ENV_SERVER_URLS="${ENVIRONMENT_SERVER_URLS:-http://host.docker.internal:8001}"
echo "[env] Using environment servers: $ENV_SERVER_URLS"

# Pre-flight: when the payload contains OpenSpiel games, verify at least one
# env server is actually listening BEFORE downloading models / booting the
# trainer. host.docker.internal == this host, so test via localhost here.
if echo "$DATASET_TYPE" | grep -qE "liars_dice|gin_rummy|leduc_poker"; then
  reachable=0
  IFS=',' read -ra _urls <<< "$ENV_SERVER_URLS"
  for u in "${_urls[@]}"; do
    probe="${u/host.docker.internal/localhost}"
    if curl -s -o /dev/null --max-time 3 "$probe"; then
      reachable=1; echo "[env] OK: $u reachable"
    else
      echo "[env] UNREACHABLE: $u"
    fi
  done
  if [ "$reachable" -eq 0 ]; then
    echo "[env] FATAL: no environment server reachable. Start them first:"
    echo "       bash run_environment_env.sh   (ports 8001-8004)"
    echo "       docker ps --filter name=agentgym-server"
    exit 1
  fi
fi

# ── Stale-output guard ──────────────────────────────────────────────────────
# A failed previous run can leave a noise-fallback model in outputs/$TASK_ID.
# text_trainer also guards against this in-container, but clean host-side too
# so the uploader can never ship a stale model.
if [ -d "$OUTPUTS_DIR/$TASK_ID" ]; then
  echo "[cleanup] Removing stale output from previous run: $OUTPUTS_DIR/$TASK_ID"
  rm -rf "$OUTPUTS_DIR/$TASK_ID"
fi
# Also clear this task's logs from the previous run, so post-run checks
# (add_noise detection, upload guard) never react to stale files.
rm -f "$TRAIN_LOGS_DIR/tokenize_${TASK_ID}.log" \
      "$TRAIN_LOGS_DIR/train_${TASK_ID}.log" \
      "$TRAIN_LOGS_DIR/add_noise_${TASK_ID}.log"

# ── SFT multi-env training (whitelist-compliant; GRPO fallback for envs
#    without an SFT generator) ──────────────────────────────────────────────
echo "Starting SFT environment trainer..."

TIMEOUT_SECONDS=$(echo "$HOURS_TO_COMPLETE * 3600" | bc | cut -d. -f1)
(sleep $TIMEOUT_SECONDS && echo "[WATCHDOG] TIMEOUT — stopping container..." && docker stop sft-env-trainer 2>/dev/null) &
TIMER_PID=$!

docker run --rm --gpus all \
  --shm-size=100gb \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --network "$NETWORK_NAME" \
  --add-host=host.docker.internal:host-gateway \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --volume "$OUTPUTS_DIR:/app/checkpoints/:rw" \
  --volume "$MINER_DATASETS_HOST_DIR:/cache/miner_datasets:ro" \
  --volume "$TRAIN_LOGS_DIR:/workspace/scripts/datasets:rw" \
  --env WANDB_API_KEY="$WANDB_TOKEN" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env WANDB_INIT_TIMEOUT=300 \
  --env ENVIRONMENT_SERVER_URLS="$ENV_SERVER_URLS" \
  --env MINER_DATASETS_DIR="/cache/miner_datasets" \
  --env MINER_DATASETS="$MINER_DATASETS_LIST" \
  --env HF_TOKEN="$HUGGINGFACE_TOKEN" \
  --name sft-env-trainer \
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

# ── Post-run summary + log hints ───────────────────────────────────────────
echo
echo "[logs] Trainer logs preserved in: $TRAIN_LOGS_DIR"
echo "       tokenize_${TASK_ID}.log  = trajectory gen + merge output"
echo "       train_${TASK_ID}.log     = SFT/GRPO training output"
echo "       add_noise_${TASK_ID}.log = ONLY exists if training FAILED (fallback model!)"
if [ -f "$TRAIN_LOGS_DIR/add_noise_${TASK_ID}.log" ]; then
  echo "[logs] WARNING: add_noise log found — training FAILED, saved model is noise-fallback."
  echo "       Root cause: tail -50 $TRAIN_LOGS_DIR/tokenize_${TASK_ID}.log $TRAIN_LOGS_DIR/train_${TASK_ID}.log"
fi

# Upload to HF — skip cleanly when no valid token (avoids 401 crash at the end)
if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "[upload] HUGGINGFACE_TOKEN empty — SKIPPING upload (local test mode)."
  echo "         Model output: $OUTPUTS_DIR/$TASK_ID/$EXPECTED_REPO_NAME"
  exit 0
fi
# Never upload a noise-fallback model from a failed local run (8GB of nothing).
if [ -f "$TRAIN_LOGS_DIR/add_noise_${TASK_ID}.log" ]; then
  echo "[upload] Training FAILED (noise-fallback detected) — SKIPPING upload."
  exit 1
fi
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
