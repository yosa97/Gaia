#!/bin/bash
# ============================================================
# MODEL ANON — SFT Only (PvP Tournament)
# Nama model: yosa_AnonPvP
#
# Strategi: SFT murni dari 3 dataset.
#   - Tidak ada signature khusus → sulit di-profile lawan
#   - 3 dataset: env_training + boardgame-qa + gin-rummy-trajectories
#   - Untuk Gin Rummy: ada Phase 2.5 specialist dengan 32k traces
#
# Cara pakai:
#   1. Isi variabel di bagian KONFIGURASI di bawah
#   2. Jalankan: bash examples/run_anon.sh
# ============================================================

cd "$(dirname "$0")/.." || exit 1

# ── KONFIGURASI ────────────────────────────────────────────
TASK_ID="1"
MODEL=""                        # Kosong = pakai default Hermes-3-3B
DATASET="dummy"
# Ganti game di sini: liars_dice | gin_rummy | leduc_poker
DATASET_TYPE='{
  "environment_name": "gin_rummy"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=3

WANDB_TOKEN=""
WANDB_PROJECT="environment"

HUGGINGFACE_USERNAME="yosa722"
HUGGINGFACE_TOKEN="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"
EXPECTED_REPO_NAME="yosa_AnonPvP"
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"

# ── Dataset Slots ──────────────────────────────────────────
MINER_DATASETS_HOST_DIR="$(pwd)/miner_datasets_cache"
# Slot 1: env_training (wajib)
MINER_DATASET_REPO_1="gradients-io-tournaments/env_training_gradients"
MINER_DATASET_DIR_1="gradients-io-tournaments__env_training_gradients"
# Slot 2: Boardgame-QA (warm-up reasoning)
MINER_DATASET_REPO_2="tasksource/Boardgame-QA"
MINER_DATASET_DIR_2="tasksource__Boardgame-QA"
# Slot 3: Gin Rummy specialist (32k expert traces)
MINER_DATASET_REPO_3="GoodStartLabs/gin-rummy-trajectories-32k"
MINER_DATASET_DIR_3="GoodStartLabs__gin-rummy-trajectories-32k"

DOCKER_BUILDKIT=1
# ──────────────────────────────────────────────────────────

# MODE 1: Pure SFT (tidak ada GRPO)
TRAINING_MODE_ENVS="--env SFT_ONLY=1"

# Auto-detect wandb mode
WANDB_RUN_NAME="${TASK_ID}_${EXPECTED_REPO_NAME}_anon_sft"
if [ -n "$WANDB_TOKEN" ]; then
  WANDB_MODE="online"
  echo "[wandb] Token detected — ONLINE mode"
else
  WANDB_MODE="offline"
  echo "[wandb] No token — OFFLINE mode"
fi

# Directory setup
CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$MINER_DATASETS_HOST_DIR"
chmod 777 "$CHECKPOINTS_DIR" "$OUTPUTS_DIR" "$MINER_DATASETS_HOST_DIR"

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

# Pre-download datasets
download_dataset() {
  local repo="$1"
  local dir_name="$2"
  local target="$MINER_DATASETS_HOST_DIR/$dir_name"
  if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
    echo "[sft] Dataset already cached: $target"
    return 0
  fi
  echo "[sft] Downloading $repo -> $target ..."
  TOKEN_ARG=""
  if [ -n "$HUGGINGFACE_TOKEN" ]; then
    TOKEN_ARG="--token $HUGGINGFACE_TOKEN"
  fi
  docker run --rm \
    --volume "$MINER_DATASETS_HOST_DIR:/data:rw" \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    python:3.11-slim \
    bash -c "pip install -q huggingface_hub && hf download '$repo' --repo-type dataset --local-dir /data/$dir_name $TOKEN_ARG"
  if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
    echo "[sft] Cached: $target"
  else
    echo "[sft] WARN: download $repo failed; will try direct download in trainer"
  fi
}

download_dataset "$MINER_DATASET_REPO_1" "$MINER_DATASET_DIR_1"
download_dataset "$MINER_DATASET_REPO_2" "$MINER_DATASET_DIR_2"
download_dataset "$MINER_DATASET_REPO_3" "$MINER_DATASET_DIR_3"

echo "Starting ANON SFT trainer (MODE 1 — Pure SFT)..."

TIMEOUT_SECONDS=$(echo "$HOURS_TO_COMPLETE * 3600" | bc | cut -d. -f1)
(sleep $TIMEOUT_SECONDS && echo "[WATCHDOG] TIMEOUT — stopping container..." && docker stop full-sft-trainer 2>/dev/null) &
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
  --volume "$MINER_DATASETS_HOST_DIR:/cache/miner_datasets:ro" \
  --env WANDB_API_KEY="$WANDB_TOKEN" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env WANDB_INIT_TIMEOUT=300 \
  --env MINER_DATASETS_DIR=/cache/miner_datasets \
  --env MINER_DATASETS="$MINER_DATASET_DIR_1,$MINER_DATASET_DIR_2,$MINER_DATASET_DIR_3" \
  $TRAINING_MODE_ENVS \
  --env HF_TOKEN="$HUGGINGFACE_TOKEN" \
  --name full-sft-trainer \
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
docker network rm "$NETWORK_NAME" 2>/dev/null || true

# Upload to HuggingFace
echo "Uploading yosa_AnonPvP to HuggingFace..."
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
