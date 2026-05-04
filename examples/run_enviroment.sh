#!/bin/bash

TASK_ID="1"
MODEL=""
DATASET="dummy"

# ── SFT Warm-start Dataset (opsional) ──────────────────────────────────────
# Isi dengan HF dataset ID dari whitelist G.O.D untuk SFT warm-start.
# Kosongkan ("") untuk skip SFT dan langsung GRPO.
# Whitelist: GoodStartLabs/gin-rummy-trajectories-32k
#            gradients-io-tournaments/ArkadiumGinrummy
#            SoelMgd/Poker_Dataset
#            RZ412/PokerBench
#            the-acorn-ai/textarena-player-game-traces
#            tasksource/Boardgame-QA
REQUESTED_DATASETS="GoodStartLabs/gin-rummy-trajectories-32k"

DATASET_TYPE='{
  "environment_name": "gin_rummy"
}'

# Inject requested_datasets ke DATASET_TYPE jika diset
# text_trainer.py akan membaca field ini untuk memilih SFT warm-start
if [ -n "$REQUESTED_DATASETS" ]; then
  DATASET_TYPE=$(echo "$DATASET_TYPE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
d['requested_datasets'] = ['$REQUESTED_DATASETS']
print(json.dumps(d))
")
  echo "[SFT] DATASET_TYPE with requested_datasets: $DATASET_TYPE"
fi

FILE_FORMAT="s3"
HOURS_TO_COMPLETE=3

# ── Max training steps ─────────────────────────────────────────────────────
# Berdasarkan observasi aktual step time:
#   gin_rummy   → 250  (step time 47-57s setelah curriculum max_turn naik ke 11-12)
#   liars_dice  → 280  (step time ~31s, 3 jam ≈ 280 steps)
#   leduc_poker → 300  (step time lebih cepat)
MAX_STEPS=320

# ── Wandb ──────────────────────────────────────────────────────────────────
# Set WANDB_TOKEN to enable online logging.
# Leave it empty (WANDB_TOKEN="") to fall back to offline mode automatically.
WANDB_TOKEN=""
WANDB_PROJECT="tournament-environments"   # project name on wandb.ai (change freely)

# ── HuggingFace upload ─────────────────────────────────────────────────────
HUGGINGFACE_USERNAME=""
HUGGINGFACE_TOKEN=""
EXPECTED_REPO_NAME=""
LOCAL_FOLDER="/app/checkpoints/$TASK_ID/$EXPECTED_REPO_NAME"
DOCKER_BUILDKIT=1

# ── Auto-detect wandb mode ─────────────────────────────────────────────────
# Online mode when a token is present; offline mode otherwise so logs are
# saved locally inside the container and can be synced later.
WANDB_RUN_NAME="${TASK_ID}_${EXPECTED_REPO_NAME}"

if [ -n "$WANDB_TOKEN" ]; then
  WANDB_MODE="online"
  echo "[wandb] Token detected — ONLINE mode (project: $WANDB_PROJECT, run: $WANDB_RUN_NAME)"
else
  WANDB_MODE="offline"
  echo "[wandb] No token set — OFFLINE mode (logs saved locally inside container)"
fi

# ── Directory setup ────────────────────────────────────────────────────────
CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
OUTPUTS_DIR="$(pwd)/outputs"
mkdir -p "$CHECKPOINTS_DIR"
chmod 777 "$CHECKPOINTS_DIR"
mkdir -p "$OUTPUTS_DIR"
chmod 777 "$OUTPUTS_DIR"

# Create a shared Docker network for trainer <-> env server communication
NETWORK_NAME="env_training_net"
docker network create "$NETWORK_NAME" 2>/dev/null || true

# Build the downloader image
docker build -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
# --build-arg SCRIPTS_CACHE_BUST: memaksa Docker rebuild layer COPY scripts
# agar code terbaru selalu dipakai. Layer pip install tetap cached (cepat).
docker build \
  --build-arg SCRIPTS_CACHE_BUST="$(date +%s)" \
  -t standalone-text-trainer \
  -f dockerfiles/standalone-text-trainer.dockerfile .

# Build the hf-uploader image
docker build -t hf-uploader -f dockerfiles/hf-uploader.dockerfile .

# Download model and generate dummy dataset
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

# Start the environment server (MCTS API for liars dice)
# One server per GPU — adjust if you have multiple GPUs
echo "Starting environment server..."
docker run -d --rm \
  --network "$NETWORK_NAME" \
  --name env-server-0 \
  phoenixbeaudry/game:mcts-api

# ── Cleanup container lama (jika ada) ────────────────────────────────────────
# Mencegah error "container name already in use" saat run ulang
echo "Cleaning up old trainer container if exists..."
docker rm -f grpo-text-trainer-example 2>/dev/null || true
# ─────────────────────────────────────────────────────────────────────────────
# Wait for the env server to be ready
echo "Waiting for environment server to be healthy..."
sleep 10  # Adjust as needed, or add a proper health check loop

# Build the ENVIRONMENT_SERVER_URLS
# For multi-GPU, start additional env-server-1, env-server-2, etc. and comma-separate them
ENV_SERVER_URLS="http://env-server-0:8000"

# Run the trainer
# text_trainer.py reads wandb mode/project from CLI args (--wandb-mode, --wandb-project).
# The token is read from the WANDB_API_KEY / WANDB_TOKEN env var inside the container.
# WANDB_RUN_ID and WANDB_NAME env vars are set by text_trainer.py automatically.
echo "Starting trainer (wandb mode: $WANDB_MODE)..."

# EXTERNAL WATCHDOG TIMER (Ensures strictly accurate timeout)
TIMEOUT_SECONDS=$(echo "$HOURS_TO_COMPLETE * 3600" | bc | cut -d. -f1)
(sleep $TIMEOUT_SECONDS && echo "[WATCHDOG] TIMEOUT REACHED ($HOURS_TO_COMPLETE hrs) - Stopping container..." && docker stop grpo-text-trainer-example 2>/dev/null) &
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
  --env ENVIRONMENT_SERVER_URLS="$ENV_SERVER_URLS" \
  --env WANDB_API_KEY="$WANDB_TOKEN" \
  --env WANDB_TOKEN="$WANDB_TOKEN" \
  --env WANDB_INIT_TIMEOUT=300 \
  --name grpo-text-trainer-example \
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
  --wandb-project "$WANDB_PROJECT" \
  --max-steps "$MAX_STEPS" \
  --dataset-type "$DATASET_TYPE" || true

# Batalkan Watchdog jika proses trainer selesai lebih cepat secara natural
kill $TIMER_PID 2>/dev/null || true

# Cleanup env server and network
echo "Stopping environment server..."
docker stop env-server-0 2>/dev/null || true
docker network rm "$NETWORK_NAME" 2>/dev/null || true

# Upload checkpoints to HuggingFace; also sync offline wandb runs if any
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
  --name hf-uploader \
  hf-uploader