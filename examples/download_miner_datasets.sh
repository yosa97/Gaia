#!/bin/bash
# Download whitelisted requested-datasets for LOCAL TESTING into miner_datasets/.
#
# In the real tournament the VALIDATOR downloads these (from your miner
# endpoint's requested_datasets) and mounts them read-only at
# /cache/miner_datasets/{id with "/" -> "--"}. This script replicates that
# layout locally so examples/run_environment.sh can mount it the same way.
#
# IMPORTANT: miner_datasets/ must stay OUT of git and OUT of the docker image
# (tournament Rule 1: no bundled datasets). It is gitignored; the trainer
# dockerfile only COPYs scripts/, so the image stays clean.
#
# Usage:
#   bash examples/download_miner_datasets.sh
#   HF_TOKEN=hf_xxx bash examples/download_miner_datasets.sh   # if rate-limited

set -e
cd "$(dirname "$0")/.." || exit 1

DEST_ROOT="${MINER_DATASETS_HOST_DIR:-$(pwd)/miner_datasets}"
mkdir -p "$DEST_ROOT"

# Datasets to fetch — keep in sync with your miner endpoint's requested_datasets
# (max 2 in tournament; both must be in core/whitelisted_sft_datasets.json).
DATASETS=(
  "gradients-io-tournaments/intercode_bigcode_combined_12k"
  "tasksource/Boardgame-QA"
)

AUTH_HEADER=()
if [ -n "$HF_TOKEN" ]; then
  AUTH_HEADER=(--header "Authorization: Bearer $HF_TOKEN")
fi

for ds in "${DATASETS[@]}"; do
  dirname="${ds//\//--}"
  dest="$DEST_ROOT/$dirname"
  mkdir -p "$dest"
  echo "[download] $ds -> $dest"
  if command -v hf >/dev/null 2>&1; then
    hf download "$ds" --repo-type dataset --local-dir "$dest"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$ds" --repo-type dataset --local-dir "$dest"
  else
    # Fallback: list files via API then curl each (no parquet-only repos with
    # nested paths beyond data/ are expected here).
    files=$(curl -s "${AUTH_HEADER[@]}" "https://huggingface.co/api/datasets/$ds" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(s['rfilename'] for s in d.get('siblings',[]) if not s['rfilename'].startswith('.')))")
    for f in $files; do
      mkdir -p "$dest/$(dirname "$f")"
      echo "  - $f"
      curl -sL "${AUTH_HEADER[@]}" -o "$dest/$f" \
        "https://huggingface.co/datasets/$ds/resolve/main/$f"
    done
  fi
done

echo
echo "[download] Done. Contents:"
find "$DEST_ROOT" -maxdepth 2 -type f | head -20
echo
echo "Run training with:"
echo "  ENVIRONMENTS='[\"liars_dice\",\"gin_rummy\"]' bash examples/run_environment.sh"
echo "  SINGLE_ENV=intercode bash examples/run_environment.sh"
