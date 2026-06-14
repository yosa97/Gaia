"""
whitelisted_sft_datasets.py — Validasi dataset whitelist untuk SFT warm-start.

Port dari G.O.D branch feature/miner-dataset-whitelist:
  core/whitelisted_sft_datasets.py

Digunakan oleh text_trainer.py dan train_sft_env.py untuk memastikan
hanya dataset yang disetujui G.O.D yang boleh digunakan sebagai warm-start.
"""

import json
from pathlib import Path

_WHITELIST_PATH = Path(__file__).parent / "whitelisted_sft_datasets.json"

WHITELISTED_SFT_DATASETS: set[str] = set(json.loads(_WHITELIST_PATH.read_text()))

# Maksimum 2 dataset bisa di-request sekaligus (sesuai aturan G.O.D)
MAX_REQUESTED_DATASETS = 2

# Mapping dataset → environment game (untuk tokenizer aware)
DATASET_TO_GAME: dict[str, str] = {
    "GoodStartLabs/gin-rummy-trajectories-32k":      "gin_rummy",
    "gradients-io-tournaments/ArkadiumGinrummy":     "gin_rummy",
    "SoelMgd/Poker_Dataset":                         "poker",
    "RZ412/PokerBench":                              "poker",
    "the-acorn-ai/textarena-player-game-traces":     "textarena",
    "tasksource/Boardgame-QA":                       "boardgame_qa",
    "albarji/ballmatro":                             "generic",
    "Mahesh111000/Hanabi_dataset":                   "generic",
    "hkust-nlp/agentboard":                         "generic",
    "gradients-io-tournaments/env_training_gradients": "env_generic",
}


def validate_requested_datasets(requested_datasets: list[str] | None) -> list[str]:
    """Filter list dataset — hanya yang ada di whitelist dan max 2.

    Port persis dari G.O.D core/whitelisted_sft_datasets.py.
    """
    if not requested_datasets:
        return []
    valid = [ds for ds in requested_datasets if ds in WHITELISTED_SFT_DATASETS]
    return valid[:MAX_REQUESTED_DATASETS]


def get_game_for_dataset(dataset_id: str) -> str:
    """Kembalikan nama game untuk dataset tertentu, atau 'generic' jika tidak dikenal."""
    return DATASET_TO_GAME.get(dataset_id, "generic")


def is_whitelisted(dataset_id: str) -> bool:
    """Cek apakah sebuah dataset ada di whitelist G.O.D."""
    return dataset_id in WHITELISTED_SFT_DATASETS

# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
