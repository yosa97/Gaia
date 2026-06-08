# Tournament Environments Workspace Guide

This repository ships a skill-style guide under `.claude/skills/tournament-environments/SKILL.md`.
Use this file as the fast entry point, and open the reference files when you need the full code map.

## What This Repo Actually Does

- Trains GRPO-based agents for turn-based game environments.
- Keeps environment-specific training logic inside the `scripts/envs/` package.
- Uses external environment servers over `/reset` and `/step`, not pure local simulation inside the trainer loop.
- Builds the GRPO training dataset from task-id ranges, so the `prompt` is usually just a stringified task id.
- Relies heavily on curriculum scheduling, reward shaping, action parsing, and action masking.

## Highest-Value Files

- `scripts/train_grpo_env.py`
  Main environment-training entry point. Contains `TrainingArguments`, `ActionMaskedGRPOTrainer`, registry-based environment routing via `get_env_config(...)`, and dataset construction from task-id ranges.
- `scripts/grpo_env_config.py`
  Size-bucket GRPO preset table plus `torchrun`/`deepspeed` command construction and final training-JSON generation. Merges per-env settings from `envs/env_configs.py` on top of the size bucket at runtime.
- `scripts/customized_trainer.py`
  Save/eval callbacks, time-budget logic, generation-config fixes, and resize helpers.
- `scripts/model_utility.py`
  Architecture detection, parameter counting, flash-attention/vLLM/action-mask toggles, and gradient-checkpointing rules.
- `scripts/envs/shared_env.py`
  Central `CurriculumScheduler` base class, `GAMES_TO_TASK_ID_RANGE` table, `init_env_pool(...)`, and the generic `rollout_reward_func`.
- `scripts/envs/env_configs.py`
  `EnvTrainingConfig` / `ModeConfig` registry and `get_env_config(name)` lookup. Holds per-env rollout callables, curriculum factory, and per-mode overrides for the three training modes (`reasoning`, `no_mask`, `full_prompt`).
- `scripts/envs/gin_rummy_env.py`
- `scripts/envs/gin_rummy_opponent_modeling.py`
  Variant of Gin Rummy with `DeadCardTracker`, `BayesianOpponentModel`, and `BayesianOpponentHandModel`. Routed via env name `gin_rummy_opponent_modeling` in the registry.
- `scripts/envs/liar_dice_env.py` (singular "liar")
- `scripts/envs/leduc_poker_env.py`
- `scripts/envs/goof_spiel_env.py`
- `scripts/envs/alf_world_env.py`
  No curriculum, no MCTS opponent. Only env in the registry without a `curriculum_factory`.
- `run_environment_task.sh`
  Local orchestration script that builds images, starts environment servers, runs training containers, and optionally uploads outputs.
- `run_environment_env.sh`
  Boots multiple `agentgym`-style environment servers and writes `ENVIRONMENT_SERVER_URLS`.
- `trainer/`
  Proxy trainer service used by validator/orchestrator flows.
- `affinetes/environments/openspiel/`
  Backend OpenSpiel actor, game config, and game-agent formatting used by the environment server.
- `open_spiel/`
  Upstream OpenSpiel engine submodule. Usually touched through the `affinetes/environments/openspiel/` wrapper layer, not by editing the upstream engine directly.
- `scripts/legacy/`, `scripts/legacy2/`
  Pre-refactor copies of the old `*_environment_function.py` files. Not routed; kept for historical reference only.

## Tournament SN56 — Rules & Structure (verified against official gradients-ai/G.O.D, 2026-06-08)

Official source of truth: https://github.com/gradients-ai/G.O.D (`docs/miners.md`, `core/constants.py`).
The official repo moved from `rayonlabs/G.O.D` to `gradients-ai/G.O.D`.

**Active tournament environments** (per `EnvironmentName` in official `core/constants.py`):
`gin_rummy`, `liars_dice`, `leduc_poker`, `intercode`.
`goof_spiel` is NOT in the official tournament env list. `alfworld` is not used either.
Official task-id ranges: liars_dice 100,000,000–199,999,999; leduc_poker 200,000,000–299,999,999; gin_rummy 300,000,000–399,999,999; intercode 1–200.
Eval is PvP (vs MCTS-opponent payload) for the OpenSpiel games and individual for intercode.

### Tournament Rules
1. You may NOT bundle your own dataset in the docker image.
2. You may NOT bundle a pretrained model in the docker image.
3. SFT **is allowed**, but ONLY via the requested-dataset whitelist: up to 2 Hugging Face datasets from `core/whitelisted_sft_datasets.json`, returned in `requested_datasets` and mounted read-only via `MINER_DATASETS_DIR`/`MINER_DATASETS`.
4. No obfuscated/compiled-only code; LICENSE + NOTICE must match the official repo's files.
5. Submit a full 40-char commit SHA (not a branch name). Required Dockerfile for env tournaments: `dockerfiles/standalone-text-trainer.dockerfile`.

### Tournament Structure
Environment tournaments start Mondays 14:00 UTC; minimum 5 validated miners; entry fee 0.20 TAO per coldkey.
- Participants are split into **groups of 2–6**; one environment task per group in non-final rounds.
- Round 1 uses **2 environments per task**, round 2 uses **4**, round 3 uses **6** (capped by supported env count) — composite multi-env tasks.
- The defending champion (burn hotkey) auto-advances through non-final rounds.
- **Round 2 and later can continue from each miner's previous-round model** (continuous training; see `TrainingStartPoint`: `CONTINUATION`, `FROM_SCRATCH`, `PREVIOUS_WINNER`).
- Up to one non-boss winner advances per group (ties at cutoff allowed).
- **Boss round = 3 tasks**: continuation, from-scratch, and previous-winner/target-model start.
- **Win condition**: contender dethrones boss only with NO boss-round losses AND ≥1 boss-round win. Draws OK; any loss = boss retains.
- Group tasks: `ENV_TRAINING_HOURS = 1.5`; from-scratch boss task: 3.0 hours.

### Boss Defense Threshold (exponential decay)
`threshold = max(0.03, 0.05 × 0.8^(consecutive_wins - 1))`
Boss starts with a 5% per-task advantage, decaying to a 3% floor with consecutive wins.

### Weights
Environment tournament: base weight 0.15, max 0.16. Active participants get 0.0001 each. Ranked decay base 0.3.

### Pending official updates (announced ~early June 2026 — verify before relying on them)
- Memory system for envs + tool calling: targeted release 2026-06-12.
- Continuous training across tournaments + composite tasks: targeted 2026-06-19 (between-round continuation is already merged).
- Dedup detection (per-IP / per-GitHub-account) and KL term on instruct tasks: already merged.
- Othello environment: in development, not yet in `EnvironmentName`.

## Current Code Truths

- **EnvTask routing (updated 2026-06-08)**: `text_trainer.py` now parses `environment_names` (list) for composite multi-env tasks. Envs with an SFT trajectory generator (`gin_rummy`, `liars_dice`, `leduc_poker` — see `envs/sft_env_configs.py`) route to SFT expert-trajectory training (`sft_env_config.py` → `envs/generate_trajectories.py` → `envs/merge_datasets.py` → `train_sft_env.py`). Unsupported envs (e.g. `intercode`) are dropped from multi-env tasks; single unsupported envs fall back to the GRPO path (`grpo_env_config.py`). PvP eval format is matched via `envs/pvp_format.py` + `envs/pvp_assets/` (ID-only output, both player positions).
- `intercode` is supported via `envs/intercode_trajectories.py` (dataset-builder env, no episode play): builds ReAct-format SFT data (byte-matching `eval_intercode.py`'s prompt) from the whitelisted `gradients-io-tournaments/intercode_bigcode_combined_12k`. REQUIRES the miner endpoint to return it in `requested_datasets` (max 2 total). In multi-env chains every per-env gen cmd runs with `--soft-fail` so one env failing can't kill the rest; single-env runs hard-fail. `goof_spiel`/`alfworld` remain in the GRPO registry but are not tournament envs.
- `scripts/model_utility.py:is_reasoning_tokenizer()` now inspects the tokenizer vocab for `<think>/</think>` and similar pairs. It returns `True` for reasoning-capable tokenizers and enables the reasoning-mode branch in `train_grpo_env.py`. It is no longer hardcoded to `False`.
- `scripts/train_grpo_env.py:main()` routes six environments via `get_env_config(...)`:
  `goof_spiel`, `gin_rummy`, `gin_rummy_opponent_modeling`, `liars_dice`, `leduc_poker`, `alfworld`.
- The trainer picks one of three training modes based on tokenizer/flags:
  - `reasoning` — uses `cfg.rollout_last` + `GRPOTrainer`, `max_completion_length=2048`.
  - `no_mask` — uses `cfg.rollout_last` + `GRPOTrainer`, `max_completion_length=16`.
  - `full_prompt` — uses `cfg.rollout_full` + `ActionMaskedGRPOTrainer`, `max_completion_length=16`. **Preferred for tournament: trains on the entire episode, not just first turn.**
  Per-mode overrides (`initial_max_turn`, `rollouts_per_stage`, `trainer_class`, `max_completion_length`) come from each env's `ModeConfig` slots.
- `ActionMaskedGRPOTrainer` expects rollout functions to return `action_mask` aligned exactly with `completion_ids`.
- `scripts/grpo_env_config.py` is for environment GRPO. `scripts/grpo_config.py` is the separate generic text-GRPO config path.
- `train_grpo_env.py` samples up to 200,000 task ids from the selected range and uses those ids as prompts.
- The game stack is layered as: `scripts/envs/*_env.py` -> `affinetes/environments/openspiel/` -> upstream `open_spiel/` via `pyspiel` and `open_spiel.python.*`.
- `scripts/envs/__init__.py` re-exports every rollout/reward function with a game-prefixed name (e.g., `gin_rummy_rollout_full_prompt_and_completion_parallelized_curriculum`). Use those names when referring to public entry points.
- `_VARIANT_OVERRIDES` in `envs/env_configs.py` lets a single edit swap `gin_rummy` to `gin_rummy_opponent_modeling` without touching any caller.
- Whitelisted requested datasets (`MINER_DATASETS_DIR`/`MINER_DATASETS`) can now be consumed via `envs/miner_dataset_loader.py` (optional; expert-trajectory generation is the default SFT data source). `tournament_env_utils.py` parses validator-injected env vars defensively.
- **`examples/run_environment.sh`** is the local test runner — still configured for GRPO on `goof_spiel` (not a tournament env); update it to exercise the SFT multi-env path when testing locally.

## Thinking Guidance

- For non-trivial changes to reward shaping, curriculum, action parsing, or environment-server interaction, use MCP sequential thinking first to map dependencies before editing.
- Especially for `gin_rummy_opponent_modeling.py` and `liar_dice_env.py`, treat shaping constants and parser assumptions as coupled systems rather than isolated tweaks.

## How To Navigate Changes

1. If the task is about rollout behavior, reward shaping, parsing, or curriculum:
   Read `scripts/envs/<game>_env.py` first (or `gin_rummy_opponent_modeling.py` for that variant).
2. If the task is about trainer behavior, action masking, or environment selection:
   Read `scripts/train_grpo_env.py`.
3. If the task is about the env registry, mode overrides, or adding a new env:
   Read `scripts/envs/env_configs.py` and `scripts/envs/shared_env.py`.
4. If the task is about hyperparameters, launch flags, or model-specific overrides:
   Read `scripts/grpo_env_config.py` and `scripts/model_utility.py`.
5. If the task is about container startup, environment servers, or validator orchestration:
   Read `run_environment_task.sh`, `run_environment_env.sh`, and `trainer/`.
6. If the task is about how observations/legal actions are produced by the backend:
   Read `affinetes/environments/openspiel/env.py`, `game_config.py`, and the relevant file in `affinetes/environments/openspiel/agents/`.

## Reference Index

- [Skill entry point](skills/tournament-environments/SKILL.md)
- [Architecture and pipeline](skills/tournament-environments/references/architecture-and-pipeline.md)
- [Environment functions](skills/tournament-environments/references/environment-functions.md)
- [Configs and operations](skills/tournament-environments/references/configs-and-ops.md)

## Practical Reading Order

- Start with `scripts/train_grpo_env.py` to understand the three training modes and registry lookup.
- Then open `scripts/envs/env_configs.py` to see per-env and per-mode overrides.
- Then open the matching `scripts/envs/<game>_env.py` (and `gin_rummy_opponent_modeling.py` when applicable).
- Then open `scripts/grpo_env_config.py` to see how the launcher feeds runtime arguments into that path.
- If behavior depends on the OpenSpiel backend, inspect the matching agent file in `affinetes/environments/openspiel/agents/`.

## When Editing This Repo

- Keep environment-specific logic inside the matching `scripts/envs/<game>_env.py` file unless the change is truly cross-cutting (in which case `scripts/envs/shared_env.py` is the right place).
- When changing reward shaping, check the parser and rollout loop in the same file; reward logic is tightly coupled to observation format.
- When changing curriculum knobs, check both the env file's `_curriculum_factory(...)` and the `ModeConfig` entries in `scripts/envs/env_configs.py`.
- When changing action parsing or legal-action assumptions, confirm the backend observation format in `affinetes/environments/openspiel/`.
- When changing container or orchestration behavior, inspect both the shell scripts and `trainer/image_manager.py`.
- When adding a new environment, see the "Adding a new per-env field" / "Adding a new per-mode field" docstring at the top of `scripts/envs/env_configs.py`.
