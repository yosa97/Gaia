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

## Current Code Truths

- `scripts/model_utility.py:is_reasoning_tokenizer()` now inspects the tokenizer vocab for `<think>/</think>` and similar pairs. It returns `True` for reasoning-capable tokenizers and enables the reasoning-mode branch in `train_grpo_env.py`. It is no longer hardcoded to `False`.
- `scripts/train_grpo_env.py:main()` routes six environments via `get_env_config(...)`:
  `goof_spiel`, `gin_rummy`, `gin_rummy_opponent_modeling`, `liars_dice`, `leduc_poker`, `alfworld`.
- The trainer picks one of three training modes based on tokenizer/flags:
  - `reasoning` — uses `cfg.rollout_last` + `GRPOTrainer`, `max_completion_length=2048`.
  - `no_mask` — uses `cfg.rollout_last` + `GRPOTrainer`, `max_completion_length=16`.
  - `full_prompt` — uses `cfg.rollout_full` + `ActionMaskedGRPOTrainer`, `max_completion_length=16`.
  Per-mode overrides (`initial_max_turn`, `rollouts_per_stage`, `trainer_class`, `max_completion_length`) come from each env's `ModeConfig` slots.
- `ActionMaskedGRPOTrainer` expects rollout functions to return `action_mask` aligned exactly with `completion_ids`.
- `scripts/grpo_env_config.py` is for environment GRPO. `scripts/grpo_config.py` is the separate generic text-GRPO config path.
- `train_grpo_env.py` samples up to 200,000 task ids from the selected range and uses those ids as prompts.
- The game stack is layered as: `scripts/envs/*_env.py` -> `affinetes/environments/openspiel/` -> upstream `open_spiel/` via `pyspiel` and `open_spiel.python.*`.
- `scripts/envs/__init__.py` re-exports every rollout/reward function with a game-prefixed name (e.g., `gin_rummy_rollout_full_prompt_and_completion_parallelized_curriculum`). Use those names when referring to public entry points.
- `_VARIANT_OVERRIDES` in `envs/env_configs.py` lets a single edit swap `gin_rummy` to `gin_rummy_opponent_modeling` without touching any caller.

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
