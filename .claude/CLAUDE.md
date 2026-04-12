# Tournament Environments Workspace Guide

This repository now includes a skill-style guide under `.claude/skills/tournament-environments/SKILL.md`.
Use this file as the fast entry point, and open the reference files when you need the full code map.

## What This Repo Actually Does

- Trains GRPO-based agents for turn-based game environments.
- Keeps most environment-specific training logic inside `scripts/*_environment_function.py`.
- Uses external environment servers over `/reset` and `/step`, not pure local simulation inside the trainer loop.
- Builds the GRPO training dataset from task-id ranges, so the `prompt` is usually just a stringified task id.
- Relies heavily on curriculum scheduling, reward shaping, action parsing, and action masking.

## Highest-Value Files

- `scripts/train_grpo_env.py`
  Main environment-training entry point. Contains `TrainingArguments`, `ActionMaskedGRPOTrainer`, environment routing, and dataset construction from task-id ranges.
- `scripts/grpo_env_config.py`
  Environment GRPO config presets plus command construction and final training JSON generation.
- `scripts/customized_trainer.py`
  Save/eval callbacks, time-budget logic, generation-config fixes, and resize helpers.
- `scripts/model_utility.py`
  Architecture detection, parameter counting, flash-attention/vLLM/action-mask toggles, and gradient-checkpointing rules.
- `scripts/goof_spiel_environment_function.py`
- `scripts/gin_rummy_environment_function.py`
- `scripts/liars_dice_environment_function.py`
- `scripts/leduc_poker_environment_function.py`
  The core game-specific rollout, parsing, curriculum, and reward-shaping implementations.
- `run_environment_task.sh`
  Local orchestration script that builds images, starts environment servers, runs training containers, and optionally uploads outputs.
- `run_environment_env.sh`
  Boots multiple `agentgym`-style environment servers and writes `ENVIRONMENT_SERVER_URLS`.
- `trainer/`
  Proxy trainer service used by validator/orchestrator flows.
- `affinetes/environments/openspiel/`
  Backend OpenSpiel actor, game config, and game-agent formatting used by the environment server.

## Current Code Truths

- `scripts/model_utility.py:is_reasoning_tokenizer()` currently returns `False`, so the "reasoning tokenizer" branch in `scripts/train_grpo_env.py` is effectively disabled right now.
- `scripts/train_grpo_env.py` exposes task-id ranges for more games than it currently routes in `main()`. The active branches are `goof_spiel`, `gin_rummy`, `liars_dice`, and `leduc_poker`.
- `ActionMaskedGRPOTrainer` expects rollout functions to return `action_mask` aligned exactly with `completion_ids`.
- `scripts/grpo_env_config.py` is for environment GRPO. `scripts/grpo_config.py` is the separate generic text-GRPO config path.
- `train_grpo_env.py` samples up to 200,000 task ids from the selected range and uses those ids as prompts.

## Thinking Guidance

- For non-trivial changes to reward shaping, curriculum, action parsing, or environment-server interaction, use MCP sequential thinking first to map dependencies before editing.
- Especially for `gin_rummy` and `liars_dice`, treat shaping constants and parser assumptions as coupled systems rather than isolated tweaks.

## How To Navigate Changes

1. If the task is about rollout behavior, reward shaping, parsing, or curriculum:
   Read `scripts/<game>_environment_function.py` first.
2. If the task is about trainer behavior, action masking, or environment selection:
   Read `scripts/train_grpo_env.py`.
3. If the task is about hyperparameters, launch flags, or model-specific overrides:
   Read `scripts/grpo_env_config.py` and `scripts/model_utility.py`.
4. If the task is about container startup, environment servers, or validator orchestration:
   Read `run_environment_task.sh`, `run_environment_env.sh`, and `trainer/`.
5. If the task is about how observations/legal actions are produced by the backend:
   Read `affinetes/environments/openspiel/env.py`, `game_config.py`, and the relevant file in `affinetes/environments/openspiel/agents/`.

## Reference Index

- [Skill entry point](skills/tournament-environments/SKILL.md)
- [Architecture and pipeline](skills/tournament-environments/references/architecture-and-pipeline.md)
- [Environment functions](skills/tournament-environments/references/environment-functions.md)
- [Configs and operations](skills/tournament-environments/references/configs-and-ops.md)

## Practical Reading Order

- Start with `scripts/train_grpo_env.py` to understand which environment path is actually active.
- Then open the matching environment function file.
- Then open `scripts/grpo_env_config.py` to see how the launcher feeds runtime arguments into that path.
- If behavior depends on the OpenSpiel backend, inspect the matching agent file in `affinetes/environments/openspiel/agents/`.

## When Editing This Repo

- Keep environment-specific logic inside the matching environment function file unless the change is truly cross-cutting.
- When changing reward shaping, check the parser and rollout loop in the same file; reward logic is tightly coupled to observation format.
- When changing curriculum knobs, check both the environment file and `scripts/grpo_env_config.py`.
- When changing action parsing or legal-action assumptions, confirm the backend observation format in `affinetes/environments/openspiel/`.
- When changing container or orchestration behavior, inspect both the shell scripts and `trainer/image_manager.py`.
