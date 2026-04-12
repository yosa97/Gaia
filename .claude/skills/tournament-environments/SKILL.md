---
name: tournament-environments
description: Repo-specific skill for the G.O.D tournament environment training workspace. Use when working on scripts/train_grpo_env.py, scripts/grpo_env_config.py, environment rollout functions, affinetes OpenSpiel backends, reward shaping, curriculum scheduling, or action masking for goof_spiel, gin_rummy, liars_dice, leduc_poker, and related environment-training code.
---

# Tournament Environments

Use this skill when the task touches environment GRPO training, environment-server rollouts, game-specific reward shaping, task-id routing, action masking, or backend OpenSpiel formatting in this repository.

## Quick Start

1. Identify the layer you are changing.
   - Trainer routing or action masking: `scripts/train_grpo_env.py`
   - Hyperparameters or launch command generation: `scripts/grpo_env_config.py`
   - Game-specific rollout/parsing/reward logic: `scripts/*_environment_function.py`
   - Backend observation/legal-action formatting: `affinetes/environments/openspiel/`
   - Upstream game engine internals: `open_spiel/`
   - Containerized orchestration: `run_environment_task.sh`, `run_environment_env.sh`, `trainer/`

2. Confirm the currently active code path.
   - `scripts/model_utility.py:is_reasoning_tokenizer()` currently returns `False`, so the reasoning branch in `scripts/train_grpo_env.py` is not active.
   - `train_grpo_env.py` currently routes `goof_spiel`, `gin_rummy`, `liars_dice`, and `leduc_poker`.

3. Read only the reference file you need.
   - Architecture and end-to-end flow: `references/architecture-and-pipeline.md`
   - Game-by-game function map: `references/environment-functions.md`
   - Config knobs, env vars, and runtime ops: `references/configs-and-ops.md`

## Working Rules

- Treat each environment function file as the source of truth for that game's parser, curriculum, rollout loop, and reward shaping.
- For non-trivial reward, curriculum, parsing, or backend-interface changes, use MCP sequential thinking before editing so the dependency chain is explicit.
- When changing `action_mask`, verify it stays aligned with `completion_ids`; `ActionMaskedGRPOTrainer` validates lengths and shape.
- When changing curriculum, inspect both the environment file and `scripts/grpo_env_config.py`; some rollout schedulers derive values from trainer args.
- When changing observation parsing, verify the corresponding backend formatter in `affinetes/environments/openspiel/agents/`.
- When changing launch/runtime behavior, inspect shell scripts and `trainer/image_manager.py` together.

## Current Gotchas

- `GAMES_TO_TASK_ID_RANGE` contains more environments than `train_grpo_env.py:main()` actually supports today.
- `train_grpo_env.py` constructs the dataset from task-id strings, not normal text prompts.
- `gin_rummy` has extra state augmentation with dead-card and Bayesian summaries.
- `liars_dice` supports both classic multi-dice play and a single-die "liars_die" variant.
- `grpo_env_config.py` is environment-specific; `grpo_config.py` is the separate generic GRPO path.
- Most repo-level game changes should happen in `scripts/` or `affinetes/environments/openspiel/`; `open_spiel/` is the upstream engine layer underneath.

## References

- `references/architecture-and-pipeline.md`
- `references/environment-functions.md`
- `references/configs-and-ops.md`
