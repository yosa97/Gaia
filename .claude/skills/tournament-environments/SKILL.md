---
name: tournament-environments
description: Repo-specific skill for the G.O.D tournament environment training workspace. Use when working on scripts/train_grpo_env.py, scripts/grpo_env_config.py, the scripts/envs/ package (rollout functions, reward shaping, curriculum scheduling, action masking), OpenSpiel backends, or variant routing for goof_spiel, gin_rummy, gin_rummy_opponent_modeling, liars_dice, leduc_poker, and alfworld.
---

# Tournament Environments

Use this skill when the task touches environment GRPO training, environment-server rollouts, game-specific reward shaping, task-id routing, action masking, or backend OpenSpiel formatting in this repository.

## Quick Start

1. Identify the layer you are changing.
   - Trainer routing, action masking, or training-mode dispatch: `scripts/train_grpo_env.py`
   - Environment registry, per-env defaults, per-mode overrides: `scripts/envs/env_configs.py`
   - Shared curriculum base class, task-id ranges, env-pool init: `scripts/envs/shared_env.py`
   - Game-specific rollout/parsing/reward logic: `scripts/envs/<game>_env.py` (or `gin_rummy_opponent_modeling.py`)
   - Size-bucket hyperparameters and launch command: `scripts/grpo_env_config.py`
   - Backend observation/legal-action formatting: `affinetes/environments/openspiel/`
   - Upstream game engine internals: `open_spiel/`
   - Containerized orchestration: `run_environment_task.sh`, `run_environment_env.sh`, `trainer/`

2. Confirm the currently active code path.
   - `scripts/model_utility.py:is_reasoning_tokenizer()` now inspects the tokenizer vocab for `<think>/</think>` pairs. Reasoning-capable tokenizers activate the reasoning-mode branch in `train_grpo_env.py`.
   - `train_grpo_env.py:main()` routes six environments via `get_env_config(...)`: `goof_spiel`, `gin_rummy`, `gin_rummy_opponent_modeling`, `liars_dice`, `leduc_poker`, `alfworld`.
   - Three training modes are dispatched based on tokenizer/flags:
     - `reasoning` — `rollout_last`, `GRPOTrainer`, `max_completion_length=2048`.
     - `no_mask` — `rollout_last`, `GRPOTrainer`, `max_completion_length=16`.
     - `full_prompt` — `rollout_full`, `ActionMaskedGRPOTrainer`, `max_completion_length=16`.

3. Read only the reference file you need.
   - Architecture and end-to-end flow: `references/architecture-and-pipeline.md`
   - Game-by-game function map: `references/environment-functions.md`
   - Config knobs and runtime ops: `references/configs-and-ops.md`

## Working Rules

- Treat each env file under `scripts/envs/` as the source of truth for that game's parser, curriculum factory, rollout loop, and reward shaping.
- For non-trivial reward, curriculum, parsing, or backend-interface changes, use MCP sequential thinking before editing so the dependency chain is explicit.
- Preserve the default tournament opponent as MCTS unless the task explicitly says otherwise.
- MCTS sim counts are set per-game inside each env's `_ensure_initialized()` (or at module scope for the opponent-modeling variant). The shared `CurriculumScheduler` has **no** `get_mcts_sims()` method; progressive warmup is not part of the current refactored path.
- Current MCTS defaults (fixed, no warmup ramp):
  - Gin Rummy (`gin_rummy_env.py`): `MCTS(25, 1)`.
  - Gin Rummy opponent-modeling (`gin_rummy_opponent_modeling.py`): `MCTS(25, 1)` via module-level `_MCTS_SIMS = 25`.
  - Liars Dice (`liar_dice_env.py`): `MCTS(225, 1)`.
  - Leduc Poker (`leduc_poker_env.py`): `MCTS(50, 1)`.
  - Goof Spiel and AlfWorld: no MCTS opponent.
- MCTS evaluator is still `SafeRandomRolloutEvaluator(n_rollouts=1)` with `uct_c=1.414` (random playouts, not neural-net priors). Strategy hints in `_HINT_PROMPT` / equivalent encode how to exploit these weaknesses:
  - High evaluation variance per node (1 random rollout only).
  - No Nash equilibrium convergence at low sim counts.
  - No bluff/history tracking across episodes.
- `trainer.args.initial_max_turn` is a curriculum turn count. It may be overridden per-mode via the relevant `ModeConfig` in `scripts/envs/env_configs.py`.
- When changing `action_mask`, verify it stays aligned with `completion_ids`; `ActionMaskedGRPOTrainer` validates lengths and shape.
- When changing curriculum, inspect both the env file's `_curriculum_factory(...)` and the matching `ModeConfig` in `scripts/envs/env_configs.py`.
- When changing observation parsing, verify the corresponding backend formatter in `affinetes/environments/openspiel/agents/`.
- When changing launch/runtime behavior, inspect the shell scripts and `trainer/image_manager.py` together.
- When changing strategy hints, always update BOTH the code file AND this skill doc.

## Hint Curriculum System

All five game envs participate in a hint curriculum; `alfworld` does not.

| Env file | Initial hint prob | Final hint prob | Hint source |
|---|---|---|---|
| `scripts/envs/gin_rummy_env.py` | 0.5 | 0.0 | file-local hint prompt |
| `scripts/envs/gin_rummy_opponent_modeling.py` | 0.5 | 0.0 | file-local hint prompt |
| `scripts/envs/liar_dice_env.py` | 0.5 | 0.0 | file-local strategy tips |
| `scripts/envs/leduc_poker_env.py` | 0.75 | 0.0 | `_HINT_PROMPT` module constant |
| `scripts/envs/goof_spiel_env.py` | 0.75 | 0.0 | file-local hint |
| `scripts/envs/alf_world_env.py` | — | — | no curriculum, no hints |

Hint decay is driven by the shared `CurriculumScheduler.get_hint_prob()` in `scripts/envs/shared_env.py` (override per-game via subclass if needed). Env-var overrides for hint probabilities (`LIARS_DICE_INITIAL_HINT_PROB`, etc.) were dropped in the refactor; set probabilities in the env's `_curriculum_factory` instead.

## Current Models Used In Tournament Environment

- Qwen/Qwen2-7B-Instruct
- unsloth/Llama-3.2-3B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- mistralai/Mistral-7B-Instruct-v0.2
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- codellama/CodeLlama-7b-Instruct-hf
- NousResearch/Hermes-3-Llama-3.2-3B

## Current Gotchas

- `GAMES_TO_TASK_ID_RANGE` in `scripts/envs/shared_env.py` contains more entries than `_REGISTRY` in `scripts/envs/env_configs.py` actually routes. Only the six registry entries are live.
- `train_grpo_env.py` constructs the dataset from task-id strings, not normal text prompts.
- `gin_rummy_env.py` is the leaner default path; `gin_rummy_opponent_modeling.py` is the Bayesian-opponent variant that ships with the `DeadCardTracker`, `BayesianOpponentModel`, and `BayesianOpponentHandModel` classes plus the extended reward coefficient block.
- `liar_dice_env.py` exposes a risky-play bonus scheme (`BLUFF_WIN_BONUS`, `RISKY_LIAR_WIN_BONUS`, `BLUFF_PROB_THRESHOLD`, etc.). The old `PASS_MISSED_CHALLENGE_PENALTY`/`BID_PLAUSIBILITY_*` constants were removed in the refactor — don't reintroduce them without checking the new shaping path.
- The two-layer config pattern is:
  1. `scripts/grpo_env_config.py:GRPO_CONFIG` provides size-bucket defaults (lr, batch size, beta, etc.).
  2. `scripts/envs/env_configs.py:_REGISTRY` provides per-env defaults (`vllm_max_model_length`, `num_generations`, `temperature`, `top_k`) and per-mode `ModeConfig` overrides.
  3. `get_training_json(...)` merges the per-env/per-mode values on top of the size-bucket before building the CLI.
  `_VARIANT_OVERRIDES` in `env_configs.py` can swap an env name (e.g., `gin_rummy` → `gin_rummy_opponent_modeling`) in a single edit.
- `grpo_env_config.py` is environment-specific; `grpo_config.py` is the separate generic GRPO path.
- Most repo-level game changes should happen in `scripts/envs/` or `affinetes/environments/openspiel/`; `open_spiel/` is the upstream engine layer underneath.
- Agent files (`affinetes/environments/openspiel/agents/`) contain high server-side MCTS defaults (e.g., `(3000,200)` for Leduc). The training payload always overrides these with lower values — trust the MCTS sim count set inside each env's `_ensure_initialized()`, not the agent file defaults.
- `scripts/legacy/` and `scripts/legacy2/` contain the old `*_environment_function.py` files. They are not routed; use them only for historical reference.

## References

- `references/architecture-and-pipeline.md`
- `references/environment-functions.md`
- `references/configs-and-ops.md`
