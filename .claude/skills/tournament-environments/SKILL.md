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
- Preserve the default tournament opponent as MCTS unless the task explicitly says otherwise.
- Use these target MCTS settings as the default rule:
  - Gin Rummy: `MCTS(50,1)` — IS-MCTS, progressive warmup from 10 sims
  - Liars Dice: `MCTS(225,1)` — classic UCT, highest sim count
  - Leduc Poker: `MCTS(50,1)` — classic UCT, progressive warmup from 10 sims
- The MCTS opponent uses `SafeRandomRolloutEvaluator(n_rollouts=1)` with `uct_c=1.414` — NOT a neural network. This means:
  - Very high evaluation variance per node (1 random rollout only)
  - No Nash equilibrium convergence at low sim counts
  - No bluff/history tracking across episodes
  - Strategy hints in `_HINT_PROMPT` / `STRATEGY_TIPS_*` encode how to exploit these weaknesses
- `trainer.args.initial_max_turn` stores the MCTS sim count (set by `train_grpo_env.py`), NOT a curriculum turn. Each env file uses its own `CURRICULUM_INITIAL_TURN` constant to avoid this naming collision.
- When changing `action_mask`, verify it stays aligned with `completion_ids`; `ActionMaskedGRPOTrainer` validates lengths and shape.
- When changing curriculum, inspect both the environment file and `scripts/grpo_env_config.py`; some rollout schedulers derive values from trainer args.
- When changing observation parsing, verify the corresponding backend formatter in `affinetes/environments/openspiel/agents/`.
- When changing launch/runtime behavior, inspect shell scripts and `trainer/image_manager.py` together.
- When changing strategy hints, always update BOTH the code file AND this skill doc.

## Hint Curriculum System (all 3 active games)

All three games now use an active hint curriculum:

| Game | Initial hint prob | Final hint prob | Hint type |
|------|------------------|-----------------|-----------|
| Gin Rummy | 0.5 | 0.0 | `_HINT_PROMPT` module constant |
| Liars Dice | 0.5 | 0.0 | `STRATEGY_TIPS_CLASSIC` / `STRATEGY_TIPS_LIARS_DIE` |
| Leduc Poker | 0.5 | 0.0 | `_HINT_PROMPT` module constant |

Hint decay: Gin Rummy uses optimizer steps; Liars Dice and Leduc Poker use rollout count.
Liars Dice: hint probs are also overridable via `LIARS_DICE_INITIAL_HINT_PROB` and `LIARS_DICE_FINAL_HINT_PROB` env vars.

## Curren Models used in tournament enviroment
- Qwen/Qwen2-7B-Instruct
- unsloth/Llama-3.2-3B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- mistralai/Mistral-7B-Instruct-v0.2
- Qwen/Qwen2.5-3B-Instruct
- Qwen/Qwen2.5-7B-Instruct
- codellama/CodeLlama-7b-Instruct-hf
- NousResearch/Hermes-3-Llama-3.2-3B
## Current Gotchas

- `GAMES_TO_TASK_ID_RANGE` contains more environments than `train_grpo_env.py:main()` actually supports today.
- `train_grpo_env.py` constructs the dataset from task-id strings, not normal text prompts.
- `gin_rummy` has extra state augmentation with dead-card and Bayesian summaries.
- `liars_dice` supports both classic multi-dice play and a single-die "liars_die" variant.
- `grpo_env_config.py` is environment-specific; `grpo_config.py` is the separate generic GRPO path.
- Most repo-level game changes should happen in `scripts/` or `affinetes/environments/openspiel/`; `open_spiel/` is the upstream engine layer underneath.
- Agent files (`affinetes/environments/openspiel/agents/`) contain high server-side MCTS defaults (e.g., `(3000,200)` for Leduc). The training payload always overrides these with lower values — trust the `MCTS_CONFIG` in the environment function file, not the agent file defaults.

## References

- `references/architecture-and-pipeline.md`
- `references/environment-functions.md`
- `references/configs-and-ops.md`
