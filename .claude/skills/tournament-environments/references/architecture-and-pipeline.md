# Architecture And Pipeline

This reference maps the main code paths for environment GRPO training in the current repository.

## End-To-End Flow

1. `run_environment_task.sh`
   Builds downloader/trainer Docker images, starts environment servers via `run_environment_env.sh`, downloads model and dataset assets, launches the training container, streams logs, and optionally uploads the final model.
2. `run_environment_env.sh`
   Starts 4 environment containers named `agentgym-server-0..3`, exposes them on the internal Docker network, and writes the comma-separated `ENVIRONMENT_SERVER_URLS` file that the trainer consumes.
3. `scripts/grpo_env_config.py`
   Converts model size and task metadata into a launch config (merging with `scripts/envs/env_configs.py`) and then into the actual `torchrun ... train_grpo_env.py` command line.
4. `scripts/train_grpo_env.py`
   Loads request JSON, builds tokenizer/model/PEFT config, creates a synthetic GRPO dataset from task-id ranges, looks up the env training config via `get_env_config(...)`, dispatches by training mode, and starts training.
5. `scripts/envs/*_env.py`
   Reset episodes via `/reset`, step the environment via `/step`, parse observations, compute shaping rewards, manage curriculum state via each file's `_curriculum_factory`, and return rollout artifacts such as `prompt_ids`, `completion_ids`, `logprobs`, and optionally `action_mask`.
6. `affinetes/environments/openspiel/`
   Produces the backend game observations, legal actions, and MCTS opponent behavior that the rollout functions depend on.
7. `open_spiel/`
   Provides the upstream game engine, `pyspiel`, and `open_spiel.python.*` modules used by the Affinetes OpenSpiel wrapper.

## Layer Relationship

For the active board/card/dice games in this repo, the practical stack is:

1. `scripts/envs/<game>_env.py` (or `gin_rummy_opponent_modeling.py`)
   Training-side rollout, shaping, parsing, and per-env curriculum factory.
2. `scripts/envs/shared_env.py`
   Shared `CurriculumScheduler` base, task-id ranges, env-pool init.
3. `scripts/envs/env_configs.py`
   Registry mapping env names to `EnvTrainingConfig` (rollout callables, curriculum factory, per-mode overrides).
4. `affinetes/environments/openspiel/`
   Environment-server wrapper that formats observations, maps task ids, constructs agents, and runs MCTS or other opponents.
5. `open_spiel/`
   Upstream engine and Python bindings such as `pyspiel` and `open_spiel.python.algorithms.mcts`.

## Main Training Entry Point

### `scripts/train_grpo_env.py`

Current important symbols:

- `TrainingArguments(GRPOConfig)`
  Adds environment-specific flags such as `request_path`, `disable_action_mask`, `initial_max_turn`, `rollouts_per_stage`, `environment_name`, `use_liger`, and `disable_fa`.
- `print_trainable_parameters(model)`
  Logs trainable parameter counts, including LoRA and embedding/lm-head splits.
- `ActionMaskedGRPOTrainer(GRPOTrainer)`
  Overrides `_generate_and_score_completions()` to:
  - call the rollout function
  - pad prompt/completion ids
  - validate `action_mask`
  - build `loss_mask = completion_mask * action_mask`
  - keep importance-sampling and metrics aligned with masked tokens
- `main()`
  Core responsibilities:
  - parses `TrainingArguments` and `ModelConfig`
  - loads `train_request` from `request_path`
  - loads tokenizer/model and optional LoRA adapter
  - builds `train_ds` from stringified task ids
  - sets environment-specific `max_steps`
  - calls `cfg = get_env_config(training_args.environment_name)` to fetch the `EnvTrainingConfig`
  - dispatches to one of three training modes (see below) and applies the matching `ModeConfig` overrides
  - instantiates `GRPOTrainer` or `ActionMaskedGRPOTrainer` depending on the mode
  - attaches `GRPOCustomEvalSaveCallback`
  - calls `trainer.train()`

### Training Modes

`train_grpo_env.py:main()` picks exactly one of these branches:

| Mode | Condition | Rollout | Trainer class | `max_completion_length` default |
|---|---|---|---|---|
| `reasoning` | `is_reasoning_tokenizer(tokenizer)` returns True | `cfg.rollout_last` | `GRPOTrainer` (always) | 2048 |
| `no_mask` | `--disable_action_mask True` (or model in `disable_action_mask` list) | `cfg.rollout_last` | `GRPOTrainer` (by default) | 16 |
| `full_prompt` | otherwise (action mask enabled) | `cfg.rollout_full` | `ActionMaskedGRPOTrainer` | 16 |

Each env's `ModeConfig` slot (`cfg.reasoning`, `cfg.no_mask`, `cfg.full_prompt`) can override `initial_max_turn`, `rollouts_per_stage`, `trainer_class`, and `max_completion_length`. Unset fields use the mode default above.

Important current behavior:

- Active supported environment names are `goof_spiel`, `gin_rummy`, `gin_rummy_opponent_modeling`, `liars_dice`, `leduc_poker`, `alfworld`.
- `is_reasoning_tokenizer()` now inspects the tokenizer vocab for `<think>/</think>`, `<thinking>/</thinking>`, etc. and returns `True` when the tokenizer recognizes any of those pairs.
- The dataset is synthetic: each sample is `{"prompt": str(task_id)}`.
- The trainer caps the sampled task-id dataset at 200,000 examples.

## Environment Registry

### `scripts/envs/env_configs.py`

Central lookup for per-env training config.

- `ModeConfig`
  Per-training-mode overrides. All fields default to `None`, meaning "use the mode-level default from `train_grpo_env.py`":
  - `initial_max_turn: int | None`
  - `rollouts_per_stage: int | None`
  - `trainer_class: type | None`
  - `max_completion_length: int | None`
- `EnvTrainingConfig`
  Full per-env config:
  - `rollout_full: Callable` — used by `full_prompt` mode
  - `rollout_last: Callable` — used by `reasoning` and `no_mask` modes
  - `reward_func: Callable`
  - `curriculum_factory: Callable | None` — returns a `CurriculumScheduler` (or subclass) given training args; `None` means no curriculum
  - `vllm_max_model_length: int = 5248` — initial context window; reasoning mode adds 2048 at runtime
  - `num_generations: int = 4`
  - `temperature: float = 1.0`
  - `top_k: int = 0`
  - `reasoning: ModeConfig`, `no_mask: ModeConfig`, `full_prompt: ModeConfig` — per-mode overrides
- `_REGISTRY` entries:
  - `goof_spiel` — `reasoning.initial_max_turn=1`, `no_mask.initial_max_turn=1`
  - `gin_rummy` — `reasoning.initial_max_turn=8`, `no_mask.initial_max_turn=4 / rollouts_per_stage=512`, `full_prompt.initial_max_turn=8`
  - `gin_rummy_opponent_modeling` — same shape as `gin_rummy`
  - `liars_dice` — all three modes: `rollouts_per_stage=2048, initial_max_turn=1`; env-level `num_generations=8, temperature=2.0, top_k=5`
  - `leduc_poker` — env-level `num_generations=8, temperature=2.0, top_k=5`; no per-mode overrides
  - `alfworld` — defaults only; `curriculum_factory=None`
- `_VARIANT_OVERRIDES: dict[str, str]`
  Optional swap table. Currently commented out. Example usage: set `{"gin_rummy": "gin_rummy_opponent_modeling"}` to redirect every `gin_rummy` lookup to the Bayesian-opponent variant.
- `get_env_config(name: str) -> EnvTrainingConfig`
  Applies `_VARIANT_OVERRIDES` then looks up `_REGISTRY`. Raises `ValueError` for unknown names.

### `scripts/envs/shared_env.py`

- `GAMES_TO_TASK_ID_RANGE` — master task-id-range table (kept in `shared_env` rather than `train_grpo_env.py`).
- `CurriculumScheduler` — base class with `get_max_turn()`, `get_hint_prob()`, `step(num_rollouts)`, and `get_status()`. Subclassed per-game.
- `init_env_pool(reset_payload, reset_endpoint="reset", lock_per_server=False)` — initialize the environment server pool once per process.
- `rollout_reward_func(completions, **kwargs)` — generic passthrough used by every game's re-export.

### `scripts/envs/__init__.py`

Exports every rollout/reward function with a **game-prefixed name**. The public surface is:

- `goof_spiel_rollout_first_prompt_and_completion`, `goof_spiel_rollout_full_prompt_and_completion_parallelized_curriculum`, `goof_spiel_rollout_last_prompt_and_completion_parallelized_curriculum`, `goof_spiel_rollout_reward_func`
- `gin_rummy_rollout_full_prompt_and_completion_parallelized_curriculum`, `gin_rummy_rollout_last_prompt_and_completion_parallelized_curriculum`, `gin_rummy_rollout_reward_func`
- `liar_dice_rollout_full_prompt_and_completion_parallelized_curriculum`, `liar_dice_rollout_last_prompt_and_completion_parallelized_curriculum`, `liar_dice_rollout_reward_func`
- `leduc_poker_rollout_full_prompt_and_completion_parallelized_curriculum`, `leduc_poker_rollout_last_prompt_and_completion_parallelized_curriculum`, `leduc_poker_rollout_reward_func`
- `alfworld_rollout_first_prompt_and_completion_parallelized`, `alfworld_rollout_full_prompt_and_completion_parallelized`, `alfworld_rollout_reward_func`
- `EnvTrainingConfig`, `get_env_config`, `GAMES_TO_TASK_ID_RANGE`

Note: `gin_rummy_opponent_modeling` is wired through `env_configs.py` directly and is not re-exported under a prefixed name in `__init__.py`; it's used via `get_env_config("gin_rummy_opponent_modeling")`.

## Trainer Helpers

### `scripts/customized_trainer.py`

Current important classes and functions:

- `CustomEvalSaveCallback`
  Decides when to evaluate/save, handles time-budgeted stopping, and copies the best checkpoint into the submission directory.
- `GRPOCustomEvalSaveCallback`
  Uses `eval_reward` from `state.log_history` as the selection signal and negates it into a loss-like value.
- `WhenToEvalHandler`
  Triggers eval/save on epoch boundaries, periodic-save steps, low remaining time, or `max_steps`.
- `set_generation_config(model_name, model)`
  Patches generation config for known problematic models.
- `resize_if_needed(model_name, model, token_nums)`
  Resizes token embeddings for models with vocab-size mismatch issues.
- `init_wandb(train_request)`
  Currently returns immediately and does not actively initialize WandB.

### `scripts/model_utility.py`

Current important functions:

- `is_reasoning_tokenizer(tokenizer)`
  Inspects the tokenizer vocab for reasoning-tag pairs (`<think>/</think>`, `<thinking>/</thinking>`, etc.). Returns `True` if any pair is present in the vocab. Activates the reasoning-mode branch in `train_grpo_env.py`.
- `get_model_architecture(model_path)`
  Reads architecture from `AutoConfig`.
- `get_model_num_params(model_id, model_path)`
  Uses hardcoded lookup, local shard counting, or regex fallback.
- `disable_flash_attention(architecture, model)`
  Applies model-specific FA disable rules.
- `disable_action_mask(model)`
  Disables action masking for known problematic BPE tokenizer models.
- `get_use_vllm(architecture, model)`
  Applies model/architecture-specific vLLM allowlist and denylist logic.
- `get_gradient_checkpointing(model)`
  Disables gradient checkpointing for Falcon-RW variants.

## Config Entry Point

### `scripts/grpo_env_config.py`

Current important symbols:

- `GRPO_CONFIG`
  Size-bucket presets for environment GRPO.
- `get_grpo_config(param_nums)`
  Maps parameter count to a bucket.
- `get_run_cmd(config, gpu_nums)`
  Builds the `torchrun` or `deepspeed` launch command for `train_grpo_env.py`.
- `get_training_json(train_info)`
  Produces the final `train_request` and `run_cmd`. Calls `get_env_config(env_name)` from `scripts/envs/env_configs.py` and merges per-env `vllm_max_model_length`, `num_generations`, `temperature`, `top_k`, and per-mode `initial_max_turn` / `rollouts_per_stage` on top of the size-bucket config before building the CLI.

This file is environment-specific. The similarly named `scripts/grpo_config.py` is the separate generic GRPO path for text tasks.

## Runtime Orchestration

### Shell Scripts

- `run_environment_task.sh`
  Handles local end-to-end runs, including downloader image build, trainer image build, environment-server startup, container launch, timeout handling, and optional Hugging Face upload.
- `run_environment_env.sh`
  Starts four environment servers and writes `.environment_server_urls.txt`.

### `trainer/`

Important files:

- `trainer/endpoints.py`
  FastAPI routes for start training, GPU availability, recent tasks, and task details.
- `trainer/image_manager.py`
  Builds images, starts training containers, starts environment server containers, waits for IPs, uploads repos to HF, and chooses Dockerfiles by task type.
- `trainer/tasks.py`
  In-memory plus persisted task-history management.
- `trainer/utils/trainer_downloader.py`
  Downloads datasets/base models and writes proxy datasets for environment tasks.
- `trainer/utils/hf_upload.py`
  Syncs WandB logs and uploads the trained model directory to Hugging Face.

## Backend Environment Layer

### `affinetes/environments/openspiel/env.py`

Current important classes:

- `EpisodeState`
  Per-episode training state for the OpenSpiel actor.
- `SafeRandomRolloutEvaluator`
  Safe MCTS evaluator that guards against empty legal-action edge cases.
- `TimedMCTSBot`
  Wraps MCTS to record time spent in MCTS calls.
- `Actor`
  Provides the OpenEnv training interface and evaluation interface.

Important current `Actor` methods:

- `_format_observation(...)`
- `_parse_action(...)`
- `_auto_play_opponents(...)`
- `_create_training_opponent_bot(...)`
- `reset(...)`
- `step(...)`
- `state(...)`
- `stop(...)`
- `evaluate(...)`
- `_run_evaluation(...)`
- `_compute_score(...)`
- `_create_opponent_bot(...)`
- `_build_result(...)`

### Other backend files

- `affinetes/environments/openspiel/base_agent.py`
  Defines the game-agent interface used to format rules, state, params, and prompts.
- `affinetes/environments/openspiel/game_config.py`
  Decodes task ids and creates game instances.
- `affinetes/environments/openspiel/llm_bot.py`
  LLM bot wrapper for evaluation and action parsing.
- `affinetes/environments/openspiel/agents/gin_rummy.py`
- `affinetes/environments/openspiel/agents/liars_dice_agent.py`
- `affinetes/environments/openspiel/agents/leduc_poker_agent.py`
- `affinetes/environments/openspiel/agents/goofspiel.py`
  These agent files are the best place to inspect how the backend formats rules and observations for the corresponding game.

### Upstream engine files

- `open_spiel/`
  Separate submodule from Google DeepMind.
- In the current repo architecture, usually not the first place to edit.
- The Affinetes wrapper imports from it directly, for example `import pyspiel` and `from open_spiel.python.algorithms import mcts`.

## Other Training Code In This Repo

These files exist in the workspace but are not the main environment-GRPO path:

- `scripts/train_grpo.py`
- `scripts/train_dpo.py`
- `scripts/train_instruct.py`
- `scripts/tokenize_grpo.py`
- `scripts/tokenize_dpo.py`
- `scripts/tokenize_instruct.py`
- `scripts/job_handler.py`
- `scripts/core/`

Open them when the task crosses from environment training into the generic text/DPO/instruct infrastructure.

## Legacy Files

`scripts/legacy/` and `scripts/legacy2/` contain pre-refactor copies of the old `*_environment_function.py` files. They are not routed through `get_env_config(...)` and should be treated as read-only historical references. All active code lives under `scripts/envs/`.
