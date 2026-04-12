# Architecture And Pipeline

This reference maps the main code paths for environment GRPO training in the current repository.

## End-To-End Flow

1. `run_environment_task.sh`
   Builds downloader/trainer Docker images, starts environment servers via `run_environment_env.sh`, downloads model and dataset assets, launches the training container, streams logs, and optionally uploads the final model.
2. `run_environment_env.sh`
   Starts 4 environment containers named `agentgym-server-0..3`, exposes them on the internal Docker network, and writes the comma-separated `ENVIRONMENT_SERVER_URLS` file that the trainer consumes.
3. `scripts/grpo_env_config.py`
   Converts model size and task metadata into a launch config and then into the actual `torchrun ... train_grpo_env.py` command line.
4. `scripts/train_grpo_env.py`
   Loads request JSON, builds tokenizer/model/PEFT config, creates a synthetic GRPO dataset from task-id ranges, selects the environment-specific rollout and reward function, and starts training.
5. `scripts/*_environment_function.py`
   Resets episodes via `/reset`, steps the environment via `/step`, parses observations, computes shaping rewards, manages curriculum state, and returns rollout artifacts such as `prompt_ids`, `completion_ids`, `logprobs`, and optionally `action_mask`.
6. `affinetes/environments/openspiel/`
   Produces the backend game observations, legal actions, and MCTS opponent behavior that the rollout functions depend on.

## Main Training Entry Point

### `scripts/train_grpo_env.py`

Current important symbols:

- `GAMES_TO_TASK_ID_RANGE`
  Global task-id ranges for all game families.
- `TrainingArguments(GRPOConfig)`
  Adds environment-specific flags such as `request_path`, `disable_action_mask`, `initial_max_turn`, `rollouts_per_stage`, `rollout_warmup_rollouts`, `mcts_warmup_optimizer_steps`, and `environment_name`.
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
  - applies the gin-rummy vLLM IS-mode override to `token_truncate`
  - selects rollout/reward function pair by `environment_name`
  - selects `GRPOTrainer` or `ActionMaskedGRPOTrainer`
  - attaches `GRPOCustomEvalSaveCallback`
  - calls `trainer.train()`

Important current behavior:

- The active supported environment names in `main()` are `goof_spiel`, `gin_rummy`, `liars_dice`, and `leduc_poker`.
- `is_reasoning_tokenizer()` currently returns `False`, so the reasoning-tokenizer branch is effectively inactive.
- Default non-reasoning environment training uses `ActionMaskedGRPOTrainer`.
- `max_completion_length` is usually `16` for the active non-reasoning branch.
- The dataset is synthetic: each sample is `{"prompt": str(task_id)}`.
- The trainer caps the sampled task-id dataset at 200,000 examples.

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
  Currently hardcoded to `False`.
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
  Produces the final `train_request` and `run_cmd`.

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
