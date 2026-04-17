# Configs And Operations

This reference covers the environment-training config path, runtime flags, and operational entry points.

## `scripts/grpo_env_config.py`

### Config buckets

`GRPO_CONFIG` currently contains these buckets:

- `0_1_b` — `lr=3e-5`, `distributed=ddp`, `gpu_count=1`, `batch_size=4`, `gradient_accumulation_steps=6`, `beta=0.02`, `initial_max_turn=1`, `rollouts_per_stage=1280`, `use_lora=True`.
- `1_2_b` — `lr=1e-5`, `gpu_count=1`, `batch_size=3`, `gradient_accumulation_steps=12`, `beta=0.04`, `rollouts_per_stage=1280`.
- `2_4_b` — `lr=1e-5`, `gpu_count=2`, `batch_size=2`, `gradient_accumulation_steps=8`, `beta=0.01`, `rollouts_per_stage=1280`, `use_lora=True`.
- `4_5_b` — `lr=8e-6`, `gpu_count=2`, `batch_size=2`, `gradient_accumulation_steps=8`, `vllm_gpu_memory_utilization=0.35`, `rollouts_per_stage=1280`, `use_lora=True`.
- `5_6_b` — `lr=8e-6`, `gpu_count=2`, `batch_size=2`, `gradient_accumulation_steps=8`, `vllm_gpu_memory_utilization=0.35`, `rollouts_per_stage=1280`, `use_lora=True`.
- `6_9_b` — `lr=8e-6`, `gpu_count=4`, `batch_size=2`, `gradient_accumulation_steps=4`, `vllm_gpu_memory_utilization=0.35`, `rollouts_per_stage=1024`, `use_lora=True`.
- `9_12_b` — `lr=6e-6`, `gpu_count=4`, `batch_size=16`, `vllm_gpu_memory_utilization=0.6`.
- `12_15_b` — `lr=5e-6`, `gpu_count=4`, `batch_size=2`, `vllm_gpu_memory_utilization=0.8`.
- `15_20_b` — `lr=5e-6`, `gpu_count=4`, `batch_size=16`, `vllm_gpu_memory_utilization=0.6`, `use_vllm=False`.
- `20_40_b` — `lr=4e-6`, `gpu_count=8`, `batch_size=16`, `vllm_gpu_memory_utilization=0.6`, `use_vllm=False`, `use_4bit=True`.
- `40_80_b` — `lr=3e-6`, `gpu_count=8`, `batch_size=2`, `vllm_gpu_memory_utilization=0.7`, `use_vllm=False`, `use_4bit=True`.

Every bucket is also labeled with its own key through `config["label"] = key`. All buckets default to `distributed="ddp"` and `beta=0.01` unless noted.

Note: the previous `2_4_b_qwen` bucket and the `Qwen/Qwen2.5-3B-Instruct` override were removed; Qwen 2.5 3B now flows through the default bucket for its size. Per-env generation knobs (`num_generations`, `temperature`, `top_k`) moved out of the size buckets and into `scripts/envs/env_configs.py`.

### Current public functions

- `get_grpo_config(param_nums)` — maps parameter count into a config bucket.
- `get_run_cmd(config, gpu_nums)` — builds the actual `torchrun` / `deepspeed` command line for `train_grpo_env.py`.
- `get_training_json(train_info)` — produces `{"train_request": ..., "run_cmd": ...}`.

### Important current behavior in `get_training_json(train_info)`

- Reads `model_name`, `model_path`, and `dataset_type.environment_name`.
- Calls `get_model_architecture()` and `get_model_num_params()`.
- Applies the Mistral override: `mistralai/Mistral-7B-Instruct-v0.2` and `v0.3` → `6_9_b`.
- Calls `get_env_config(env_name)` from `scripts/envs/env_configs.py` and merges per-env fields:
  - `run_config["num_generations"]      = env_cfg.num_generations      if env_cfg else 4`
  - `run_config["temperature"]          = env_cfg.temperature          if env_cfg else 1.0`
  - `run_config["top_k"]                = env_cfg.top_k                if env_cfg else 0`
  - `run_config["vllm_max_model_length"] = env_cfg.vllm_max_model_length if env_cfg else 5248`
- Derives runtime toggles from `scripts/model_utility.py`:
  - `disable_flash_attention(...)`
  - `disable_action_mask(...)`
  - `get_gradient_checkpointing(...)`
  - `get_use_vllm(...)`
- Multiplies the chosen learning rate by `train_info["reg_ratio"]`.
- Sets operational fields on the train request:
  - `save_before_remaining_time = 10`
  - `min_steps = 100`
  - `adjust_batch_size = False`
  - `periodic_save_steps = 75`
- Special-cases:
  - `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5` → force `use_lora=True`.
  - `starcoder` in the model name → `batch_size //= 1.5`.

### Launch command shape

`get_run_cmd(...)` builds a command around:

- `torchrun --nproc_per_node=<gpu_count>` by default
- `deepspeed` when `distributed == "ds"`

Important CLI flags in the generated command:

- `--request_path`
- `--environment_name`
- `--bf16 True`
- `--report_to wandb`
- `--output_dir /workspace/data/trained_model`
- `--num_train_epochs`
- `--per_device_train_batch_size`
- `--per_device_eval_batch_size`
- `--gradient_accumulation_steps`
- `--eval_accumulation_steps 1`
- `--eval_strategy no`
- `--save_strategy no`
- `--logging_steps 1`
- `--learning_rate`
- `--weight_decay 0.`
- `--warmup_steps 35`
- `--lr_scheduler_type cosine_with_min_lr`
- `--lr_scheduler_kwargs {"min_lr_rate": <min_lr_rate>}`
- `--tf32 True`
- `--gradient_checkpointing`
- `--optim`
- `--use_liger`
- `--num_generations`
- `--vllm_mode colocate`
- `--vllm_gpu_memory_utilization`
- `--temperature`
- `--top_k`
- `--disable_fa`
- `--disable_action_mask`
- `--beta`
- `--loss_type dr_grpo`
- `--num_iterations 2`
- `--do_eval False`
- `--vllm_max_model_length`

Optional additions:

- `--use_peft --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear` when `use_lora=True`.
- `--use_vllm True|False`.
- `--deepspeed ds_config/zero3.json` when `distributed == "ds"`.
- `--vllm_tensor_parallel_size <gpu_count>` when `tensor_parallel=True`.
- `--load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4` when `use_4bit=True`.
- `--initial_max_turn <n>` when the value differs from the default of `2`.
- `--rollouts_per_stage <n>` when the value differs from the default of `1280`.

Note: the old `--rollout_warmup_rollouts` and `--mcts_warmup_optimizer_steps` flags were removed along with the corresponding bucket fields. The old `--num_iterations 1` default is now `2`.

## `scripts/envs/env_configs.py`

Central per-env registry merged on top of the size-bucket config above.

### `ModeConfig`

Per-training-mode overrides for one environment. All fields default to `None`, meaning "use the mode default in `train_grpo_env.py`".

- `initial_max_turn: int | None`
- `rollouts_per_stage: int | None`
- `trainer_class: type | None` — `None` uses the mode default (`GRPOTrainer` for reasoning/no_mask, `ActionMaskedGRPOTrainer` for full_prompt). `reasoning` always uses `GRPOTrainer` regardless.
- `max_completion_length: int | None` — `None` uses the mode default (2048 for reasoning, 16 for no_mask/full_prompt).

### `EnvTrainingConfig`

- `rollout_full: Callable`
- `rollout_last: Callable`
- `reward_func: Callable`
- `curriculum_factory: Callable | None` — `None` means no curriculum for this env.
- `vllm_max_model_length: int = 5248` — reasoning mode adds 2048 at runtime.
- `num_generations: int = 4`
- `temperature: float = 1.0`
- `top_k: int = 0`
- `reasoning: ModeConfig`
- `no_mask: ModeConfig`
- `full_prompt: ModeConfig`

### `_REGISTRY` entries

- `goof_spiel` — `reasoning.initial_max_turn=1`, `no_mask.initial_max_turn=1`, default generation params.
- `gin_rummy` — `reasoning.initial_max_turn=8`, `no_mask.initial_max_turn=4 / rollouts_per_stage=512`, `full_prompt.initial_max_turn=8`, default generation params.
- `gin_rummy_opponent_modeling` — same shape as `gin_rummy`.
- `liars_dice` — all three modes: `rollouts_per_stage=2048, initial_max_turn=1`; env-level `num_generations=8, temperature=2.0, top_k=5`.
- `leduc_poker` — no per-mode overrides; env-level `num_generations=8, temperature=2.0, top_k=5`.
- `alfworld` — defaults only; `curriculum_factory=None`.

### `_VARIANT_OVERRIDES` + `get_env_config`

`_VARIANT_OVERRIDES: dict[str, str]` lets a single edit redirect an env name to a different registry key. The canonical example (currently commented out) is `"gin_rummy": "gin_rummy_opponent_modeling"` to swap the standard gin rummy path for the Bayesian-opponent variant without touching any caller.

`get_env_config(name)` applies `_VARIANT_OVERRIDES` first, then looks up `_REGISTRY`. It raises `ValueError` with the list of known names if the resolved entry is missing.

## `scripts/model_utility.py` Behavior That Affects Training

Important functions and current operational consequences:

- `is_reasoning_tokenizer(tokenizer)`
  Inspects the tokenizer vocab for `<think>/</think>`, `<thinking>/</thinking>`, and similar pairs. Returns `True` if any pair is present in the vocab, which activates the reasoning-mode branch in `train_grpo_env.py` (rollout_last + `GRPOTrainer` + `max_completion_length=2048`).
- `disable_action_mask(model)`
  Disables masking for:
  - `codellama/CodeLlama-7b-Instruct-hf`
  - `deepseek-ai/deepseek-coder-6.7b-instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `mistralai/Mistral-7B-Instruct-v0.2`
- `disable_flash_attention(architecture, model)`
  Disables FA for specific architectures and models such as Phi-2 and Falcon-RW.
- `get_use_vllm(architecture, model)`
  Disables vLLM for specific unsupported models and architectures.

## Environment Variables Used By The Environment Paths

Shared/common:

- `ENVIRONMENT_SERVER_URLS`
  Comma-separated list of backend server base URLs.
- `LOCAL_RANK`
  Used for distributed rank-local server selection and output behavior.

General training/runtime:

- `WANDB_TOKEN`
- `PYTORCH_CUDA_ALLOC_CONF`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_USERNAME`

Game-specific env vars (`LIARS_DICE_RULESET`, `LIARS_DICE_FINAL_MAX_TURN`, `LIARS_DICE_INITIAL_HINT_PROB`, `LIARS_DICE_FINAL_HINT_PROB`, `EPISODE_TRACE_*`) that the pre-refactor `liars_dice_environment_function.py` read were removed during the refactor; `scripts/envs/liar_dice_env.py` does not reference them.

## Shell Operations

### `run_environment_env.sh`

Current behavior:

- creates Docker network `agent_eval_net`
- starts 4 containers:
  - `agentgym-server-0`
  - `agentgym-server-1`
  - `agentgym-server-2`
  - `agentgym-server-3`
- uses image `phoenixbeaudry/game:mcts-api`
- writes comma-separated internal URLs into `.environment_server_urls.txt`

### `run_environment_task.sh`

Current behavior:

- parses `--gpus`
- builds:
  - `trainer-downloader:latest`
  - `standalone-text-trainer:latest`
- starts environment servers
- downloads model and dataset assets into cache
- runs the GRPO text trainer container with:
  - mounted checkpoints/cache volumes
  - `ENVIRONMENT_SERVER_URLS`
  - `WANDB_TOKEN`
  - `PYTORCH_CUDA_ALLOC_CONF`
- streams container logs with timeout handling
- uploads outputs through `trainer/utils/hf_upload.py` if the output directory exists

## Trainer Service Files

### `trainer/endpoints.py`

Important current routes:

- `start_training(req)`
- `get_available_gpus()`
- `get_task_details(task_id, hotkey)`
- `get_recent_tasks_list(hours)`
- `factory_router()`

Operational behavior:

- verifies orchestrator IPs through `verify_orchestrator_ip(...)`
- clones the repo via `clone_repo(...)`
- starts asynchronous training through `start_training_task(...)`

### `trainer/image_manager.py`

Important current functions:

- `ensure_internal_network(...)`
- `calculate_container_resources(gpu_ids)`
- `build_docker_image(...)`
- `delete_image_and_cleanup(tag)`
- `wait_for_env_container_ip(environment_server_container)`
- `run_trainer_container_image(...)`
- `run_trainer_container_text(...)`
- `create_volumes_if_dont_exist()`
- `run_downloader_container(...)`
- `run_environment_server_container(environment_name, log_labels)`
- `upload_repo_to_hf(...)`
- `get_task_type(request)`
- `get_dockerfile_path(task_type, training_data, local_repo_path)`
- `start_training_task(task, local_repo_path)`

### `trainer/tasks.py`

Important task-lifecycle functions:

- `start_task(...)`
- `complete_task(...)`
- `get_task(...)`
- `log_task(...)`
- `update_wandb_url(...)`
- `get_running_tasks()`
- `get_recent_tasks(hours=1.0)`
- `save_task_history()`
- `load_task_history()`

## Backend Files Worth Checking During Ops Bugs

- `affinetes/environments/openspiel/env.py`
  Training interface and MCTS opponent handling.
- `affinetes/environments/openspiel/game_config.py`
  Task-id decoding and game creation.
- `affinetes/environments/openspiel/agents/gin_rummy.py`
- `affinetes/environments/openspiel/agents/liars_dice_agent.py`
- `affinetes/environments/openspiel/agents/leduc_poker_agent.py`
- `affinetes/environments/openspiel/agents/goofspiel.py`

If observations, legal actions, or task ids look wrong, these backend files are usually the right second stop after the env file in `scripts/envs/`.
