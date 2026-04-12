# Configs And Operations

This reference covers the environment-training config path, runtime flags, and operational entry points.

## `scripts/grpo_env_config.py`

### Config buckets

`GRPO_CONFIG` currently contains these buckets:

- `0_1_b`
  `lr=3e-5`, `gpu_count=1`, `batch_size=4`, `gradient_accumulation_steps=6`, `num_generations=4`, `beta=0.02`, `initial_max_turn=1`, `rollouts_per_stage=1280`
- `1_2_b`
  `lr=8e-6`, `gpu_count=1`, `batch_size=3`, `gradient_accumulation_steps=12`, `num_generations=4`, `beta=0.04`
- `2_4_b`
  `lr=1e-5`, `gpu_count=2`, `batch_size=1`, `gradient_accumulation_steps=16`, `num_generations=8`, `beta=0.01`, `rollout_warmup_rollouts=0`, `mcts_warmup_optimizer_steps=20`
- `2_4_b_qwen`
  Same shape as `2_4_b` but with `lr=1e-4`
- `4_5_b`
  `lr=1e-4`, `gpu_count=2`, `batch_size=1`, `gradient_accumulation_steps=16`, `vllm_gpu_memory_utilization=0.35`
- `5_6_b`
  `lr=1e-5`, `gpu_count=2`, `batch_size=1`, `gradient_accumulation_steps=16`, `vllm_gpu_memory_utilization=0.35`
- `6_9_b`
  `lr=1e-5`, `gpu_count=4`, `batch_size=1`, `gradient_accumulation_steps=16`, `vllm_gpu_memory_utilization=0.35`, `rollouts_per_stage=1024`
- `9_12_b`
  `lr=6e-6`, `gpu_count=4`, `batch_size=16`, `vllm_gpu_memory_utilization=0.6`
- `12_15_b`
  `lr=5e-6`, `gpu_count=4`, `batch_size=2`, `vllm_gpu_memory_utilization=0.8`
- `15_20_b`
  `lr=5e-6`, `gpu_count=4`, `batch_size=16`, `use_vllm=False`
- `20_40_b`
  `lr=4e-6`, `gpu_count=8`, `batch_size=16`, `use_vllm=False`, `use_4bit=True`
- `40_80_b`
  `lr=3e-6`, `gpu_count=8`, `batch_size=2`, `use_vllm=False`, `use_4bit=True`

Every bucket is also labeled with its own key through `config["label"] = key`.

### Current public functions

- `get_grpo_config(param_nums)`
  Maps parameter count into a config bucket.
- `get_run_cmd(config, gpu_nums)`
  Builds the actual `torchrun` or `deepspeed` command line for `train_grpo_env.py`.
- `get_training_json(train_info)`
  Produces:
  - `train_request`
  - `run_cmd`

### Important current behavior in `get_training_json(train_info)`

- Reads `model_name`, `model_path`, and `dataset_type.environment_name`.
- Calls `get_model_architecture()` and `get_model_num_params()`.
- Applies special overrides:
  - `Qwen/Qwen2.5-3B-Instruct` -> `2_4_b_qwen`
  - `mistralai/Mistral-7B-Instruct-v0.2` and `v0.3` -> `6_9_b`
- Derives runtime toggles from `scripts/model_utility.py`:
  - `disable_flash_attention(...)`
  - `disable_action_mask(...)`
  - `get_gradient_checkpointing(...)`
  - `get_use_vllm(...)`
- Multiplies the chosen learning rate by `train_info["reg_ratio"]`.
- Sets operational fields like:
  - `save_before_remaining_time = 10`
  - `min_steps = 100`
  - `adjust_batch_size = False`
  - `periodic_save_steps = 75`

## Launch Command Shape

`get_run_cmd(...)` currently builds a command around:

- `torchrun --nproc_per_node=<gpu_count>` by default
- `deepspeed` when `distributed == "ds"`

Important CLI flags included in the generated command:

- `--request_path`
- `--environment_name`
- `--bf16 True`
- `--output_dir /workspace/data/trained_model`
- `--num_train_epochs`
- `--per_device_train_batch_size`
- `--per_device_eval_batch_size`
- `--gradient_accumulation_steps`
- `--learning_rate`
- `--warmup_steps 35`
- `--lr_scheduler_type cosine_with_min_lr`
- `--lr_scheduler_kwargs`
- `--gradient_checkpointing`
- `--optim`
- `--use_liger`
- `--num_generations`
- `--vllm_mode colocate`
- `--vllm_gpu_memory_utilization`
- `--disable_fa`
- `--disable_action_mask`
- `--beta`
- `--loss_type dr_grpo`
- `--num_iterations 1`
- `--do_eval False`
- `--vllm_max_model_length 16384`

Optional additions:

- `--use_peft --lora_r 32 --lora_alpha 64 --lora_target_modules all-linear`
- `--use_vllm True|False`
- `--deepspeed ds_config/zero3.json`
- `--vllm_tensor_parallel_size`
- `--load_in_4bit True --use_bnb_nested_quant True --bnb_4bit_quant_type nf4`
- `--initial_max_turn`
- `--rollouts_per_stage`
- `--rollout_warmup_rollouts`
- `--mcts_warmup_optimizer_steps`

## `scripts/model_utility.py` Behavior That Affects Training

Important functions and current operational consequences:

- `is_reasoning_tokenizer(...)`
  Currently returns `False`, so the reasoning-tokenizer route is disabled.
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

Liar's Dice specific:

- `LIARS_DICE_RULESET`
- `LIARS_DICE_FINAL_MAX_TURN`
- `LIARS_DICE_INITIAL_HINT_PROB`
- `LIARS_DICE_FINAL_HINT_PROB`
- `EPISODE_TRACE_ENABLED`
- `EPISODE_TRACE_DIR`
- `EPISODE_TRACE_MAX_TEXT_CHARS`
- `EPISODE_TRACE_SAMPLE_RATE`

General training/runtime:

- `WANDB_TOKEN`
- `PYTORCH_CUDA_ALLOC_CONF`
- `HUGGINGFACE_TOKEN`
- `HUGGINGFACE_USERNAME`

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

If observations, legal actions, or task ids look wrong, these backend files are usually the right second stop after the environment function file.
