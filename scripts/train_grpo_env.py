import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import transformers
from customized_trainer import (
    CustomEvalSaveCallback,
    GRPOCustomEvalSaveCallback,
    WhenToEvalHandler,
    resize_if_needed,
    set_generation_config,
)
from datasets import Dataset
from peft import PeftModelForCausalLM
from transformers import AutoTokenizer, TrainerCallback
from transformers.modeling_utils import is_deepspeed_zero3_enabled
from transformers.trainer_utils import is_main_process
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.grpo_trainer import (
    apply_chat_template,
    gather_object,
    nanmax,
    nanmin,
    nanstd,
    pad,
    prepare_multimodal_messages,
)
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import use_adapter

from utility import log_info
from model_utility import is_reasoning_tokenizer
from envs import GAMES_TO_TASK_ID_RANGE
from envs.env_configs import EnvTrainingConfig, get_env_config

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
STANDARD_GRPO_EXTRA_COLUMN = "extra_data"
STANDARD_GRPO_PROMPT_COLUMN = "prompt"


# ============================================================================
# SFT Cold-Start Stage
# ============================================================================

def _load_sft_datasets(datasets_dir: str, dataset_list: list[str], env_name: str | None):
    """Load whitelisted SFT datasets from the pre-downloaded cache.

    Returns a HuggingFace Dataset ready for SFT, or None if no data found.
    The datasets are expected at ``{datasets_dir}/{org--name}/``.

    **Game-aware filtering**: If ``env_name`` is provided, only examples
    containing the game name are kept. This gives a genuine edge over miners
    who blindly feed all data — our SFT is pure signal, not noise from
    unrelated games.
    """
    try:
        from datasets import load_from_disk, concatenate_datasets
    except ImportError:
        log_info("[SFT] 'datasets' package not available, skipping SFT stage.")
        return None

    # Game name variants for robust filtering
    _GAME_KEYWORDS: dict[str, list[str]] = {
        "gin_rummy":   ["gin_rummy", "gin rummy", "ginrummy"],
        "liars_dice":  ["liars_dice", "liar's dice", "liars dice", "liarsdice"],
        "leduc_poker": ["leduc_poker", "leduc poker", "leducpoker"],
        "goof_spiel":  ["goof_spiel", "goofspiel", "goof spiel"],
    }

    loaded = []
    for name in dataset_list:
        ds_path = os.path.join(datasets_dir, name)
        if not os.path.isdir(ds_path):
            # Auto-download from HuggingFace if not cached locally
            log_info(f"[SFT] Dataset not found locally: {ds_path}, attempting HuggingFace download...")
            try:
                # huggingface_hub reads HF_HUB_ENABLE_HF_TRANSFER at import time and caches it.
                # We must reset both the env var AND the module-level constant.
                _prev_hf_transfer = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                try:
                    from huggingface_hub import constants as _hf_const
                    _hf_const.HF_HUB_ENABLE_HF_TRANSFER = False
                except Exception:
                    pass
                from datasets import load_dataset
                ds = load_dataset(name, split="train")
                os.makedirs(ds_path, exist_ok=True)
                ds.save_to_disk(ds_path)
                log_info(f"[SFT] ✅ Downloaded & cached {len(ds)} examples from HF: {name}")
                if _prev_hf_transfer is not None:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = _prev_hf_transfer
            except Exception as dl_err:
                log_info(f"[SFT] ⚠️ Auto-download failed for {name}: {dl_err}, skipping.")
                continue
        else:
            try:
                ds = load_from_disk(ds_path)
                log_info(f"[SFT] Loaded {len(ds)} examples from {ds_path}")
            except Exception as e:
                log_info(f"[SFT] Failed to load {ds_path}: {e}")
                continue

        # Game-aware filtering — only keep examples relevant to current game
        if env_name and env_name in _GAME_KEYWORDS:
            pre_filter_len = len(ds)

            has_game_col = "game" in ds.column_names

            if has_game_col:
                keywords = _GAME_KEYWORDS[env_name]

                def _matches_game(example):
                    return example.get("game", "").lower() in keywords

            else:
                keywords = _GAME_KEYWORDS[env_name]

                def _matches_game(example):
                    text = str(example).lower()
                    return any(kw in text for kw in keywords)

            try:
                filtered = ds.filter(_matches_game, desc=f"Filtering for {env_name}")
                if len(filtered) > 0:
                    log_info(
                        f"[SFT] 🎯 Game-filtered {env_name} "
                        f"(via {'column' if has_game_col else 'text'}): "
                        f"{pre_filter_len} → {len(filtered)} examples "
                        f"({len(filtered)/max(pre_filter_len,1)*100:.0f}% relevant)"
                    )
                    ds = filtered
                else:
                    log_info(
                        f"[SFT] ⚠️ Game filter for {env_name} returned 0 results, "
                        f"using all {pre_filter_len} examples as fallback"
                    )
            except Exception as fe:
                log_info(f"[SFT] Filter failed ({fe}), using unfiltered dataset")

        loaded.append(ds)

    if not loaded:
        return None

    combined = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]
    log_info(f"[SFT] Total SFT examples (post-filter): {len(combined)}")

    # ── Format conversion: ShareGPT → OpenAI chat format ──────────────
    # Dataset uses: {"from": "user/assistant", "value": "..."}  (ShareGPT)
    # SFTTrainer expects: {"role": "user/assistant", "content": "..."}  (OpenAI)
    if "conversations" in combined.column_names:
        _ROLE_MAP = {"user": "user", "assistant": "assistant", "system": "system",
                     "human": "user", "gpt": "assistant"}

        def _convert_sharegpt(example):
            messages = []
            for msg in example.get("conversations", []):
                role = _ROLE_MAP.get(msg.get("from", ""), msg.get("from", "user"))
                content = msg.get("value", "")
                messages.append({"role": role, "content": content})
            return {"messages": messages}

        combined = combined.map(_convert_sharegpt, desc="Converting ShareGPT → OpenAI format")
        # Drop columns SFTTrainer doesn't need
        drop_cols = [c for c in combined.column_names if c not in ("messages",)]
        if drop_cols:
            combined = combined.remove_columns(drop_cols)
        log_info(f"[SFT] ✅ Converted {len(combined)} examples to OpenAI chat format")

    return combined


def run_sft_cold_start(model, tokenizer, training_args, peft_config):
    """Run a short SFT stage using whitelisted datasets before GRPO.

    This stabilises model output format (numeric action IDs) and reduces
    variance from cold-start randomness.  Only runs if the validator has
    pre-downloaded datasets (env vars ``MINER_DATASETS_DIR`` / ``MINER_DATASETS``).
    """
    datasets_dir = os.environ.get("MINER_DATASETS_DIR")
    datasets_csv = os.environ.get("MINER_DATASETS", "")
    dataset_list = [d.strip() for d in datasets_csv.split(",") if d.strip()]
    env_name = training_args.environment_name

    if not datasets_dir or not dataset_list:
        log_info("[SFT] No miner datasets available (MINER_DATASETS_DIR / MINER_DATASETS not set). Skipping SFT.")
        return model

    log_info(f"[SFT] Datasets dir: {datasets_dir}")
    log_info(f"[SFT] Requested datasets: {dataset_list}")

    sft_dataset = _load_sft_datasets(datasets_dir, dataset_list, env_name)
    if sft_dataset is None or len(sft_dataset) == 0:
        log_info("[SFT] No SFT data loaded, skipping SFT stage.")
        return model

    try:
        from trl import SFTTrainer, SFTConfig

        # Scale SFT LR proportionally to GRPO LR.
        # Hardcoded 2e-5 is fine for small models (GRPO LR=3e-5),
        # but for 6B+ models (GRPO LR=6-8e-6), 2e-5 is 2.5-3x higher
        # which causes over-fitting → GRPO wastes early steps undoing it.
        grpo_lr = training_args.learning_rate
        sft_lr = min(2e-5, grpo_lr * 2.5) if grpo_lr > 0 else 2e-5

        sft_args = SFTConfig(
            output_dir="/tmp/sft_checkpoint",
            num_train_epochs=1,
            per_device_train_batch_size=min(4, training_args.per_device_train_batch_size),
            learning_rate=sft_lr,
            warmup_steps=10,
            max_grad_norm=1.0,
            save_strategy="no",
            logging_steps=5,
            bf16=True,
            report_to="wandb",
            gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
            if training_args.gradient_checkpointing
            else None,
        )

        sft_trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=sft_dataset,
            processing_class=tokenizer,
            peft_config=peft_config if not isinstance(model, PeftModelForCausalLM) else None,
        )

        log_info("[SFT] Starting SFT cold-start training...")
        sft_trainer.train()
        log_info(f"[SFT] ✅ SFT stage complete. Trained on {len(sft_dataset)} examples.")

        # Signal to curriculum schedulers that SFT was completed successfully.
        # This allows them to skip warmup and start at higher complexity.
        os.environ["SFT_COMPLETED"] = "1"

        # Merge LoRA back into base model so GRPO can attach its own LoRA adapter.
        # GRPOTrainer crashes if it receives a PeftModel + peft_config simultaneously.
        sft_model = sft_trainer.model
        if hasattr(sft_model, "merge_and_unload"):
            log_info("[SFT] Merging SFT LoRA adapter into base model...")
            sft_model = sft_model.merge_and_unload()
            log_info("[SFT] ✅ LoRA merged — base model ready for GRPO.")
        return sft_model

    except ImportError:
        log_info("[SFT] SFTTrainer not available (trl version too old?). Skipping SFT.")
        return model
    except Exception as e:
        log_info(f"[SFT] ⚠️ SFT stage failed: {e}. Continuing with GRPO only.")
        return model


# ============================================================================
# Gradient Monitoring Callback
# ============================================================================

class GradientMonitorCallback(TrainerCallback):
    """Log warnings when gradient norms spike, helping diagnose instability."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "grad_norm" not in logs:
            return
        grad_norm = logs["grad_norm"]
        step = state.global_step
        if grad_norm > 10.0:
            log_info(f"🔴 [GradMonitor] CRITICAL grad_norm={grad_norm:.4f} at step {step} — training may diverge")
        elif grad_norm > 5.0:
            log_info(f"⚠️ [GradMonitor] HIGH grad_norm={grad_norm:.4f} at step {step}")


@dataclass
class TrainingArguments(GRPOConfig):
    request_path: Optional[str] = field(default=None)
    use_liger: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    disable_action_mask: Optional[bool] = field(default=False)
    initial_max_turn: Optional[int] = field(default=2)
    rollouts_per_stage: Optional[int] = field(default=1280)
    environment_name: Optional[str] = field(default=None)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    lora_param_count = 0
    all_param = 0
    embedding_lm_head_param_count = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            log_info(f"trainable: {name}, num_params: {num_params}")
            if "lm_head" in name or "embed_tokens" in name:
                embedding_lm_head_param_count += num_params
            else:
                lora_param_count += num_params
    trainable_params = embedding_lm_head_param_count + lora_param_count
    log_info(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    log_info(
        f"embedding_lm_head_param_count: {embedding_lm_head_param_count} = {embedding_lm_head_param_count * 100 / all_param} %"
    )
    log_info(
        f"loara_param: {lora_param_count} = {lora_param_count * 100 / all_param} %"
    )


class ActionMaskedGRPOTrainer(GRPOTrainer):
    """GRPO trainer that applies an action mask to loss/IS/metrics."""

    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]) -> dict[str, Any]:
        if getattr(self, "rollout_func", None) is None:
            return super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if self.tools:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")  # 0 for tool result tokens, 1 elsewhere

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        action_mask = None
        if extra_fields and "action_mask" in extra_fields:
            action_mask_list = extra_fields["action_mask"]
            # Check if action_mask is a flat list (all elements are integers)
            # This can happen when batch_size=1 and rollout returns [0, 1, 1, 0] instead of [[0, 1, 1, 0]]
            is_flat_list = (
                isinstance(action_mask_list, list)
                and action_mask_list
                and all(isinstance(x, (int, float)) for x in action_mask_list)
            )
            if is_flat_list:
                if len(completion_ids_list) != 1:
                    raise ValueError(
                        f"Flat action_mask received but batch has {len(completion_ids_list)} completions. "
                        f"action_mask must be a list-of-lists when batch_size > 1."
                    )
                action_mask_list = [action_mask_list]
            if not isinstance(action_mask_list, list) or len(action_mask_list) != len(completion_ids_list):
                raise ValueError("action_mask must be a list-of-lists aligned to completions.")

            # Validate per-sample alignment before padding
            for idx, (mask, comp_ids) in enumerate(zip(action_mask_list, completion_ids_list)):
                if len(mask) != len(comp_ids):
                    raise ValueError(
                        f"action_mask[{idx}] length ({len(mask)}) does not match "
                        f"completion_ids[{idx}] length ({len(comp_ids)}). "
                        f"Rollout function returned misaligned data."
                    )

            action_mask = [torch.tensor(mask, device=device) for mask in action_mask_list]
            action_mask = pad(action_mask, padding_value=0, padding_side="right").to(dtype=completion_mask.dtype)
            # Shape check after padding kept for safety
            if action_mask.shape != completion_mask.shape:
                raise ValueError("action_mask shape does not match completion_ids after padding.")

        loss_mask = completion_mask if action_mask is None else completion_mask * action_mask

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = loss_mask if not self.tools else loss_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        "Unknown vLLM importance sampling level: "
                        f"{self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', "
                        "'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            if num_generations > 1:
                std_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
            else:  # this case doesn't occur during training, but could in eval when num_generations_eval=1
                std_rewards = torch.zeros_like(rewards)
        elif self.scale_rewards == "batch":
            # Compute global std
            if rewards.numel() > 1:
                std_rewards = rewards.std().expand_as(rewards)
            else:  # this case doesn't occur during training, but could in eval when num_generations_eval=batch_size=1
                std_rewards = torch.zeros_like(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            mask = loss_mask if not self.tools else loss_mask * tool_mask
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask.bool()]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if self.tools:
            output["tool_mask"] = tool_mask
        if action_mask is not None:
            output["action_mask"] = action_mask
        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        action_mask = inputs.get("action_mask")
        loss_mask = completion_mask if action_mask is None else completion_mask * action_mask
        if self.tools and "tool_mask" in inputs:
            loss_mask = loss_mask * inputs["tool_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        action_mask = inputs.get("action_mask")
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if action_mask is None else completion_mask * action_mask
        if self.tools:
            mask = mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                old_per_token_logps=old_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss


def main():
    """Format of training requests"""
    print("--------------------------------")
    print("TRAINING GRPO ENVIRONMENT")
    print("--------------------------------")
    try:
        argument_parser = transformers.HfArgumentParser((TrainingArguments, ModelConfig))
        training_args, model_args = argument_parser.parse_args_into_dataclasses()

        train_info = json.load(open(training_args.request_path, "r"))
        train_request = train_info["train_request"]
        task_id = train_request["task_id"]
        
        output_dir = training_args.output_dir
        tokenizer = AutoTokenizer.from_pretrained(train_request["model_path"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = get_quantization_config(model_args)
        device_string = "cuda:" + str(LOCAL_RANK)
        device_map = (
            get_kbit_device_map()
            if quantization_config is not None
            else {"": device_string}
        )
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            device_map = None

        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=(
                "flash_attention_2" if not training_args.disable_fa else "eager"
            ),
            torch_dtype=torch.bfloat16,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=device_map,
            quantization_config=quantization_config,
        )

        log_info(f"final training_args: {training_args}")

        if training_args.use_liger:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            model_class = AutoLigerKernelForCausalLM
        else:
            model_class = transformers.AutoModelForCausalLM

        model = model_class.from_pretrained(train_request["model_path"], **model_kwargs)

        # ── SFT Cold-Start Stage ──────────────────────────────────────────
        # Run a short SFT phase using whitelisted datasets before GRPO.
        # This stabilises output format and reduces cold-start variance.
        # Only runs when MINER_DATASETS_DIR / MINER_DATASETS env vars are set
        # (i.e. when the validator has pre-downloaded whitelisted datasets).
        if is_main_process(LOCAL_RANK):
            log_info("[Main] Checking for SFT cold-start datasets...")
        model = run_sft_cold_start(model, tokenizer, training_args, peft_config=get_peft_config(model_args))

        # some model need to set the generation config or encounter the invalid generation config error
        set_generation_config(train_request["model_name"], model)

        peft_config = get_peft_config(model_args)
        if "lora_model" in train_request:
            model = PeftModelForCausalLM.from_pretrained(
                model, train_request["lora_model"], is_trainable=True, **model_kwargs
            )

        if peft_config is None:  # this is full-weight training
            # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
            resize_if_needed(train_request["model_name"], model, len(tokenizer))

        # Check if this is the main process and create the output directory
        if is_main_process(LOCAL_RANK):  # Only create directory on main process
            os.makedirs(training_args.output_dir, exist_ok=True)
            log_info(f"Created output directory: {training_args.output_dir}")
            
        # Limit to at most 200_000 samples to avoid creating too large a dataset
        start_idx, end_idx = GAMES_TO_TASK_ID_RANGE[training_args.environment_name]
        max_samples = 200_000
        total_range = end_idx - start_idx
        if total_range > max_samples:
            # evenly sample max_samples task ids from the range
            selected_indices = sorted(random.sample(range(start_idx, end_idx), max_samples))
        else:
            selected_indices = list(range(start_idx, end_idx))
        train_ds = Dataset.from_list([{"prompt": str(i)} for i in selected_indices])
        dev_ds = train_ds.select(random.sample(range(len(train_ds)), 10))

        log_info(f"world_size: {training_args.world_size}")
        total_steps_per_epoch = (
            len(train_ds)
            * training_args.num_generations
            // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
        )

        log_info(f"total_steps_per_epoch: {total_steps_per_epoch}")

        periodic_save_steps = train_request.get("periodic_save_steps", -1)
        if periodic_save_steps > total_steps_per_epoch:
            periodic_save_steps = -1
            log_info(
                f"The periodic_save_steps ({periodic_save_steps}) is greater than the total_steps_per_epoch ({total_steps_per_epoch}), set periodic_save_steps to -1, do not save the model regularly"
            )
        log_info(f"periodic_save_steps: {periodic_save_steps}")

        training_args.save_only_model = True  # only save the model, not the optimizer
    
        if training_args.gradient_checkpointing:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            
        print("train_ds.column_names: ", train_ds.column_names)

        max_steps = train_request.get("max_steps", -1)
        log_info(f"max_steps: {max_steps}")

        cfg = get_env_config(training_args.environment_name)

        # Callback construction is identical across all three training modes.
        _callback = GRPOCustomEvalSaveCallback(
            WhenToEvalHandler(
                train_request["end_time"],
                train_request["save_before_remaining_time"],
                periodic_save_steps=periodic_save_steps,
                steps_per_epoch=total_steps_per_epoch,
                max_steps=max_steps,
            ),
            train_request["submission_dir"],
            training_args.output_dir,
            train_request["model_name"],
            max_steps,
        )

        # Select training mode: resolve rollout func, trainer class, mode defaults.
        if is_reasoning_tokenizer(tokenizer):
            mode_cfg = cfg.reasoning
            rollout_func = cfg.rollout_last
            default_trainer_class = GRPOTrainer
            default_completion_length = 2048
            training_args.vllm_max_model_length += 2048
            use_eval_dataset = True
        elif training_args.disable_action_mask:
            mode_cfg = cfg.no_mask
            rollout_func = cfg.rollout_last
            default_trainer_class = GRPOTrainer
            default_completion_length = 16
            use_eval_dataset = True
        else:
            mode_cfg = cfg.full_prompt
            rollout_func = cfg.rollout_full
            default_trainer_class = ActionMaskedGRPOTrainer
            default_completion_length = 16
            use_eval_dataset = False

        # Apply per-env/per-mode overrides from ModeConfig.
        if mode_cfg.initial_max_turn is not None:
            training_args.initial_max_turn = mode_cfg.initial_max_turn
        if mode_cfg.rollouts_per_stage is not None:
            training_args.rollouts_per_stage = mode_cfg.rollouts_per_stage
        training_args.max_completion_length = mode_cfg.max_completion_length or default_completion_length
        trainer_class = mode_cfg.trainer_class or default_trainer_class
        print(f"Training with {trainer_class.__name__} (env={training_args.environment_name})")

        common_trainer_kwargs = dict(
            model=model,
            rollout_func=rollout_func,
            reward_funcs=[cfg.reward_func],
            args=training_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[_callback, GradientMonitorCallback()],
        )
        if use_eval_dataset:
            trainer = trainer_class(**common_trainer_kwargs, eval_dataset=dev_ds)
        else:
            trainer = trainer_class(**common_trainer_kwargs)

        trainer.train()
    except Exception as e:
        import traceback
        print(f"Error training: {e}")
        print(traceback.format_exc())
        raise e
    
    if is_main_process(LOCAL_RANK):
        with open(os.path.join(training_args.output_dir, "success.txt"), "w") as f:
            f.write("Success")


if __name__ == "__main__":
    main()
