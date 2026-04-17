"""Inspect a WandB run for the reward-hacking probe metrics.

Reads WANDB_API_KEY from the environment. Does not print the key.
"""

import os
import sys

import wandb


RUN_PATH = "bima-bagaskhoro-basgas/tournament-environments/1_environment_test_gin_rummy_qwen_3B_Instruct"

api = wandb.Api(api_key="")
try:
    run = api.run(RUN_PATH)
except Exception as e:
    print(f"ERR fetching run: {e}")
    sys.exit(1)

print("=== RUN SUMMARY ===")
print(f"Name:    {run.name}")
print(f"State:   {run.state}")
print(f"Created: {run.created_at}")
print(f"URL:     {run.url}")
print()

print("=== CONFIG (env name, model) ===")
cfg = dict(run.config)
for k in sorted(cfg):
    lk = k.lower()
    if any(s in lk for s in ("env", "model", "reward_weight", "reward_func")):
        print(f"  {k}: {cfg[k]}")
print()

print("=== SUMMARY KEYS (final metric values) ===")
summary = dict(run.summary)
# TRL logs per-reward-function metrics under train/rewards/<func_name>/{mean,std}.
reward_keys = sorted(k for k in summary if "rewards/" in k)
other_reward_keys = sorted(
    k for k in summary
    if "reward" in k.lower() and "rewards/" not in k
)

print("rewards/* keys:")
for k in reward_keys:
    v = summary[k]
    print(f"  {k} = {v}")
if not reward_keys:
    print("  (none)")
print()

print("other reward-related keys:")
for k in other_reward_keys:
    v = summary[k]
    print(f"  {k} = {v}")
if not other_reward_keys:
    print("  (none)")
print()

print("=== PROBE PRESENCE CHECK ===")
# TRL prefixes all reward metrics with the mode name (train/ or eval/).
# Check both so this works during and after training.
_PROBES = [
    "rewards/probe_shaping_dominance/mean",
    "rewards/probe_shaping_dominance/std",
    "rewards/probe_completion_length/mean",
    "rewards/probe_completion_length/std",
    "rewards/probe_invalid_count/mean",
    "rewards/probe_terminal_raw/mean",
    "rewards/format_guard/mean",
]
for key in _PROBES:
    train_key = f"train/{key}"
    eval_key  = f"eval/{key}"
    if train_key in summary:
        print(f"  [YES] {train_key} = {summary[train_key]}")
    elif eval_key in summary:
        print(f"  [YES] {eval_key} = {summary[eval_key]}")
    else:
        print(f"  [no ] {key}")

print()
print("=== LAST HISTORY ROW ===")
try:
    hist = run.history(samples=5, pandas=False)
    if hist:
        last = hist[-1]
        reward_row = {k: v for k, v in last.items() if "reward" in k.lower() or "probe" in k.lower() or "format_guard" in k.lower()}
        for k in sorted(reward_row):
            print(f"  {k} = {reward_row[k]}")
    else:
        print("  (no history rows)")
except Exception as e:
    print(f"  ERR fetching history: {e}")
