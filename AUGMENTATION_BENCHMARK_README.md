# Augmentation Strategy Benchmarking Framework — Gaia-2 Branch

## 📌 Overview

This framework enables comprehensive benchmarking of augmentation strategies across 6 tournament environments:
- **goof_spiel**, **gin_rummy**, **gin_rummy_opponent_modeling**
- **liars_dice**, **leduc_poker**, **alfworld**

With 4 augmentation strategies:
- **none** (baseline), **gaussian_noise**, **weight_scaling**, **magnitude_pruning**

**Goal**: Determine which augmentation strategy optimizes:
- Win rate / task success
- Convergence speed (training steps to peak performance)
- Parameter efficiency (performance per parameter count)

---

## 🚀 Getting Started

### Phase 1: Framework Validation ✅ DONE
- [x] `augmentation_strategies.py` — Registry of 4 augmentation strategies
- [x] `benchmark_metrics.py` — Metrics collection framework
- [x] Integration into `train_grpo_env.py` — CLI args + augmentation application
- [x] All framework tests passing

### Phase 2: Pilot Experiment (In Progress)
**Goal**: Validate metrics collection on 1 environment + 2 strategies before scaling to all 6

**Configuration**:
- Environment: `gin_rummy`
- Strategies: `gaussian_noise`, `weight_scaling`
- Training steps: ~5000 (quick validation run)
- Metrics: convergence curves, final win rate, training time

**Run Pilot**:
```bash
cd /Users/admin/Documents/jembut

# Pilot 1: Baseline (no augmentation)
python scripts/train_grpo_env.py \
  --environment gin_rummy \
  --augmentation_strategy none \
  --benchmark_mode \
  --output_dir ./results/pilot/gin_rummy_none/ \
  [... other required args ...]

# Pilot 2: Gaussian Noise
python scripts/train_grpo_env.py \
  --environment gin_rummy \
  --augmentation_strategy gaussian_noise \
  --benchmark_mode \
  --output_dir ./results/pilot/gin_rummy_gaussian_noise/ \
  [... other required args ...]

# Pilot 3: Weight Scaling
python scripts/train_grpo_env.py \
  --environment gin_rummy \
  --augmentation_strategy weight_scaling \
  --benchmark_mode \
  --output_dir ./results/pilot/gin_rummy_weight_scaling/ \
  [... other required args ...]
```

### Phase 3: Full Benchmark (Next)
Once pilot validates metrics collection:
- Run all 4 strategies on all 6 environments
- Collect convergence curves, final metrics, param efficiency
- Generate comparison plots (heatmaps, convergence curves)
- Identify winning strategy per environment

---

## 📊 New Files Created

### Core Framework Files

| File | Purpose | Status |
|------|---------|--------|
| `scripts/augmentation_strategies.py` | Strategy registry (4 strategies, factory) | ✅ Created |
| `scripts/benchmark_metrics.py` | Metrics collection + aggregation | ✅ Created |
| `benchmark_config.yaml` | Experiment matrix specification | ✅ Created |

### Integration & Tests

| File | Purpose | Status |
|------|---------|--------|
| `scripts/train_grpo_env.py` (modified) | Added CLI args + augmentation application | ✅ Integrated |
| `test_framework_structure.py` | Validates imports + structure (no torch required) | ✅ Passing |
| `test_augmentation.py` | Full test with model loading (requires torch) | 📝 Ready |

### Documentation

| File | Purpose |
|------|---------|
| `AUGMENTATION_BENCHMARK_README.md` | This file — quickstart guide |
| `benchmark_config.yaml` | Declarative experiment specification |

---

## 🧪 Testing the Framework

### Test 1: Structure Validation (No Torch Required)
```bash
cd /Users/admin/Documents/jembut
python test_framework_structure.py
```

**Output**:
```
✅ Augmentation Framework:   PASS
✅ Metrics Framework:        PASS
✅ Train Script Integration: PASS
✨ All tests passed!
```

### Test 2: Full Framework Test (Requires Torch + Transformers)
```bash
python test_augmentation.py
```

This loads a real model and applies each augmentation strategy.

---

## 📝 Usage: CLI Arguments

The augmentation & benchmarking flags are now part of `TrainingArguments`:

```bash
python scripts/train_grpo_env.py \
  --environment gin_rummy \                    # Environment to train on
  --augmentation_strategy gaussian_noise \     # Strategy: none | gaussian_noise | weight_scaling | magnitude_pruning
  --benchmark_mode \                           # Enable metrics collection
  --output_dir ./results/gin_rummy_gs/ \
  [other standard GRPO args...]
```

### New Arguments

- `--augmentation_strategy` (str, default: "none")
  - Which augmentation to apply before GRPO training
  - Options: none, gaussian_noise, weight_scaling, magnitude_pruning

- `--benchmark_mode` (bool, default: False)
  - Enable metrics collection during training
  - Metrics saved to output_dir/metrics.json

---

## 🔍 How It Works

### 1. Augmentation Application

After model loading but before GRPO training:
```python
if training_args.augmentation_strategy != "none":
    strategy = get_strategy(training_args.augmentation_strategy)
    model = strategy.apply(model)
```

**Each strategy**:
- Gaussian Noise: Add random noise (std=0.005) to all weights
- Weight Scaling: Multiply linear layer weights by factor (1.001)
- Magnitude Pruning: Zero out smallest 0.1% of weights per layer

### 2. Metrics Collection

If `--benchmark_mode` is enabled:
```python
metrics_collector = EnvironmentMetricsCollector(
    env_name="gin_rummy",
    strategy_name="gaussian_noise",
    model_name="Hermes-3-Llama-3.2-3B"
)
```

**Tracks**:
- Episode rewards (running mean/std)
- Task success rate (win rate)
- Convergence windows (every 100 episodes)
- Training time

### 3. Metrics Output

Post-training, metrics are saved as JSON:
```json
{
  "gin_rummy_gaussian_noise": {
    "env_name": "gin_rummy",
    "strategy_name": "gaussian_noise",
    "episode_count": 5000,
    "final_mean_reward": 0.65,
    "final_success_rate": 0.72,
    "window_stats": [...]
  }
}
```

---

## 📈 Benchmarking Workflow

### Step 1: Run Pilot (Validation)
```bash
# Quick validation with 1 env × 3 strategies
for STRATEGY in none gaussian_noise weight_scaling; do
  python scripts/train_grpo_env.py \
    --environment gin_rummy \
    --augmentation_strategy $STRATEGY \
    --benchmark_mode \
    --output_dir ./results/pilot/gin_rummy_$STRATEGY/
done
```

### Step 2: Analyze Pilot Results
```bash
# Metrics will be in:
./results/pilot/gin_rummy_none/metrics.json
./results/pilot/gin_rummy_gaussian_noise/metrics.json
./results/pilot/gin_rummy_weight_scaling/metrics.json

# Compare convergence curves, final metrics
```

### Step 3: Scale to All Environments (If Pilot Successful)
```bash
# 6 environments × 4 strategies = 24 training runs
# Can run in parallel on multiple GPUs

for ENV in goof_spiel gin_rummy gin_rummy_opponent_modeling liars_dice leduc_poker alfworld; do
  for STRATEGY in none gaussian_noise weight_scaling magnitude_pruning; do
    python scripts/train_grpo_env.py \
      --environment $ENV \
      --augmentation_strategy $STRATEGY \
      --benchmark_mode \
      --output_dir ./results/full/$ENV/augment_$STRATEGY/ &
  done
done
wait
```

### Step 4: Generate Report
```python
from scripts.benchmark_metrics import BenchmarkAggregator

# Load all results
agg = BenchmarkAggregator()
for env in [...]:
  for strategy in [...]:
    # Load metrics from JSON
    agg.add_collector(collector)

# Generate CSV + plots
agg.generate_csv_report("results/comparison.csv")
# Add plot generation script here
```

---

## 🎯 Expected Outputs

### Per-Run Outputs
```
./results/pilot/gin_rummy_gaussian_noise/
├── metrics.json              # Metrics collector state
├── checkpoint_1000/          # Model checkpoint
├── checkpoint_5000/          
└── logs/                      # WandB logs
```

### Final Comparison
```
./results/
├── pilot/
│   ├── gin_rummy_none/
│   ├── gin_rummy_gaussian_noise/
│   └── gin_rummy_weight_scaling/
│
├── full/                      # (After scaling)
│   ├── goof_spiel/
│   ├── gin_rummy/
│   ├── leduc_poker/
│   └── [...]
│
├── comparison.csv             # Summary table
└── plots/
    ├── convergence_curves.png      # 6×4 grid
    ├── win_rate_heatmap.png        # Strategy performance
    └── efficiency_scatter.png      # Parameter efficiency
```

---

## ⚙️ Configuration Reference

### `benchmark_config.yaml` Sections

**`strategies`**: Define augmentation strategy hyperparameters
```yaml
strategies:
  - name: gaussian_noise
    hyperparams:
      std: 0.005          # Noise standard deviation
```

**`experiment`**: Global training parameters
```yaml
experiment:
  model_name: "NousResearch/Hermes-3-Llama-3.2-3B"
  training_steps: 10000
  batch_size: 4
  eval_checkpoints: [1000, 5000, 10000]
```

**`pilot`**: Quick validation settings
```yaml
pilot:
  environments: [gin_rummy]
  strategies: [gaussian_noise, weight_scaling]
  training_steps: 5000
```

---

## 🐛 Troubleshooting

### Issue: Model fails to load
**Solution**: Ensure HuggingFace token is set
```bash
export HF_TOKEN=hf_xxx...
```

### Issue: Augmentation crashes at apply()
**Possible cause**: torch not available
**Solution**: Ensure conda env is activated with torch installed

### Issue: Metrics not collected
**Check**: Did you pass `--benchmark_mode` flag?
**Location**: Metrics saved to `{output_dir}/metrics.json`

---

## 📚 Architecture Diagram

```
train_grpo_env.py
├── Load model
├── Apply augmentation strategy    ← NEW
│   ├── gaussian_noise
│   ├── weight_scaling
│   └── magnitude_pruning
├── Initialize metrics collector   ← NEW
├── Run GRPO training
│   └── [Hook reward collection → metrics_collector.log_episode()]
└── Save metrics.json              ← NEW
```

---

## 🔮 Next Steps

### Immediate (This Sprint)
1. ✅ Framework validation (Phase 1 tests passing)
2. 🔄 Run pilot experiment (gin_rummy × 2 strategies)
3. 📊 Validate metrics collection works
4. 🎯 Identify any bugs before full scale-out

### Following Sprint
5. 📈 Scale to all 6 environments
6. 📉 Generate convergence comparison plots
7. 📋 Produce final benchmark report
8. 🎁 Recommend best augmentation per environment

---

## 📞 Support

For questions on:
- **Framework structure**: See `scripts/augmentation_strategies.py`, `scripts/benchmark_metrics.py`
- **Integration**: See modifications to `scripts/train_grpo_env.py` (search `[Augment]`, `[Benchmark]`)
- **Testing**: Run `python test_framework_structure.py`
- **Configuration**: Edit `benchmark_config.yaml`

