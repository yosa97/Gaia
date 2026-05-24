# ⚡ Quick Start: How to Test Safely

**Time needed**: 5-30 minutes (depending on your setup)

---

## 🎯 Test Decision Tree

```
Do you want to submit to SN 56 tournament without augmentation?
  ├─ YES → Section A (Use existing workflow)
  └─ NO → Section B (Test augmentation)

Want to benchmark augmentation strategies?
  ├─ YES → Section C (Pilot experiment)
  └─ NO → Section B (Single augmentation test)

Have environment servers running?
  ├─ YES → Run full training
  └─ NO → Run structure tests only
```

---

## 📋 Section A: Production/Tournament Submission (NO CHANGES NEEDED)

**Use your EXISTING workflow exactly as-is:**

```bash
# Option 1: Docker-based (if using tournament setup)
bash examples/run_enviroment.sh

# Option 2: Direct Python (if using local setup)
python scripts/train_grpo_env.py \
  --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
  --environment gin_rummy \
  [... your normal args ...]

# Option 3: Your custom submission script
[your_submission_script]
```

**Result**: 
- ✅ Works exactly as before
- ✅ Augmentation disabled (default)
- ✅ Zero performance overhead
- ✅ Safe for tournament

---

## 🔍 Section B: Test No-Regression First

**Validate framework doesn't break existing code:**

### Step 1: Quick Structure Test (5 minutes, no GPU)
```bash
cd /Users/admin/Documents/jembut
python test_framework_structure.py
```

**Expected Output**:
```
✅ Augmentation Framework:   PASS
✅ Metrics Framework:        PASS
✅ Train Script Integration: PASS
✨ All tests passed! Framework structure is correct.
```

**If this passes** ✅ → Safe to run training  
**If this fails** ❌ → Check error message, likely missing Python package

---

### Step 2: Baseline Training Test (10-30 minutes, GPU needed)

Run training **WITHOUT any augmentation flags** (safest test):

```bash
python scripts/train_grpo_env.py \
  --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
  --environment gin_rummy \
  --augmentation_strategy none \
  --output_dir ./results/test_baseline/ \
  [... other required args ...]
```

**Key points**:
- Explicitly pass `--augmentation_strategy none` (shows you're using new code)
- Training should complete normally
- Metrics NOT collected (benchmark_mode=False by default)

**If successful** ✅ → Framework doesn't break normal training

---

### Step 3: Test with Augmentation (10-30 minutes, GPU needed)

Once baseline passes, test with augmentation enabled:

```bash
python scripts/train_grpo_env.py \
  --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
  --environment gin_rummy \
  --augmentation_strategy gaussian_noise \
  --benchmark_mode \
  --output_dir ./results/test_augment/ \
  [... other required args ...]
```

**Key points**:
- Now using `--augmentation_strategy gaussian_noise`
- `--benchmark_mode` enables metrics collection
- Training should complete normally
- Check output directory for `metrics.json`

**If successful** ✅ → Augmentation works correctly

---

## 📊 Section C: Pilot Experiment (Full Benchmark)

**Test augmentation strategy comparison on gin_rummy:**

### Setup Phase (5 minutes)
```bash
mkdir -p ./results/pilot/{baseline,gaussian,scaling}
```

### Run 3 Strategies (20-90 minutes total, can parallelize)

#### Strategy 1: Baseline (No Augmentation)
```bash
python scripts/train_grpo_env.py \
  --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
  --environment gin_rummy \
  --augmentation_strategy none \
  --benchmark_mode \
  --output_dir ./results/pilot/baseline/ \
  [... other args ...]
```

#### Strategy 2: Gaussian Noise
```bash
python scripts/train_grpo_env.py \
  --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
  --environment gin_rummy \
  --augmentation_strategy gaussian_noise \
  --benchmark_mode \
  --output_dir ./results/pilot/gaussian/ \
  [... other args ...]
```

#### Strategy 3: Weight Scaling
```bash
python scripts/train_grpo_env.py \
  --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
  --environment gin_rummy \
  --augmentation_strategy weight_scaling \
  --benchmark_mode \
  --output_dir ./results/pilot/scaling/ \
  [... other args ...]
```

### Parallel Execution (Faster)
```bash
# Run all 3 simultaneously (requires 3 GPUs or time-sharing)
for STRATEGY in none gaussian_noise weight_scaling; do
  python scripts/train_grpo_env.py \
    --model_name "NousResearch/Hermes-3-Llama-3.2-3B" \
    --environment gin_rummy \
    --augmentation_strategy "$STRATEGY" \
    --benchmark_mode \
    --output_dir "./results/pilot/$STRATEGY/" \
    [... other args ...] &
done
wait
```

### Analyze Results
```bash
# Each run creates metrics.json
ls -la ./results/pilot/*/metrics.json

# Compare convergence curves, win rates, etc.
# (Full analysis script coming in Phase 2)
```

---

## 🚀 Recommended Test Flow

### For Tournament Submission:
```
✅ Option A: Don't change anything
   Run: bash examples/run_enviroment.sh
   (augmentation disabled by default)

OR

✅ Option B: Test baseline first, then submit
   Run: python test_framework_structure.py  [5 min]
   Run: training with --augmentation_strategy none  [30 min]
   Verify: training completes successfully
   Submit: existing workflow (augmentation disabled)
```

### For Augmentation Research:
```
✅ Step 1: Structure test
   python test_framework_structure.py  [5 min]

✅ Step 2: Baseline training
   python train_grpo_env.py --augmentation_strategy none  [30 min]

✅ Step 3: Single augmentation test
   python train_grpo_env.py --augmentation_strategy gaussian_noise  [30 min]

✅ Step 4: Full pilot (3 strategies)
   for STRATEGY in none gaussian_noise weight_scaling
   (3 parallel runs × 30 min = 30-90 min total)

✅ Step 5: Analyze metrics
   Compare convergence_curves, win_rates
   Document findings
```

---

## 🔧 Environment Setup (if needed)

### For Structure Tests (No GPU):
```bash
# Python 3.9+ required
python --version  # Check Python version
python test_framework_structure.py  # Should work
```

### For Full Training (GPU Required):
```bash
# Conda environment example
conda activate your_env
# Verify these are available:
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "from trl import GRPOTrainer; print('✓ TRL installed')"

# Set environment variables (if needed)
export ENVIRONMENT_SERVER_URLS="http://localhost:8000"
export HF_TOKEN="your_token_here"
```

---

## 📞 Troubleshooting

### ❌ Test fails: `ModuleNotFoundError: No module named 'augmentation_strategies'`
**Solution**: Ensure files exist
```bash
ls scripts/augmentation_strategies.py
ls scripts/benchmark_metrics.py
```

### ❌ Test fails: `AttributeError: 'TrainingArguments' has no field 'augmentation_strategy'`
**Solution**: train_grpo_env.py not modified correctly
```bash
# Check file was updated
grep "augmentation_strategy" scripts/train_grpo_env.py
```

### ❌ Training fails: `RuntimeError: CUDA out of memory`
**Solution**: Augmentation adds minimal memory; likely unrelated
- Try with smaller batch size
- Try `--augmentation_strategy none` to verify augmentation isn't cause

### ❌ Training slow: `Convergence slower than expected`
**Solution**: Metrics collection only active with `--benchmark_mode`
- Default behavior: no metrics collection, no slowdown
- With flag: < 1% slowdown expected (Phase 2)

---

## ✅ Success Criteria

### Structure Test Success:
```
✅ All 3 test suites PASS
✅ No import errors
✅ No missing files
```

### Baseline Training Success:
```
✅ Training completes without crash
✅ Model converges (loss decreases)
✅ No GPU memory errors
```

### Augmentation Test Success:
```
✅ Training with --augmentation_strategy gaussian_noise completes
✅ Model converges (should be similar to baseline)
✅ metrics.json created in output directory
✅ Metrics show convergence curves
```

### Pilot Experiment Success:
```
✅ All 3 strategies (none, gaussian, scaling) complete
✅ Each creates metrics.json
✅ Can compare win_rate trends across strategies
✅ Identify which strategy performs best
```

---

## 📋 Checklist Before Running

- [ ] Python version 3.9+ installed
- [ ] GPU available (if doing training)
- [ ] torch/transformers installed
- [ ] HF_TOKEN set (if downloading models)
- [ ] ENVIRONMENT_SERVER_URLS configured (if needed)
- [ ] Output directory writable
- [ ] 30+ GB disk space available (for models + results)

---

## Next: After Successful Tests

Once you've completed testing:

1. ✅ **No Augmentation Case** → Use for tournament submission
   ```bash
   bash examples/run_enviroment.sh
   ```

2. ✅ **Augmentation Case** → Continue to Phase 2
   - Extend metrics collection to all 6 environments
   - Generate convergence comparison plots
   - Document best strategy per environment

3. ✅ **Production Case** → Same as before
   - Framework disabled by default
   - Zero performance impact
   - Safe for all scenarios

---

**Ready? Start with:** `python test_framework_structure.py`

Good luck! 🚀
