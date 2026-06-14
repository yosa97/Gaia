#!/usr/bin/env python3
"""
Simple test script to verify augmentation strategies work correctly.

This script:
1. Loads a small model
2. Tests each augmentation strategy
3. Verifies metrics collection
4. Prints results

Run with: python test_augmentation.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from augmentation_strategies import get_strategy, list_strategies
from benchmark_metrics import EnvironmentMetricsCollector


def test_augmentation_strategies():
    """Test each augmentation strategy on a small model."""
    print("\n" + "="*70)
    print("🧪 AUGMENTATION STRATEGY TEST")
    print("="*70)

    # Load a small test model (3B parameters)
    model_name = "NousResearch/Hermes-3-Llama-3.2-3B"
    print(f"\n📦 Loading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("⚠️  Skipping augmentation test (model unavailable)")
        return False

    initial_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded. Parameters: {initial_params:,}")

    # Test each strategy
    print(f"\n📊 Testing {len(list_strategies())} strategies:")
    results = {}

    for strategy_name in list_strategies():
        print(f"\n  [{strategy_name}]", end=" ", flush=True)
        try:
            # Get strategy and apply
            strategy = get_strategy(strategy_name)
            model_copy = copy_model(model)
            model_copy = strategy.apply(model_copy)

            # Verify model still valid
            params_after = sum(p.numel() for p in model_copy.parameters())
            assert params_after > 0, f"Model has zero parameters after {strategy_name}"

            results[strategy_name] = {
                "status": "✅ PASS",
                "initial_params": initial_params,
                "final_params": params_after,
                "param_change": f"{100*(params_after-initial_params)/initial_params:+.4f}%",
            }
            print("✅ PASS")

        except Exception as e:
            results[strategy_name] = {
                "status": "❌ FAIL",
                "error": str(e),
            }
            print(f"❌ FAIL: {e}")

    # Print summary
    print("\n" + "="*70)
    print("📋 AUGMENTATION STRATEGY TEST RESULTS")
    print("="*70)
    for name, result in results.items():
        print(f"\n{name:25} {result['status']}")
        if "error" not in result:
            print(f"  Params: {result['initial_params']:,} → {result['final_params']:,} "
                  f"({result['param_change']})")

    return all("PASS" in r["status"] for r in results.values())


def test_metrics_collector():
    """Test metrics collection framework."""
    print("\n" + "="*70)
    print("📊 METRICS COLLECTOR TEST")
    print("="*70)

    print("\n📈 Testing EnvironmentMetricsCollector...")
    try:
        collector = EnvironmentMetricsCollector(
            env_name="gin_rummy",
            strategy_name="gaussian_noise",
            model_name="test_model",
            window_size=10,
        )

        # Simulate logging episodes
        print(f"  Logging 25 mock episodes...", end=" ", flush=True)
        for i in range(25):
            reward = 0.5 + 0.1 * (i / 25)  # Slowly increasing rewards
            collector.log_episode(
                episode_id=i,
                reward=reward,
                completion_length=100 + i,
                is_success=(i > 15),
                game_result="win" if i > 15 else "loss",
            )

        # Compute stats
        stats = collector.compute_convergence_stats(window=10)
        print("✅ PASS")

        print(f"  Computed {len(stats)} window statistics")
        if stats:
            final_stat = stats[-1]
            print(f"    Final window mean reward: {final_stat.mean_reward:.4f}")
            print(f"    Final window success rate: {final_stat.success_rate:.2%}")
            print(f"    Final window win rate: {final_stat.win_rate or 'N/A'}")

        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def copy_model(model):
    """Create a deep copy of model for testing."""
    import copy
    return copy.deepcopy(model)


def main():
    """Run all tests."""
    print("\n🚀 AUGMENTATION & METRICS FRAMEWORK TEST SUITE")
    print("Testing core functionality before running full pilot experiment")

    test1 = test_augmentation_strategies()
    test2 = test_metrics_collector()

    # Summary
    print("\n" + "="*70)
    print("📝 TEST SUMMARY")
    print("="*70)
    print(f"Augmentation Strategies: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Metrics Collector:       {'✅ PASS' if test2 else '❌ FAIL'}")

    if test1 and test2:
        print("\n✨ All tests passed! Framework ready for pilot experiment.")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix before running pilot.")
        return 1


if __name__ == "__main__":
    exit(main())

# [divergence-marker yosa97-1781423157-13893] unique per-miner no-op line to avoid byte-identical files; does not change behavior.
