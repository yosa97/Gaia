"""Unit tests for scripts/envs/reward_monitoring.py.

Run from the repo root:
    python scripts/tests/test_reward_monitoring.py
or with pytest:
    pytest scripts/tests/test_reward_monitoring.py -v
"""

import importlib.util
import math
import os
import sys

# Load scripts/envs/reward_monitoring.py directly, bypassing the envs package
# __init__ (which imports alf_world_env.py -> trl and isn't needed here).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MONITOR_PATH = os.path.abspath(os.path.join(_HERE, "..", "envs", "reward_monitoring.py"))
_spec = importlib.util.spec_from_file_location("reward_monitoring", _MONITOR_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

shaping_dominance_probe     = _mod.shaping_dominance_probe
length_probe                = _mod.length_probe
invalid_count_probe         = _mod.invalid_count_probe
terminal_raw_probe          = _mod.terminal_raw_probe
format_guard_reasoning      = _mod.format_guard_reasoning
_THINK_PENALTY              = _mod._THINK_PENALTY
_LEN_SOFT_CAP_CHARS         = _mod._LEN_SOFT_CAP_CHARS
_LEN_PENALTY_PER_OVER_CHAR  = _mod._LEN_PENALTY_PER_OVER_CHAR


# ---------------------------------------------------------------------------
# Name stability (used as WandB label at train_grpo_env.py:375)
# ---------------------------------------------------------------------------

def test_probe_names_stable():
    assert shaping_dominance_probe.__name__ == "probe_shaping_dominance"
    assert length_probe.__name__            == "probe_completion_length"
    assert invalid_count_probe.__name__     == "probe_invalid_count"
    assert terminal_raw_probe.__name__      == "probe_terminal_raw"
    assert format_guard_reasoning.__name__  == "format_guard"


# ---------------------------------------------------------------------------
# shaping_dominance_probe
# ---------------------------------------------------------------------------

def test_shaping_dominance_nan_when_missing_keys():
    out = shaping_dominance_probe(["a", "b", "c"])
    assert len(out) == 3
    assert all(math.isnan(x) for x in out)


def test_shaping_dominance_extreme_shaping():
    out = shaping_dominance_probe(["a"], terminal_raw=[1.0], shaping_sum=[50.0])
    assert out == [50.0 / 51.0]


def test_shaping_dominance_extreme_terminal():
    out = shaping_dominance_probe(["a"], terminal_raw=[50.0], shaping_sum=[0.0])
    assert out == [0.0]


def test_shaping_dominance_both_zero():
    out = shaping_dominance_probe(["a"], terminal_raw=[0.0], shaping_sum=[0.0])
    assert out == [0.0]


def test_shaping_dominance_negative_values_use_abs():
    out = shaping_dominance_probe(["a"], terminal_raw=[-50.0], shaping_sum=[-10.0])
    assert out == [10.0 / 60.0]


def test_shaping_dominance_per_sample_nan_when_one_missing():
    out = shaping_dominance_probe(
        ["a", "b"],
        terminal_raw=[1.0, float("nan")],
        shaping_sum=[5.0, 2.0],
    )
    assert out[0] == 5.0 / 6.0
    assert math.isnan(out[1])


# ---------------------------------------------------------------------------
# length_probe
# ---------------------------------------------------------------------------

def test_length_probe_counts_chars():
    assert length_probe(["", "abc", "hello"]) == [0.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# invalid_count_probe
# ---------------------------------------------------------------------------

def test_invalid_count_probe_passthrough():
    assert invalid_count_probe(["a", "b"], invalid_count=[0, 3]) == [0.0, 3.0]


def test_invalid_count_probe_nan_when_missing():
    out = invalid_count_probe(["a"])
    assert math.isnan(out[0])


# ---------------------------------------------------------------------------
# terminal_raw_probe
# ---------------------------------------------------------------------------

def test_terminal_raw_passthrough():
    assert terminal_raw_probe(["a", "b"], terminal_raw=[1.0, -50.0]) == [1.0, -50.0]


def test_terminal_raw_nan_when_missing():
    out = terminal_raw_probe(["a"])
    assert math.isnan(out[0])


# ---------------------------------------------------------------------------
# format_guard_reasoning
# ---------------------------------------------------------------------------

def test_format_guard_happy_path():
    out = format_guard_reasoning(["<think>reasoning here</think>answer"])
    assert out == [0.0]


def test_format_guard_missing_think():
    out = format_guard_reasoning(["just an answer with no think"])
    assert out == [-_THINK_PENALTY]


def test_format_guard_empty_think():
    out = format_guard_reasoning(["<think></think>answer"])
    assert out == [-_THINK_PENALTY]


def test_format_guard_whitespace_only_think():
    out = format_guard_reasoning(["<think>   \n\t </think>answer"])
    assert out == [-_THINK_PENALTY]


def test_format_guard_duplicate_think():
    out = format_guard_reasoning(["<think>one</think><think>two</think>answer"])
    assert out == [-_THINK_PENALTY]


def test_format_guard_length_overage():
    long_completion = "<think>t</think>" + ("x" * (_LEN_SOFT_CAP_CHARS + 100))
    out = format_guard_reasoning([long_completion])
    overage = len(long_completion) - _LEN_SOFT_CAP_CHARS
    expected = -_LEN_PENALTY_PER_OVER_CHAR * overage
    assert abs(out[0] - expected) < 1e-9


def test_format_guard_both_violations_stack():
    long_completion = "no think tag here " + ("x" * _LEN_SOFT_CAP_CHARS)
    out = format_guard_reasoning([long_completion])
    overage = len(long_completion) - _LEN_SOFT_CAP_CHARS
    expected = -_THINK_PENALTY - _LEN_PENALTY_PER_OVER_CHAR * overage
    assert abs(out[0] - expected) < 1e-9


def test_format_guard_never_positive():
    cases = [
        "",
        "<think>ok</think>",
        "<think></think>",
        "no tags",
        "<think>" + "y" * 10_000 + "</think>",
    ]
    out = format_guard_reasoning(cases)
    assert all(v <= 0.0 for v in out)


# ---------------------------------------------------------------------------
# Magnitude sanity: format guard must be small vs env rewards
# ---------------------------------------------------------------------------

def test_format_guard_total_bounded_for_typical_completion():
    # Worst-case typical: missing think tag + 2x soft-cap completion.
    completion = "x" * (2 * _LEN_SOFT_CAP_CHARS)
    penalty = format_guard_reasoning([completion])[0]
    # Typical gin_rummy terminal is clipped to +/- 50; guard should stay
    # at most ~few percent of that even on worst-case inputs of reasonable size.
    assert penalty > -5.0, f"format guard too harsh: {penalty}"


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect as _inspect
    mod = sys.modules[__name__]
    tests = [
        (name, fn)
        for name, fn in _inspect.getmembers(mod, _inspect.isfunction)
        if name.startswith("test_")
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
