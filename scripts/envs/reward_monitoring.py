"""Passive reward-hacking probes and an active format guard.

All callables follow the GRPO reward-func signature:
    func(completions: list[str], **kwargs) -> list[float]

The trainer logs each function's mean/std under ``rewards/{func.__name__}``
(see train_grpo_env.py:375-379). Probes return their measurement and are
wired with weight 0.0 so they do not contribute to the training loss.
The format guard returns a non-positive penalty and is wired with weight
1.0 in reasoning mode only.

Extra rollout fields consumed from ``**kwargs``:
    - terminal_raw:  list[float]   raw environment terminal reward
    - shaping_sum:   list[float]   shaped - terminal (total shaping magnitude)
    - invalid_count: list[int]     illegal-action count per episode

Missing keys resolve to NaN so the trainer's nanmean/nanstd at
train_grpo_env.py:376-379 ignore envs that don't export them.
"""

import math
import re
from typing import List

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_PENALTY = 0.05
_LEN_SOFT_CAP_CHARS = 1536 * 4
_LEN_PENALTY_PER_OVER_CHAR = 0.0005


def _nan_list(n: int) -> List[float]:
    return [float("nan")] * n


def shaping_dominance_probe(completions, **kwargs) -> List[float]:
    terms = kwargs.get("terminal_raw") or _nan_list(len(completions))
    shape = kwargs.get("shaping_sum") or _nan_list(len(completions))
    out: List[float] = []
    for t, s in zip(terms, shape):
        if t is None or s is None:
            out.append(float("nan"))
            continue
        if isinstance(t, float) and math.isnan(t):
            out.append(float("nan"))
            continue
        if isinstance(s, float) and math.isnan(s):
            out.append(float("nan"))
            continue
        denom = abs(t) + abs(s)
        out.append(abs(s) / denom if denom > 1e-9 else 0.0)
    return out


shaping_dominance_probe.__name__ = "probe_shaping_dominance"


def length_probe(completions, **kwargs) -> List[float]:
    return [float(len(c)) for c in completions]


length_probe.__name__ = "probe_completion_length"


def invalid_count_probe(completions, **kwargs) -> List[float]:
    invs = kwargs.get("invalid_count") or _nan_list(len(completions))
    return [float(v) if v is not None else float("nan") for v in invs]


invalid_count_probe.__name__ = "probe_invalid_count"


def terminal_raw_probe(completions, **kwargs) -> List[float]:
    terms = kwargs.get("terminal_raw") or _nan_list(len(completions))
    return [float(t) if t is not None else float("nan") for t in terms]


terminal_raw_probe.__name__ = "probe_terminal_raw"


def format_guard_reasoning(completions, **kwargs) -> List[float]:
    out: List[float] = []
    for c in completions:
        p = 0.0
        matches = _THINK_RE.findall(c)
        if len(matches) != 1 or not matches[0].strip():
            p -= _THINK_PENALTY
        if len(c) > _LEN_SOFT_CAP_CHARS:
            p -= _LEN_PENALTY_PER_OVER_CHAR * (len(c) - _LEN_SOFT_CAP_CHARS)
        out.append(p)
    return out


format_guard_reasoning.__name__ = "format_guard"
