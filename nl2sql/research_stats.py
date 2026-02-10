"""
Statistical helpers for reproducible NL->SQL comparisons.

These utilities support dissertation-style claims by quantifying uncertainty and
paired differences (e.g., baseline vs few-shot on the same items).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli rate."""
    if n <= 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1.0 + (z**2) / n
    center = (phat + (z**2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((phat * (1.0 - phat) / n) + (z**2) / (4.0 * (n**2)))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_switch_counts(
    left: Sequence[int | bool | None],
    right: Sequence[int | bool | None],
) -> tuple[int, int, int]:
    """
    Return (n, left_to_right_improvements, left_to_right_degradations).

    Inputs are paired per-example outcomes for the same metric and examples.
    Missing values in either side are skipped.
    """
    n = 0
    improve = 0
    degrade = 0
    for l_raw, r_raw in zip(left, right):
        if l_raw is None or r_raw is None:
            continue
        l = int(bool(l_raw))
        r = int(bool(r_raw))
        n += 1
        if r > l:
            improve += 1
        elif r < l:
            degrade += 1
    return n, improve, degrade


def _binom_two_sided_p(n: int, k: int, p: float = 0.5) -> float:
    """
    Exact two-sided binomial p-value for observing k successes in n trials.
    """
    if n <= 0:
        return math.nan

    def pmf(x: int) -> float:
        return math.comb(n, x) * (p**x) * ((1.0 - p) ** (n - x))

    p_obs = pmf(k)
    total = 0.0
    for x in range(n + 1):
        px = pmf(x)
        if px <= p_obs + 1e-12:
            total += px
    return min(1.0, total)


def mcnemar_exact_p(improved: int, degraded: int) -> float:
    """
    Exact McNemar test p-value (binomial form) for paired binary outcomes.
    """
    n_discordant = improved + degraded
    if n_discordant == 0:
        return 1.0
    k = min(improved, degraded)
    return _binom_two_sided_p(n_discordant, k, p=0.5)


def summarize_binary(values: Iterable[int | bool | None]) -> dict[str, float | int]:
    """
    Summarize a binary metric with Wilson interval.
    """
    clean = [int(bool(v)) for v in values if v is not None]
    n = len(clean)
    successes = sum(clean)
    rate = successes / n if n else math.nan
    lo, hi = wilson_interval(successes, n)
    return {
        "n": n,
        "successes": successes,
        "rate": rate,
        "ci_low": lo,
        "ci_high": hi,
    }


def categorical_counts(values: Iterable[str]) -> dict[str, int]:
    """
    Convenience counter with deterministic key ordering.
    """
    c = Counter(values)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
