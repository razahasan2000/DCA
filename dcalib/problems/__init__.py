"""Problem factory."""

from __future__ import annotations

from .base import Problem, population_diversity
from .knapsack import KnapsackProblem
from .feature_selection import FeatureSelectionProblem


def make_problem(kind: str, **kwargs) -> Problem:
    kind = kind.lower()
    if kind in ("knapsack", "kp"):
        return KnapsackProblem(**kwargs)
    if kind in ("feature_selection", "fs", "uci"):
        return FeatureSelectionProblem(**kwargs)
    raise ValueError(f"Unknown problem kind: {kind!r}")


__all__ = [
    "Problem",
    "KnapsackProblem",
    "FeatureSelectionProblem",
    "population_diversity",
    "make_problem",
]
