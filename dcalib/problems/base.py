"""Abstract problem interface shared by all benchmarks.

Every problem is a *maximisation* problem over binary vectors of a fixed
dimension ``D``.  The class provides:

* ``evaluate(x)`` -- memoised fitness evaluation with an evaluation counter
  (essential for the wrapper-based feature-selection benchmark, where a
  single evaluation trains a classifier).
* ``quality(k)``  -- a problem-specific quality metric for item ``k`` used by
  the DCA targeted-trading operator to identify the "best item to get" and
  the "worst item to give" (Section 3.3 of the manuscript).
"""

from __future__ import annotations

import numpy as np


class Problem:
    """Base class for binary maximisation problems."""

    name: str = "problem"

    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n_evaluations = 0
        self._cache: dict[bytes, float] = {}

    # ------------------------------------------------------------------
    def raw_fitness(self, x: np.ndarray) -> float:
        """Un-memoised fitness of a binary vector (to be overridden)."""
        raise NotImplementedError

    def quality(self, k: int) -> float:
        """Problem-specific desirability of item ``k`` (higher = better)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def evaluate(self, x: np.ndarray) -> float:
        """Memoised fitness evaluation."""
        key = np.packbits(x.astype(np.uint8), axis=None).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        self.n_evaluations += 1
        value = float(self.raw_fitness(np.asarray(x)))
        self._cache[key] = value
        return value

    # ------------------------------------------------------------------
    @property
    def cache_size(self) -> int:
        return len(self._cache)


def population_diversity(pop: np.ndarray) -> float:
    """Normalised mean pairwise Hamming distance of a binary population.

    Computed exactly through per-bit counts:
        sum_{i<j} hamming(x_i, x_j) = sum_b  n_b * (N - n_b)
    where ``n_b`` is the number of ones in bit ``b``.  The result is divided
    by D so that the metric lives in [0, 1].
    """
    pop = np.asarray(pop)
    n = pop.shape[0]
    if n < 2:
        return 0.0
    ones = pop.sum(axis=0)
    total_hamming = float(np.sum(ones * (n - ones)))
    denom = n * (n - 1) / 2.0
    return total_hamming / (denom * pop.shape[1])
