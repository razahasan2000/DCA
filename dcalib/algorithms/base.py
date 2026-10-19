"""Shared optimizer infrastructure.

All algorithms are *maximisers* over binary vectors and share the same
protocol:

* ``optimize(problem, rng, pop_size, max_iter)`` returns a ``RunResult``
  holding the best-so-far convergence curve, the per-iteration population
  diversity, the number of fitness evaluations and the final best solution.
* Continuous-domain algorithms (DE, GWO, HHO, WOA, MFO, ABC) are adapted to
  the binary domain with a standard sigmoid transfer function, exactly as
  described in Section 4.2 of the manuscript: a continuous position ``z`` is
  mapped to the probability of the bit being 1 via ``sigmoid(z)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..problems.base import Problem, population_diversity

# Continuous search-space bounds used by the sigmoid-transfer adaptations.
CONTINUOUS_BOUNDS = (-5.0, 5.0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def binarize(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stochastic sigmoid binarization: P(bit = 1) = sigmoid(z)."""
    return (rng.random(z.shape) < sigmoid(z)).astype(int)


@dataclass
class RunResult:
    best_fitness: float
    best_solution: np.ndarray
    curve: np.ndarray                 # best-so-far fitness per iteration
    diversity: np.ndarray             # population diversity per iteration
    evaluations: int
    wall_time: float
    extra: dict = field(default_factory=dict)


class BinaryOptimizer:
    name: str = "base"

    # Sub-populations of results tables ---------------------------------
    is_proposed = False               # DCA itself
    is_variant = False                # Hybrid GA-DCA

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        raise NotImplementedError

    # ------------------------------------------------------------------
    @staticmethod
    def _random_binary_pop(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.random((n, d)) < 0.5).astype(int)

    @staticmethod
    def _record(pop: np.ndarray, best_fitness: float):
        return best_fitness, population_diversity(pop)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return self.name
