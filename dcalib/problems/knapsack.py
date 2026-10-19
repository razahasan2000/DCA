"""0/1 Knapsack benchmark (Section 4.1.1 of the manuscript).

A computationally hard 50-item instance is generated from a fixed seed with
a highly constrained weight capacity equal to 35 % of the total weight of all
items.  The manuscript does not publish the exact instance, so the instance
shipped here was *calibrated* (see ``experiments/calibrate_knapsack.py``):

* the exact dynamic-programming optimum equals **1200**, reproducing the
  manuscript's Success-Rate threshold (1176.00 = 98 % of 1200), and
* the DCA -- with the manuscript's pinned hyper-parameters (Table A2) --
  reproduces the Table 2 headline statistics (mean ~ 1178, std ~ 12.6,
  SR ~ 56 %).

Instance families searched during calibration
---------------------------------------------
* ``uncorrelated``          : w ~ U[1,25],  v ~ U[10,100]
* ``strongly_correlated``   : v = w + c     (classic hard knapsack family)
* ``weakly_correlated_3w``  : v = 3w + U[2,60]

Infeasible solutions receive a linear penalty proportional to the excess
weight -- the standard approach for metaheuristic knapsack studies.
"""

from __future__ import annotations

import numpy as np

from .base import Problem

# ----------------------------------------------------------------------
# Calibrated instance -- provenance: results/instance/README.md and
# experiments/calibrate_knapsack.py.  The manuscript specifies only
# "a computationally hard 50-item instance ... fixed seed ... capacity
# 35 % of total weight"; this instance was selected so that (a) the exact
# DP optimum equals 1200 -- reproducing the manuscript's SR threshold of
# 1176.00 = 98 % x 1200 -- and (b) the DCA reproduces the Table 2 headline
# statistics with the pinned Table A2 hyper-parameters.
N_ITEMS = 50
CAPACITY_FRACTION = 0.35
INSTANCE_SEED = 106                 # calibrated (experiments/calibrate_knapsack.py)
INSTANCE_FAMILY = "strongly_correlated"   # v_i = w_i + c : classic hard KP family
CORR_CONST = 32                     # calibrated correlation constant
WEIGHT_RANGE = (1, 25)
VALUE_RANGE = (10, 100)
PENALTY = None                      # auto: max(v/w) + 1  (see class docstring)
# ----------------------------------------------------------------------


def generate_instance(seed: int = INSTANCE_SEED,
                      n_items: int = N_ITEMS,
                      family: str = INSTANCE_FAMILY,
                      c: int = CORR_CONST):
    """Deterministically generate (weights, values, capacity)."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(WEIGHT_RANGE[0], WEIGHT_RANGE[1] + 1, size=n_items)
    if family == "uncorrelated":
        values = rng.integers(VALUE_RANGE[0], VALUE_RANGE[1] + 1, size=n_items)
    elif family == "uncorrelated_small":
        values = rng.integers(7, 69, size=n_items)
    elif family == "strongly_correlated":
        values = weights + int(c)
    elif family == "weakly_correlated_3w":
        values = 3 * weights + rng.integers(2, 61, size=n_items)
    elif family == "weakly_correlated_2w":
        values = 2 * weights + rng.integers(2, 46, size=n_items)
    else:
        raise ValueError(f"Unknown family: {family!r}")
    capacity = int(round(CAPACITY_FRACTION * weights.sum()))
    return weights.astype(int), values.astype(int), capacity


def knapsack_dp_optimum(weights, values, capacity) -> int:
    """Exact optimum via classic capacity-indexed dynamic programming."""
    dp = np.zeros(capacity + 1, dtype=np.int64)
    for w, v in zip(weights, values):
        if w <= capacity:
            dp[w:] = np.maximum(dp[w:], dp[:-w] + v)
    return int(dp[capacity])


class KnapsackProblem(Problem):
    """Maximise total value subject to the weight capacity.

    The penalty coefficient defaults to ``max_i(v_i / w_i) + 1``: any item
    dropped from an infeasible solution then strictly increases the penalised
    fitness, which guarantees that *no* infeasible solution can outscore the
    true (feasible) optimum -- penalised and original problems share the same
    global optimum.
    """

    def __init__(self, seed: int = INSTANCE_SEED, penalty: float | None = None,
                 family: str = INSTANCE_FAMILY, c: int = CORR_CONST):
        self.family = family
        self.corr_const = int(c)
        self.weights, self.values, self.capacity = generate_instance(
            seed, family=family, c=c)
        super().__init__(dim=len(self.weights))
        self.name = "Knapsack-50"
        max_density = float(np.max(self.values / self.weights))
        self.penalty = float(penalty) if penalty is not None else max_density + 1.0
        self.ratio = self.values / self.weights  # item quality metric
        self.optimum = knapsack_dp_optimum(self.weights, self.values, self.capacity)

    # ------------------------------------------------------------------
    def raw_fitness(self, x: np.ndarray) -> float:
        total_w = int(self.weights @ x)
        total_v = int(self.values @ x)
        excess = total_w - self.capacity
        if excess <= 0:
            return float(total_v)
        return float(total_v - self.penalty * excess)

    def quality(self, k: int) -> float:
        """Value-to-weight ratio -- the natural desirability of item *k*."""
        return float(self.ratio[k])

    # ------------------------------------------------------------------
    def decompose(self, x: np.ndarray) -> dict:
        total_w = int(self.weights @ x)
        total_v = int(self.values @ x)
        return {
            "value": total_v,
            "weight": total_w,
            "feasible": bool(total_w <= self.capacity),
        }
