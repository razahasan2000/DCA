"""Whale Optimization Algorithm and Moth-Flame Optimization, adapted to the
binary domain with the manuscript's standard sigmoid transfer function.

WOA (Table A2: a linearly 2 -> 0, spiral constant b = 1): encircling, spiral
and random-prey phases around the best whale.  MFO (Table A2: flame count
linearly N -> 1, spiral constant b = 1): moths spiral around an
annealed (sorted, shrinking) flame set.  Both binarise stochastically via
P(bit = 1) = sigmoid(z).
"""

from __future__ import annotations

import time

import numpy as np

from ..problems.base import Problem
from .base import BinaryOptimizer, RunResult, binarize, CONTINUOUS_BOUNDS


class WOA(BinaryOptimizer):
    name = "WOA"

    def __init__(self, b: float = 1.0):
        self.b = float(b)

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        n, d = pop_size, problem.dim
        lo, hi = CONTINUOUS_BOUNDS

        z = rng.uniform(lo, hi, size=(n, d))
        pop = binarize(z, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        for t in range(max_iter):
            a = 2.0 - 2.0 * (t + 1) / max_iter
            best = int(np.argmax(fit))
            best_z = z[best]

            for i in range(n):
                r1, r2 = rng.random(), rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = rng.random()
                l = rng.uniform(-1.0, 1.0)

                if p < 0.5:
                    if abs(A) < 1:                      # encircle prey
                        z_new = best_z - A * np.abs(C * best_z - z[i])
                    else:                               # search for prey
                        r_idx = int(rng.choice([j for j in range(n) if j != i])) if n > 1 else i
                        z_new = z[r_idx] - A * np.abs(C * z[r_idx] - z[i])
                else:                                   # spiral update
                    dz = np.abs(best_z - z[i])
                    z_new = dz * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_z

                z_new = np.clip(z_new, lo, hi)
                cand = binarize(z_new, rng)
                cand_fit = problem.evaluate(cand)
                if cand_fit > fit[i]:
                    z[i], pop[i], fit[i] = z_new, cand, cand_fit

            best = int(np.argmax(fit))
            curve[t], diversity[t] = float(fit[best]), self._record(pop, 0)[1]

        best = int(np.argmax(fit))
        return RunResult(
            best_fitness=float(fit[best]),
            best_solution=pop[best].copy(),
            curve=curve,
            diversity=diversity,
            evaluations=problem.n_evaluations,
            wall_time=time.perf_counter() - t0,
        )


class MFO(BinaryOptimizer):
    name = "MFO"

    def __init__(self, b: float = 1.0):
        self.b = float(b)

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        n, d = pop_size, problem.dim
        lo, hi = CONTINUOUS_BOUNDS

        z = rng.uniform(lo, hi, size=(n, d))
        pop = binarize(z, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        for t in range(max_iter):
            # Flames: population sorted by fitness, shrinking count N -> 1.
            order = np.argsort(-fit)
            n_flames = max(1, int(round(n - (t + 1) * (n - 1) / max_iter)))
            flame_z = z[order[:n_flames]]
            flame_fit = fit[order[:n_flames]]

            a = -1.0 + (t + 1) * (-1.0 / max_iter)      # spiral span [a, 1]

            for i in range(n):
                flame_no = i if i < n_flames else int(rng.integers(n_flames))
                fz = flame_z[flame_no]
                dist = np.abs(fz - z[i])
                l = rng.uniform(a, 1.0)
                z_new = dist * np.exp(self.b * l) * np.cos(2 * np.pi * l) + fz
                z_new = np.clip(z_new, lo, hi)

                cand = binarize(z_new, rng)
                cand_fit = problem.evaluate(cand)
                if cand_fit > fit[i]:
                    z[i], pop[i], fit[i] = z_new, cand, cand_fit

            best = int(np.argmax(fit))
            curve[t], diversity[t] = float(fit[best]), self._record(pop, 0)[1]

        best = int(np.argmax(fit))
        return RunResult(
            best_fitness=float(fit[best]),
            best_solution=pop[best].copy(),
            curve=curve,
            diversity=diversity,
            evaluations=problem.n_evaluations,
            wall_time=time.perf_counter() - t0,
        )
