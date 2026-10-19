"""Standard Differential Evolution adapted to the binary domain (Table A2).

Continuous DE/rand/1/bin (F = 0.5, CR = 0.7) with the manuscript's standard
sigmoid transfer function mapping continuous positions to bit probabilities
(Section 4.2).  Greedy pairwise selection is performed on the *binarised*
trial vector.
"""

from __future__ import annotations

import time

import numpy as np

from ..problems.base import Problem
from .base import BinaryOptimizer, RunResult, binarize, sigmoid, CONTINUOUS_BOUNDS


class DE(BinaryOptimizer):
    name = "Standard DE"

    def __init__(self, f: float = 0.5, cr: float = 0.7, bounds=(-20.0, 20.0)):
        self.f = float(f)
        self.cr = float(cr)
        self.bounds = bounds

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        n, d = pop_size, problem.dim
        lo, hi = self.bounds

        # Initialised in the standard continuous range, then allowed to
        # saturate within the wider adaptation bounds (calibrated choice,
        # see docs/ALGORITHMS.md).
        z = rng.uniform(CONTINUOUS_BOUNDS[0], CONTINUOUS_BOUNDS[1], size=(n, d))
        pop = binarize(z, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        for t in range(max_iter):
            for i in range(n):
                r1, r2, r3 = rng.choice(n, size=3, replace=False)
                mutant = z[r1] + self.f * (z[r2] - z[r3])
                cross = rng.random(d) < self.cr
                if not cross.any():
                    cross[rng.integers(d)] = True
                trial = np.where(cross, mutant, z[i])
                trial = np.clip(trial, lo, hi)

                trial_bin = binarize(trial, rng)
                trial_fit = problem.evaluate(trial_bin)
                if trial_fit > fit[i]:
                    z[i], pop[i], fit[i] = trial, trial_bin, trial_fit

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


class BPSO(BinaryOptimizer):
    """Binary Particle Swarm Optimization (Table A2: w = 0.5, c1 = c2 = 1.5).

    Classic Kennedy-&-Eberhart binary PSO: velocities are updated in
    continuous space and bits are sampled from sigmoid(velocity).
    """

    name = "Binary PSO"

    def __init__(self, w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
                 v_max: float = 6.0):
        self.w, self.c1, self.c2 = float(w), float(c1), float(c2)
        self.v_max = float(v_max)

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        n, d = pop_size, problem.dim

        pop = self._random_binary_pop(n, d, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])
        v = rng.uniform(-1.0, 1.0, size=(n, d))

        pbest = pop.copy()
        pbest_fit = fit.copy()
        g_idx = int(np.argmax(fit))
        gbest, gbest_fit = pop[g_idx].copy(), float(fit[g_idx])

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        for t in range(max_iter):
            r1 = rng.random((n, d))
            r2 = rng.random((n, d))
            v = (self.w * v
                 + self.c1 * r1 * (pbest - pop)
                 + self.c2 * r2 * (gbest[None, :] - pop))
            v = np.clip(v, -self.v_max, self.v_max)
            pop = (rng.random((n, d)) < sigmoid(v)).astype(int)
            fit = np.array([problem.evaluate(ind) for ind in pop])

            improved = fit > pbest_fit
            pbest[improved] = pop[improved]
            pbest_fit[improved] = fit[improved]
            g = int(np.argmax(pbest_fit))
            if pbest_fit[g] > gbest_fit:
                gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])

            curve[t], diversity[t] = gbest_fit, self._record(pop, 0)[1]

        return RunResult(
            best_fitness=gbest_fit,
            best_solution=gbest,
            curve=curve,
            diversity=diversity,
            evaluations=problem.n_evaluations,
            wall_time=time.perf_counter() - t0,
        )
