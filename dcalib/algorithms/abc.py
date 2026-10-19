"""Artificial Bee Colony adapted to the binary domain (Table A2: limit = 20).

Continuous ABC core (employed bees, onlooker roulette-wheel recruitment,
scout abandonment) with the manuscript's standard sigmoid transfer function
binarising positions.  Fitness-shifted roulette keeps the onlooker phase
well-defined for penalised (possibly negative) fitness values.
"""

from __future__ import annotations

import time

import numpy as np

from ..problems.base import Problem
from .base import BinaryOptimizer, RunResult, binarize, CONTINUOUS_BOUNDS


class ABC(BinaryOptimizer):
    name = "ABC"

    def __init__(self, limit: int = 20):
        self.limit = int(limit)

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        n, d = pop_size, problem.dim
        lo, hi = CONTINUOUS_BOUNDS

        z = rng.uniform(lo, hi, size=(n, d))
        pop = binarize(z, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])
        trials = np.zeros(n, dtype=int)

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        def employee_phase():
            for i in range(n):
                k = int(rng.choice([j for j in range(n) if j != i])) if n > 1 else i
                jdim = int(rng.integers(d))
                phi = rng.uniform(-1.0, 1.0)
                cand_z = z[i].copy()
                cand_z[jdim] = z[i, jdim] + phi * (z[i, jdim] - z[k, jdim])
                cand_z = np.clip(cand_z, lo, hi)
                cand = binarize(cand_z, rng)
                cand_fit = problem.evaluate(cand)
                if cand_fit > fit[i]:
                    z[i], pop[i], fit[i] = cand_z, cand, cand_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

        def onlooker_phase():
            shifted = fit - fit.min()
            denom = shifted.sum()
            if denom <= 0:
                probs = np.full(n, 1.0 / n)
            else:
                probs = shifted / denom
            for _ in range(n):
                i = int(rng.choice(n, p=probs))
                k = int(rng.choice([j for j in range(n) if j != i])) if n > 1 else i
                jdim = int(rng.integers(d))
                phi = rng.uniform(-1.0, 1.0)
                cand_z = z[i].copy()
                cand_z[jdim] = z[i, jdim] + phi * (z[i, jdim] - z[k, jdim])
                cand_z = np.clip(cand_z, lo, hi)
                cand = binarize(cand_z, rng)
                cand_fit = problem.evaluate(cand)
                if cand_fit > fit[i]:
                    z[i], pop[i], fit[i] = cand_z, cand, cand_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

        def scout_phase():
            abandoned = np.where(trials > self.limit)[0]
            for i in abandoned:
                z[i] = rng.uniform(lo, hi, size=d)
                pop[i] = binarize(z[i], rng)
                fit[i] = problem.evaluate(pop[i])
                trials[i] = 0

        for t in range(max_iter):
            employee_phase()
            onlooker_phase()
            scout_phase()
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
