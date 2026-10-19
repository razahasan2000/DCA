"""Grey Wolf Optimizer and Harris Hawks Optimization, adapted to the binary
domain with the manuscript's standard sigmoid transfer function (Section 4.2).

Both keep their canonical continuous dynamics (GWO: a linearly 2 -> 0 with
alpha/beta/delta leadership; HHO: escaping-energy-driven exploration and
soft/hard besiege with Lévy dives) and are binarised stochastically via
P(bit = 1) = sigmoid(z).  The continuous leader positions act as attractors
for the pack.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..problems.base import Problem
from .base import BinaryOptimizer, RunResult, binarize, CONTINUOUS_BOUNDS


class GWO(BinaryOptimizer):
    name = "GWO"

    def __init__(self):
        pass

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
            a = 2.0 - 2.0 * (t + 1) / max_iter              # linearly 2 -> 0
            leaders = np.argsort(-fit)[:3]                   # alpha, beta, delta
            alpha_z, beta_z, delta_z = z[leaders[0]], z[leaders[1]], z[leaders[2]]

            for i in range(n):
                r = rng.random(6)
                A1, A2, A3 = 2 * a * r[0] - a, 2 * a * r[2] - a, 2 * a * r[4] - a
                C1, C2, C3 = 2 * r[1], 2 * r[3], 2 * r[5]

                X1 = alpha_z - A1 * np.abs(C1 * alpha_z - z[i])
                X2 = beta_z - A2 * np.abs(C2 * beta_z - z[i])
                X3 = delta_z - A3 * np.abs(C3 * delta_z - z[i])
                z_new = np.clip((X1 + X2 + X3) / 3.0, lo, hi)

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


def levy_flight(size: int, rng: np.random.Generator, beta: float = 1.5) -> np.ndarray:
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
               / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)
    return u / (np.abs(v) ** (1 / beta))


class HHO(BinaryOptimizer):
    name = "HHO"

    def __init__(self):
        pass

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
            E0 = 2.0 * rng.random() - 1.0
            E = 2.0 * E0 * (1.0 - (t + 1) / max_iter)        # escaping energy
            rabbit = int(np.argmax(fit))
            rabbit_z = z[rabbit]

            E1 = 2.0 * (1.0 - (t + 1) / max_iter)            # rapid-dive energy
            for i in range(n):
                if abs(E) >= 1:                              # exploration
                    r_idx = int(rng.choice([j for j in range(n) if j != i])) if n > 1 else i
                    z_new = z[r_idx] - E * np.abs(2 * rng.random() * z[r_idx] - z[i])
                elif rng.random() >= 0.5:                    # prey seen -- besiege
                    dx = rabbit_z - z[i]
                    if abs(E) >= 0.5:                        # soft besiege
                        j = 2 * (1 - rng.random())
                        z_new = dx - E * np.abs(j * rabbit_z - z[i])
                    else:                                    # hard besiege
                        z_new = rabbit_z - E * np.abs(dx)
                else:                                        # rapid dives
                    lf = levy_flight(d, rng)
                    if abs(E) >= 0.5:                        # soft dive
                        y = rabbit_z - E * np.abs(2 * (1 - rng.random()) * rabbit_z - z[i])
                    else:                                    # hard dive
                        y = rabbit_z - E * np.abs(2 * (1 - rng.random()) * rabbit_z - z.mean(axis=0))
                    if rng.random() < 0.5:
                        z_new = y + E1 * lf                  # Levy dive
                    else:
                        xr = z[int(rng.integers(n))]
                        z_new = y + E1 * (xr - z[i])         # random-jump dive

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

