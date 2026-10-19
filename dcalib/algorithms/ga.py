"""Standard binary Genetic Algorithm (Section 4.2 + Table A2).

Generational GA with:
* binary tournament selection (tournament size 3),
* one-point crossover with probability p_crossover = 0.8,
* independent per-bit mutation with probability p_mutation = 0.03,
* one elite individual carried over unchanged (standard practice; the
  manuscript does not specify elitism, but every competitor is otherwise
  compared against the explicitly elitist DCA).
"""

from __future__ import annotations

import time

import numpy as np

from ..problems.base import Problem
from .base import BinaryOptimizer, RunResult


def tournament_select(pop: np.ndarray, fit: np.ndarray, k: int,
                      rng: np.random.Generator) -> np.ndarray:
    cand = rng.integers(0, len(pop), size=k)
    return pop[cand[int(np.argmax(fit[cand]))]].copy()


def one_point_crossover(p1: np.ndarray, p2: np.ndarray, pc: float,
                        rng: np.random.Generator):
    if rng.random() < pc:
        cut = int(rng.integers(1, len(p1)))
        c1 = np.concatenate([p1[:cut], p2[cut:]])
        c2 = np.concatenate([p2[:cut], p1[cut:]])
        return c1, c2
    return p1.copy(), p2.copy()


def bit_flip_mutation(x: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < pm
    x = x.copy()
    x[mask] = 1 - x[mask]
    return x


class GA(BinaryOptimizer):
    name = "Standard GA"

    def __init__(self, p_crossover: float = 0.8, p_mutation: float = 0.03,
                 tournament_size: int = 3, elitism: int = 1):
        self.pc = float(p_crossover)
        self.pm = float(p_mutation)
        self.k_tour = int(tournament_size)
        self.elitism = int(elitism)

    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        n, d = pop_size, problem.dim

        pop = self._random_binary_pop(n, d, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        for t in range(max_iter):
            order = np.argsort(-fit)
            new_pop = [pop[i].copy() for i in order[:self.elitism]]

            while len(new_pop) < n:
                p1 = tournament_select(pop, fit, self.k_tour, rng)
                p2 = tournament_select(pop, fit, self.k_tour, rng)
                c1, c2 = one_point_crossover(p1, p2, self.pc, rng)
                for child in (c1, c2):
                    if len(new_pop) < n:
                        new_pop.append(bit_flip_mutation(child, self.pm, rng))

            pop = np.array(new_pop[:n])
            fit = np.array([problem.evaluate(ind) for ind in pop])

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
