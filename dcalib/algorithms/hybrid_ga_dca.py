"""Hybrid GA-DCA (Section 4.2 "Proposed Variants" + Table A2).

The deliberately flawed variant analysed in Section 5.3 of the manuscript:
a GA framework whose offspring receive an aggressive DCA local-search
(polishing) component.  Per Table A2 it uses one-point crossover (p = 0.8),
tournament selection (size 3) and **5 DCA local-search steps after
crossover**; no bit mutation is listed for this variant, which -- combined
with the aggressive local search -- is what destroys population diversity
(the "pathological exploitation" behaviour of Figures 3-5).

A single *DCA step* = one greedy-decoration pass: survey ``k_survey`` random
available components and adopt improving ones (the DCA intensification
operator with k_survey = 4, inherited from the DCA row of Table A2).
"""

from __future__ import annotations

import time

import numpy as np

from ..problems.base import Problem
from .base import BinaryOptimizer, RunResult
from .ga import tournament_select, one_point_crossover


class HybridGADCA(BinaryOptimizer):
    name = "Hybrid GA-DCA"
    is_variant = True

    def __init__(self, p_crossover: float = 0.8, tournament_size: int = 3,
                 local_search_steps: int = 5, k_survey: int = 4, elitism: int = 1):
        self.pc = float(p_crossover)
        self.k_tour = int(tournament_size)
        self.ls_steps = int(local_search_steps)
        self.k_survey = int(k_survey)
        self.elitism = int(elitism)

    # ------------------------------------------------------------------
    def _decorate_pass(self, sol: np.ndarray, f_i: float, problem: Problem,
                       rng: np.random.Generator):
        """One greedy-decoration pass (DCA intensification operator)."""
        available = np.where(sol == 0)[0]
        if available.size == 0:
            return sol, f_i
        k = min(self.k_survey, available.size)
        survey = rng.choice(available, size=k, replace=False)
        for item in survey:
            if sol[item] == 1:
                continue
            temp = sol.copy()
            temp[item] = 1
            temp_fit = problem.evaluate(temp)
            if temp_fit > f_i:
                sol, f_i = temp, temp_fit
        return sol, f_i

    # ------------------------------------------------------------------
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
                        # Aggressive DCA local search after crossover
                        # (NO mutation -- per Table A2).
                        sol, f_i = child, problem.evaluate(child)
                        for _ in range(self.ls_steps):
                            sol, f_i = self._decorate_pass(sol, f_i, problem, rng)
                        new_pop.append(sol)

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
