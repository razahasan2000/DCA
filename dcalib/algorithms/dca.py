"""The Decorator Crab Algorithm (DCA) -- Algorithm 1 of the manuscript.

Faithful line-by-line implementation of the pseudocode in Section 3.3:

Phase 1 -- Local Improvement (intensification)
    * Greedy Decoration : each crab surveys ``k_survey`` random *available*
      components (bits currently 0) and adopts every surveyed component that
      yields an immediate fitness improvement (pseudocode lines 12-22; the
      loop continues over the whole survey set, adopting each improving
      component as it goes).
    * Probabilistic Pruning : with probability ``p_prune`` the crab removes a
      uniformly random *selected* component and re-evaluates (lines 23-31).

Phase 2 -- Information Exchange (diversification)
    * Targeted Trading : with probability ``p_trade`` per trial, a random
      distinct pair of crabs is chosen; the fitter crab identifies its best
      "gettable" item (present in the weaker crab, absent from itself, ranked
      by a problem-specific quality metric) and its worst "giveable" item
      (present in itself, ranked worst by the same metric).  The simulated
      swap is executed only if it improves the fitter crab (lines 33-57).

Phase 3 -- Selection (elitism)
    * GlobalBest is updated whenever the current population produces a
      superior solution (lines 58-62).
"""

from __future__ import annotations

import time

import numpy as np

from ..problems.base import Problem, population_diversity
from .base import BinaryOptimizer, RunResult


class DCA(BinaryOptimizer):
    name = "DCA"
    is_proposed = True

    def __init__(self, p_prune: float = 0.1, k_survey: int = 4, p_trade: float = 0.2):
        self.p_prune = float(p_prune)
        self.k_survey = int(k_survey)
        self.p_trade = float(p_trade)

    # ------------------------------------------------------------------
    def optimize(self, problem: Problem, rng: np.random.Generator,
                 pop_size: int, max_iter: int) -> RunResult:
        t0 = time.perf_counter()
        d = problem.dim
        n = pop_size

        # --- Initialisation (lines 1-8) -------------------------------
        pop = self._random_binary_pop(n, d, rng)
        fit = np.array([problem.evaluate(ind) for ind in pop])
        gb_idx = int(np.argmax(fit))
        gbest_sol = pop[gb_idx].copy()
        gbest_fit = float(fit[gb_idx])

        curve = np.empty(max_iter)
        diversity = np.empty(max_iter)

        # --- Main loop (line 10) ---------------------------------------
        for t in range(max_iter):
            # === Phase 1: Local Improvement (lines 11-31) =============
            for i in range(n):
                sol = pop[i]
                f_i = float(fit[i])

                # Greedy Decoration (lines 12-22)
                available = np.where(sol == 0)[0]
                if available.size > 0:
                    k = min(self.k_survey, available.size)
                    survey = rng.choice(available, size=k, replace=False)
                    for item in survey:
                        if sol[item] == 1:      # adopted earlier in this pass
                            continue
                        temp = sol.copy()
                        temp[item] = 1
                        temp_fit = problem.evaluate(temp)
                        if temp_fit > f_i:
                            sol, f_i = temp, temp_fit

                # Probabilistic Pruning (lines 23-31)
                if rng.random() < self.p_prune:
                    ones = np.where(sol == 1)[0]
                    if ones.size > 0:
                        sol = sol.copy()
                        sol[rng.choice(ones)] = 0
                        f_i = problem.evaluate(sol)

                pop[i] = sol
                fit[i] = f_i

            # === Phase 2: Information Exchange (lines 33-57) ==========
            for _ in range(n):
                if rng.random() < self.p_trade:
                    c1, c2 = rng.choice(n, size=2, replace=False)
                    if fit[c1] > fit[c2]:
                        fitter, weaker = c1, c2
                    else:
                        fitter, weaker = c2, c1

                    gettable = np.where((pop[weaker] == 1) & (pop[fitter] == 0))[0]
                    giveable = np.where(pop[fitter] == 1)[0]
                    if gettable.size == 0 or giveable.size == 0:
                        continue

                    item_get = gettable[int(np.argmax([problem.quality(int(j)) for j in gettable]))]
                    item_give = giveable[int(np.argmin([problem.quality(int(j)) for j in giveable]))]

                    trade = pop[fitter].copy()
                    trade[item_get] = 1
                    trade[item_give] = 0
                    trade_fit = problem.evaluate(trade)
                    if trade_fit > fit[fitter]:
                        pop[fitter] = trade
                        fit[fitter] = trade_fit

            # === Phase 3: Selection / Elitism (lines 58-62) ===========
            cur = pop[int(np.argmax(fit))]
            cur_fit = float(fit[int(np.argmax(fit))])
            if cur_fit > gbest_fit:
                gbest_fit, gbest_sol = cur_fit, cur.copy()

            curve[t], diversity[t] = gbest_fit, population_diversity(pop)

        return RunResult(
            best_fitness=gbest_fit,
            best_solution=gbest_sol,
            curve=curve,
            diversity=diversity,
            evaluations=problem.n_evaluations,
            wall_time=time.perf_counter() - t0,
            extra={"p_prune": self.p_prune, "k_survey": self.k_survey, "p_trade": self.p_trade},
        )
