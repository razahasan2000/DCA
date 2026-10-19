"""Hyperparameter sensitivity analysis (Section 5.4, Figure 6).

One-Factor-at-a-Time study of the DCA on the knapsack benchmark:
  * trading probability  p_trade  in {0.0, 0.1, ..., 0.8}
  * survey size          k_survey in {3, 5, 7, 10, 12, 15}
All other conditions fixed (Table A1/A2 settings).

Usage:
    python experiments/run_sensitivity.py [--runs 30] [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcalib.problems import KnapsackProblem
from dcalib.algorithms import DCA

ROOT = Path(__file__).resolve().parents[1]
POP, MAX_ITER = 30, 100
P_TRADE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
K_SURVEY_GRID = [3, 5, 7, 10, 12, 15]


def sweep(param: str, grid: list, runs: int, seed_base: int, fixed: dict) -> pd.DataFrame:
    rows = []
    for value in grid:
        params = dict(fixed)
        params[param] = value
        prob = KnapsackProblem()
        algo = DCA(**params)
        finals = []
        for r in range(runs):
            res = algo.optimize(prob, np.random.default_rng(seed_base + r), POP, MAX_ITER)
            finals.append(res.best_fitness)
        finals = np.asarray(finals)
        rows.append({"parameter": param, "value": value, "runs": runs,
                     "mean_fitness": finals.mean(),
                     "std_fitness": finals.std(ddof=1)})
        print(f"  {param}={value}: mean={finals.mean():.2f} ± {finals.std(ddof=1):.2f}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=20260101)
    ap.add_argument("--workers", type=int, default=1, help="kept for CLI symmetry")
    args = ap.parse_args()

    fixed = {"p_prune": 0.1, "k_survey": 4, "p_trade": 0.2}
    print(f"Sensitivity analysis ({args.runs} runs per setting, knapsack):")
    print("p_trade sweep (k_survey=4 fixed):")
    df_trade = sweep("p_trade", P_TRADE_GRID, args.runs, args.seed_base, fixed)
    print("k_survey sweep (p_trade=0.2 fixed):")
    df_survey = sweep("k_survey", K_SURVEY_GRID, args.runs, args.seed_base, fixed)

    out = ROOT / "results" / "raw" / "sensitivity.csv"
    pd.concat([df_trade, df_survey], ignore_index=True).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
