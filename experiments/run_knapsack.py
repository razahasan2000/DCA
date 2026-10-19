"""Run the Knapsack benchmark (Section 5.1): 10 algorithms x 50 runs.

Usage:
    python experiments/run_knapsack.py [--runs 50] [--workers 2] [--fast]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcalib.runner import BenchmarkConfig, run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=20260101)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = BenchmarkConfig(
        problem_kind="knapsack",
        runs=args.runs, pop_size=args.pop, max_iter=args.iters,
        base_seed=args.seed_base, workers=args.workers,
    )
    print(f"Knapsack benchmark: {cfg.runs} runs x 10 algorithms "
          f"(N={cfg.pop_size}, MaxIter={cfg.max_iter}, workers={cfg.workers})")

    df, curves, diversities = run_benchmark(cfg)
    elapsed = df.attrs["elapsed"]

    out_raw = ROOT / "results" / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["best_solution"]).to_csv(out_raw / "knapsack_runs.csv", index=False)
    df.to_pickle(out_raw / "knapsack_runs.pkl")
    np.savez_compressed(
        out_raw / "knapsack_curves.npz",
        **{f"curve__{a}": c for a, c in curves.items()},
        **{f"div__{a}": d for a, d in diversities.items()},
    )
    print(f"Done in {elapsed / 60:.1f} min -> {out_raw}/knapsack_*")


if __name__ == "__main__":
    main()
