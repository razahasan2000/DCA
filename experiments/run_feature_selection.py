"""Run the UCI feature-selection benchmark (Section 5.2): 10 algorithms x 30 runs.

Usage:
    python experiments/run_feature_selection.py [--runs 30] [--workers 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcalib.runner import BenchmarkConfig, run_benchmark
from dcalib.problems.feature_selection import FEATURE_PENALTY
from dcalib.algorithms import ALGORITHM_ORDER

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=20260101)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--penalty", type=float, default=FEATURE_PENALTY)
    ap.add_argument("--algos", type=str, default=None,
                    help="comma-separated subset of algorithms to run")
    ap.add_argument("--append", action="store_true",
                    help="merge with existing results instead of overwriting")
    args = ap.parse_args()

    algos = args.algos.split(",") if args.algos else None

    cfg = BenchmarkConfig(
        problem_kind="feature_selection",
        runs=args.runs, pop_size=args.pop, max_iter=args.iters,
        base_seed=args.seed_base, workers=args.workers,
        problem_kwargs={"penalty": args.penalty},
    )
    print(f"Feature-selection benchmark: {cfg.runs} runs x {algos or 'all 10'} "
          f"(N={cfg.pop_size}, MaxIter={cfg.max_iter}, penalty={args.penalty}, "
          f"workers={cfg.workers})")

    df, curves, diversities = run_benchmark(cfg, algorithms=algos)
    elapsed = df.attrs["elapsed"]

    out_raw = ROOT / "results" / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    run_csv = out_raw / "feature_selection_runs.pkl"
    if args.append and run_csv.exists():
        old = pd.read_pickle(run_csv)
        old = old[~old["algorithm"].isin(set(df["algorithm"]))]
        df = pd.concat([old, df], ignore_index=True)
        z_old = np.load(out_raw / "feature_selection_curves.npz")
        for a in ALGORITHM_ORDER:
            key_c, key_d = f"curve__{a}", f"div__{a}"
            if a not in curves and key_c in z_old:
                curves[a] = z_old[key_c]
                diversities[a] = z_old[key_d]
    curves = {a: c for a, c in curves.items()}
    df.to_pickle(run_csv)
    df.drop(columns=["best_solution"]).to_csv(out_raw / "feature_selection_runs.csv", index=False)
    np.savez_compressed(
        out_raw / "feature_selection_curves.npz",
        **{f"curve__{a}": c for a, c in curves.items()},
        **{f"div__{a}": d for a, d in diversities.items()},
    )
    print(f"Done in {elapsed / 60:.1f} min -> {out_raw}/feature_selection_*")


if __name__ == "__main__":
    main()
