"""Parallel benchmark runner.

Executes (algorithm x run) tasks over a multiprocessing pool with one
``Problem`` instance per worker (the evaluation cache therefore persists
across runs within a worker -- highly effective once populations converge).
Every run is seeded deterministically: ``seed = base_seed + run_id``.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .problems import make_problem
from .algorithms import build_algorithms, ALGORITHM_ORDER


@dataclass
class BenchmarkConfig:
    problem_kind: str                       # "knapsack" | "feature_selection"
    runs: int
    pop_size: int
    max_iter: int
    base_seed: int = 20260101
    workers: int = 2
    problem_kwargs: dict | None = None      # e.g. {"penalty": 0.001} for FS


_GLOBAL_PROBLEM = None
_GLOBAL_CONFIG = None


def _init_worker(problem_kind: str, problem_kwargs: dict | None):
    global _GLOBAL_PROBLEM
    _GLOBAL_PROBLEM = make_problem(problem_kind, **(problem_kwargs or {}))
    # Avoid thread over-subscription on small machines.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _run_one(args) -> dict:
    algo_name, run_id, cfg = args
    problem = _GLOBAL_PROBLEM
    before = problem.n_evaluations
    from .algorithms import build_algorithms
    algo = build_algorithms()[algo_name]

    rng = np.random.default_rng(cfg.base_seed + run_id)
    result = algo.optimize(problem, rng, cfg.pop_size, cfg.max_iter)

    row = {
        "algorithm": algo_name,
        "run": run_id,
        "final_fitness": result.best_fitness,
        "evaluations": result.evaluations - before,
        "wall_time": result.wall_time,
        "best_solution": np.packbits(result.best_solution.astype(np.uint8)).tobytes().hex(),
        "curve": result.curve,
        "diversity": result.diversity,
    }
    for key, val in result.extra.items():
        row[key] = val
    if problem.name.startswith("UCI"):
        dec = problem.decompose(result.best_solution)
        row["accuracy"] = dec["accuracy"]
        row["n_features"] = dec["n_features"]
    return row


def run_benchmark(cfg: BenchmarkConfig, algorithms: list[str] | None = None):
    """Run every (algorithm, run) pair.

    Returns ``(df, curves, diversities)`` where *df* holds the per-run scalar
    records and *curves* / *diversities* map algorithm name -> an array of
    shape (n_runs, max_iter).
    """
    algorithms = algorithms or ALGORITHM_ORDER
    problem_kwargs = cfg.problem_kwargs or {}
    # Sanity: instantiate the problem once in the parent to fail fast.
    make_problem(cfg.problem_kind, **problem_kwargs)

    tasks = [(a, r, cfg) for a in algorithms for r in range(cfg.runs)]
    t0 = time.perf_counter()
    if cfg.workers > 1:
        with mp.get_context("spawn").Pool(
            processes=cfg.workers,
            initializer=_init_worker,
            initargs=(cfg.problem_kind, problem_kwargs),
        ) as pool:
            rows = pool.map(_run_one, tasks, chunksize=1)
    else:
        _init_worker(cfg.problem_kind, problem_kwargs)
        rows = [_run_one(t) for t in tasks]
    elapsed = time.perf_counter() - t0

    max_iter = cfg.max_iter
    curves = {a: np.empty((cfg.runs, max_iter)) for a in algorithms}
    diversities = {a: np.empty((cfg.runs, max_iter)) for a in algorithms}
    scalars = []
    for row in rows:
        a = row["algorithm"]                 # keep algorithm in the scalar record
        curves[a][row["run"]] = row.pop("curve")
        diversities[a][row["run"]] = row.pop("diversity")
        scalars.append(row)

    df = pd.DataFrame(scalars)
    df.attrs["elapsed"] = elapsed
    df.attrs["problem"] = cfg.problem_kind
    return df, curves, diversities
