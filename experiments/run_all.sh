#!/usr/bin/env bash
# Reproduce every artifact of the study end-to-end.
# Usage:  bash experiments/run_all.sh [--fast]
#   --fast : reduced run counts (knapsack 20, FS 10, sensitivity 10 runs)
set -euo pipefail
cd "$(dirname "$0")/.."

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

if [ "$FAST" -eq 1 ]; then
    KP_RUNS=20; FS_RUNS=10; SENS_RUNS=10
else
    KP_RUNS=50; FS_RUNS=30; SENS_RUNS=30   # Table A1 protocol
fi

WORKERS="${WORKERS:-2}"

echo "== [1/6] Knapsack benchmark: 10 algorithms x ${KP_RUNS} runs =="
python3 experiments/run_knapsack.py --runs "$KP_RUNS" --workers "$WORKERS"

echo "== [2/6] Feature selection benchmark: 10 algorithms x ${FS_RUNS} runs =="
# Batched by algorithm so the suite also runs on modest machines / CI.
python3 experiments/run_feature_selection.py --runs "$FS_RUNS" --workers "$WORKERS" \
        --algos "Standard DE,Binary PSO,GWO,WOA,MFO"
python3 experiments/run_feature_selection.py --runs "$FS_RUNS" --workers "$WORKERS" \
        --algos "Standard GA,ABC" --append
python3 experiments/run_feature_selection.py --runs "$FS_RUNS" --workers "$WORKERS" \
        --algos "DCA,HHO" --append
python3 experiments/run_feature_selection.py --runs "$FS_RUNS" --workers "$WORKERS" \
        --algos "Hybrid GA-DCA" --append

echo "== [3/6] DCA hyperparameter sensitivity (Fig. 6) =="
python3 experiments/run_sensitivity.py --runs "$SENS_RUNS"

echo "== [4/6] Tables 2 and 3 + Friedman/Nemenyi statistics =="
python3 experiments/make_tables.py

echo "== [5/6] Figures 2-6 =="
python3 experiments/make_figures.py

echo "== [6/6] Claim-by-claim verification report =="
python3 experiments/verify_claims.py

echo "All artifacts written to results/ and verification/."
