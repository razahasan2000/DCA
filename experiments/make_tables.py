"""Build Tables 2 and 3 (CSV + Markdown) from the raw benchmark results.

Table 2 -- Knapsack: mean fitness, std dev, Success Rate (SR) with the
manuscript's definition (runs achieving >= 98 % of the best-known solution
found across the entire experiment).

Table 3 -- Feature selection: mean accuracy, std, mean features, std.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcalib.algorithms import ALGORITHM_ORDER
from dcalib.stats import full_analysis

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def score_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby("algorithm")[value_col]
    out = pd.DataFrame({
        "Mean": g.mean(),
        "Std": g.std(ddof=1),
    })
    best_overall = df[value_col].max()
    out["SuccessRate (%)"] = df.groupby("algorithm")[value_col].apply(
        lambda s: 100.0 * np.mean(s >= 0.98 * best_overall))
    out["Best"] = g.max()
    out["Median Evals"] = df.groupby("algorithm")["evaluations"].median()
    out["Mean Time (s)"] = df.groupby("algorithm")["wall_time"].mean()
    return out.reindex([a for a in ALGORITHM_ORDER if a in out.index])


def save_table(table: pd.DataFrame, name: str, float_fmt: str = "{:.4f}") -> None:
    table.to_csv(TAB / f"{name}.csv")
    lines = ["| Algorithm | " + " | ".join(table.columns) + " |",
             "|" + "---|" * (len(table.columns) + 1)]
    for algo, row in table.iterrows():
        cells = []
        for col in table.columns:
            v = row[col]
            if isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append(f"| {algo} | " + " | ".join(cells) + " |")
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    # ---------------- Table 2: knapsack ----------------
    kp = pd.read_pickle(RAW / "knapsack_runs.pkl")
    t2 = score_table(kp, "final_fitness")
    save_table(t2, "table2_knapsack", float_fmt="{:.2f}")

    # Friedman + Nemenyi on per-run final fitnesses.
    algos = [a for a in ALGORITHM_ORDER if a in set(kp.algorithm)]
    mat = (kp.pivot_table(index="run", columns="algorithm", values="final_fitness")[algos].to_numpy())
    st = full_analysis(mat, algos)
    (TAB / "table2_stats.txt").write_text(
        f"Friedman: chi2 = {st['friedman']['chi2']:.2f}, p = {st['friedman']['p_value']:.3e}\n"
        f"Nemenyi CD (alpha=0.05, k={st['friedman']['k']}, N={st['friedman']['n']}): {st['cd']:.3f}\n\n"
        f"Mean ranks (1 = best):\n" +
        "".join(f"  {a:15s} {r:.2f}\n" for a, r in sorted(st["ranks"].items(), key=lambda kv: kv[1])) +
        "\nStatistically indistinguishable cliques:\n" +
        "".join(f"  {{{', '.join(cl)}}}\n" for cl in st["cliques"]))
    print("Table 2 (knapsack):")
    print(t2.to_string(float_format=lambda v: f"{v:.2f}"))
    print((TAB / "table2_stats.txt").read_text())

    # ---------------- Table 3: feature selection ----------------
    fs = pd.read_pickle(RAW / "feature_selection_runs.pkl")
    g = fs.groupby("algorithm")
    t3 = pd.DataFrame({
        "Mean Accuracy": g["accuracy"].mean(),
        "Std Accuracy": g["accuracy"].std(ddof=1),
        "Mean Features": g["n_features"].mean(),
        "Std Features": g["n_features"].std(ddof=1),
        "Mean Fitness": g["final_fitness"].mean(),
        "Mean Time (s)": g["wall_time"].mean(),
    }).reindex([a for a in ALGORITHM_ORDER if a in g.groups])
    save_table(t3, "table3_feature_selection", float_fmt="{:.4f}")

    fmat = (fs.pivot_table(index="run", columns="algorithm", values="accuracy")[algos].to_numpy())
    st3 = full_analysis(fmat, algos)
    (TAB / "table3_stats.txt").write_text(
        f"Friedman (accuracy): chi2 = {st3['friedman']['chi2']:.2f}, p = {st3['friedman']['p_value']:.3e}\n"
        f"Nemenyi CD: {st3['cd']:.3f}\n\n"
        f"Mean ranks:\n" +
        "".join(f"  {a:15s} {r:.2f}\n" for a, r in sorted(st3["ranks"].items(), key=lambda kv: kv[1])) +
        "\nCliques:\n" + "".join(f"  {{{', '.join(cl)}}}\n" for cl in st3["cliques"]))
    print("\nTable 3 (feature selection):")
    print(t3.to_string(float_format=lambda v: f"{v:.4f}"))
    print((TAB / "table3_stats.txt").read_text())


if __name__ == "__main__":
    main()
