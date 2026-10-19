"""Generate Figures 2-6 of the manuscript from the raw benchmark results.

Fig 2 -- Nemenyi Critical Difference visualisation (ranked bar chart, paper
         style, plus a classic Demšar diagram as a supplementary panel).
Fig 3 -- Average convergence curves (mean best-so-far fitness per iteration).
Fig 4 -- Violin plots of final fitness (solution reliability).
Fig 5 -- Population diversity dynamics (exploration vs exploitation).
Fig 6 -- DCA hyperparameter sensitivity (p_trade left, k_survey right).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcalib.algorithms import ALGORITHM_ORDER
from dcalib.stats import full_analysis

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Visual identity: DCA highlighted, competitors muted, variant dashed.
STYLE = {
    "DCA":           dict(color="#C8102E", lw=2.8, zorder=10),
    "Standard DE":   dict(color="#1F77B4", lw=1.6),
    "Standard GA":   dict(color="#2CA02C", lw=1.6),
    "GWO":           dict(color="#9467BD", lw=1.4),
    "Binary PSO":    dict(color="#FF7F0E", lw=1.4),
    "ABC":           dict(color="#8C564B", lw=1.4),
    "HHO":           dict(color="#17BECF", lw=1.4),
    "WOA":           dict(color="#7F7F7F", lw=1.4),
    "Hybrid GA-DCA": dict(color="#D62728", lw=1.6, ls="--"),
    "MFO":           dict(color="#BCBD22", lw=1.4),
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def load_curves(problem: str):
    z = np.load(RAW / f"{problem}_curves.npz")
    curves = {a: z[f"curve__{a}"] for a in ALGORITHM_ORDER}
    divers = {a: z[f"div__{a}"] for a in ALGORITHM_ORDER}
    return curves, divers


def fig2_critical_difference(algos, scores, out_prefix: str, title: str):
    st = full_analysis(scores, algos)
    ranks = st["ranks"]
    order = sorted(ranks, key=ranks.get)
    cd = st["cd"]

    # Clique coloring (interval-graph cliques from the stats module)
    clique_of = {}
    for ci, cl in enumerate(st["cliques"]):
        for a in cl:
            clique_of[a] = ci
    palette = plt.cm.tab10.colors

    # ---- Panel A: ranked bar chart (manuscript Figure 2 style) ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    vals = [ranks[a] for a in order]
    colors = [palette[clique_of.get(a, -1) % 10] if a in clique_of else "#B0B0B0"
              for a in order]
    ax.barh(range(len(order)), vals, color=colors, edgecolor="black", lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Average Friedman rank (1 = best)")
    ax.set_title(f"{title}\nNemenyi CD = {cd:.2f}  "
                 f"(Friedman p = {st['friedman']['p_value']:.1e})", fontsize=11)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) + 1.2)
    fig.savefig(FIG / f"{out_prefix}.png")
    plt.close(fig)

    # ---- Panel B: classic Demsar CD diagram ----
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    y = np.linspace(1, 0, len(order))
    ax.axvline(0.5, color="k", lw=1.2)
    for cl in st["cliques"]:
        lo = min(ranks[a] for a in cl)
        hi = max(ranks[a] for a in cl)
        c = palette[clique_of[cl[0]] % 10]
        idx = [order.index(a) for a in cl]
        ax.plot([lo, hi], [y[idx].mean()] * 2, color=c, lw=2.5,
                solid_capstyle="round")
    for i, a in enumerate(order):
        ax.plot(ranks[a], y[i], "o", ms=9, color=palette[clique_of.get(a, -1) % 10]
                if a in clique_of else "#B0B0B0", mec="black", mew=0.7, zorder=5)
        ax.text(ranks[a], y[i] + 0.055, a, ha="center", fontsize=9)
    ax.set_title(f"{title} -- Critical Difference (CD = {cd:.2f})", fontsize=11)
    ax.set_yticks([])
    ax.set_xlabel("Average rank")
    ax.grid(False)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(FIG / f"{out_prefix}_demsar.png")
    plt.close(fig)
    return st


def fig3_convergence(curves, title):
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    iters = np.arange(1, curves["DCA"].shape[1] + 1)
    for name in ALGORITHM_ORDER:
        mean = curves[name].mean(axis=0)
        style = dict(STYLE[name])
        ax.plot(iters, mean, label=name, **style)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best-so-far fitness (mean of 50 runs)")
    ax.set_title(title, fontsize=12)
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    fig.savefig(FIG / "fig3_convergence.png")
    plt.close(fig)


def fig4_violin(df, title):
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    data = [df.loc[df.algorithm == a, "final_fitness"].to_numpy() for a in ALGORITHM_ORDER]
    vp = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.85)
    for i, body in enumerate(vp["bodies"]):
        name = ALGORITHM_ORDER[i]
        body.set_facecolor(STYLE[name]["color"])
        body.set_alpha(0.65 if name == "DCA" else 0.45)
    for i, d in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.05, len(d)), d,
                   s=6, color="black", alpha=0.35, zorder=3)
    ax.set_xticks(range(1, len(ALGORITHM_ORDER) + 1))
    ax.set_xticklabels(ALGORITHM_ORDER, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Final fitness")
    ax.set_title(title, fontsize=12)
    fig.savefig(FIG / "fig4_violin.png")
    plt.close(fig)


def fig5_diversity(divers, title):
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    iters = np.arange(1, divers["DCA"].shape[1] + 1)
    for name in ALGORITHM_ORDER:
        mean = divers[name].mean(axis=0)
        ax.plot(iters, mean, label=name, **dict(STYLE[name]))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Population diversity (norm. mean Hamming distance)")
    ax.set_title(title, fontsize=12)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(FIG / "fig5_diversity.png")
    plt.close(fig)


def fig6_sensitivity(title):
    df = pd.read_csv(RAW / "sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    for ax, param, xlabel in [
        (axes[0], "p_trade", "Trading probability (p_trade)"),
        (axes[1], "k_survey", "Survey size (k_survey)"),
    ]:
        sub = df[df.parameter == param]
        ax.errorbar(sub["value"], sub["mean_fitness"], yerr=sub["std_fitness"],
                    marker="o", ms=5, color="#C8102E", lw=2, capsize=3,
                    ecolor="#888888")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean final fitness")
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.savefig(FIG / "fig6_sensitivity.png")
    plt.close(fig)


def main() -> None:
    # ------------------ Knapsack figures ------------------
    kp = pd.read_pickle(RAW / "knapsack_runs.pkl")
    curves, divers = load_curves("knapsack")
    algos = [a for a in ALGORITHM_ORDER if a in set(kp.algorithm)]
    scores = (kp.pivot_table(index="run", columns="algorithm",
                             values="final_fitness")[algos].to_numpy())

    st2 = fig2_critical_difference(algos, scores, "fig2_cd_knapsack",
                                   "Knapsack-50: Critical Difference diagram")
    print("fig2 done. rank order:", st2["rank_order"])
    fig3_convergence(curves, "Knapsack-50: average convergence curves (50 runs)")
    print("fig3 done.")
    fig4_violin(kp, "Knapsack-50: distribution of final fitness (50 runs)")
    print("fig4 done.")
    fig5_diversity(divers, "Knapsack-50: population diversity dynamics")
    print("fig5 done.")

    # ------------------ Feature-selection CD (supplementary) ------------------
    try:
        fs = pd.read_pickle(RAW / "feature_selection_runs.pkl")
        fs_algos = [a for a in ALGORITHM_ORDER if a in set(fs.algorithm)]
        fs_scores = (fs.pivot_table(index="run", columns="algorithm",
                                    values="accuracy")[fs_algos].to_numpy())
        fig2_critical_difference(fs_algos, fs_scores, "fig2b_cd_feature_selection",
                                 "UCI Feature Selection: Critical Difference (accuracy)")
        print("fig2b done.")
    except FileNotFoundError:
        print("feature selection results not found -- skipping fig2b")

    # ------------------ Sensitivity ------------------
    if (RAW / "sensitivity.csv").exists():
        fig6_sensitivity("DCA hyperparameter sensitivity (knapsack, 30 runs/setting)")
        print("fig6 done.")


if __name__ == "__main__":
    main()
