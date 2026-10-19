"""Non-parametric statistical validation protocol (Section 4.4).

* Friedman test  -- scipy's friedmanchisquare over the per-run final scores
  of all algorithms (runs = "blocks", algorithms = "treatments").
* Nemenyi post-hoc -- all pairwise comparisons with the critical difference

      CD = q_alpha(k) * sqrt(k (k + 1) / (6 N))

  where k = number of algorithms, N = number of runs and q_alpha(k) is the
  Studentized-range / sqrt(2) critical value tabulated by Demšar (2006).
* Critical-Difference cliques -- maximal groups of algorithms whose average
  ranks differ by less than the CD.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Demšar (2006) Table 1: q_alpha / sqrt(2) for alpha = 0.05, k = 2..20
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.425, 17: 3.457, 18: 3.486, 19: 3.513,
    20: 3.539,
}


def mean_ranks(scores: np.ndarray) -> np.ndarray:
    """Average Friedman ranks (rank 1 = best) for a (runs x algorithms) matrix."""
    # scipy.stats.rankdata ranks ascending; maximisation -> rank of -scores.
    from scipy.stats import rankdata
    return rankdata(-scores, axis=1).mean(axis=0)


def friedman_test(scores: np.ndarray) -> dict:
    """scores: (n_runs, n_algorithms) final fitnesses (maximisation)."""
    columns = [scores[:, j] for j in range(scores.shape[1])]
    chi2, p = stats.friedmanchisquare(*columns)
    return {"chi2": float(chi2), "p_value": float(p),
            "k": scores.shape[1], "n": scores.shape[0]}


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    if k not in NEMENYI_Q05:
        raise ValueError(f"No Nemenyi critical value for k={k}")
    q = NEMENYI_Q05[k]
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_cliques(ranks: np.ndarray, cd: float) -> list[list[int]]:
    """Maximal cliques of the 'indistinguishable' graph (rank distance < CD).

    Returns cliques as lists of algorithm indices, sorted best-first; the
    graph is an interval graph on the rank line, so cliques are found with a
    simple greedy sweep.
    """
    order = np.argsort(ranks)
    cliques: list[list[int]] = []
    i = 0
    k = len(ranks)
    while i < k:
        clique = [order[i]]
        j = i + 1
        while j < k and ranks[order[j]] - ranks[order[i]] < cd:
            clique.append(order[j])
            j += 1
        if len(clique) > 1:
            cliques.append(sorted(clique, key=lambda x: ranks[x]))
        i += 1
    return cliques


def full_analysis(scores: np.ndarray, labels: list[str],
                  alpha: float = 0.05) -> dict:
    """Convenience wrapper used by the experiment scripts."""
    scores = np.asarray(scores)
    fr = friedman_test(scores)
    ranks = mean_ranks(scores)
    cd = nemenyi_cd(fr["k"], fr["n"], alpha)
    cliques_idx = nemenyi_cliques(ranks, cd)
    order = np.argsort(ranks)
    return {
        "friedman": fr,
        "ranks": {labels[i]: float(ranks[i]) for i in range(len(labels))},
        "rank_order": [labels[i] for i in order],
        "cd": cd,
        "cliques": [[labels[i] for i in cl] for cl in cliques_idx],
        "significant_pairs": [
            (labels[a], labels[b])
            for a in range(len(labels)) for b in range(a + 1, len(labels))
            if abs(ranks[a] - ranks[b]) >= cd
        ],
    }
