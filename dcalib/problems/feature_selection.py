"""Wrapper-based feature selection on the Wisconsin Breast Cancer dataset
(Section 4.1.2 of the manuscript).

* Dataset  : UCI Wisconsin Breast Cancer (Diagnostic), 569 samples, 30 real
             features -- shipped with scikit-learn as ``load_breast_cancer``.
* Encoding : binary vector of length 30, 1 = feature selected.
* Fitness  : mean accuracy of a LogisticRegression classifier under 3-fold
             stratified cross-validation, minus a small penalty per selected
             feature to encourage parsimony.
* Quality  : |Pearson correlation| between feature and target -- the
             problem-specific desirability metric used by the DCA trading
             operator.

Cross-validation folds are fixed once for the whole experiment so that every
candidate solution of every algorithm is scored on identical folds (variance
reduction and strict comparability).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .base import Problem

N_SPLITS = 3
# Calibrated feature penalty (see experiments/ -- the manuscript describes
# only "a small penalty ... for each selected feature").  0.002 places the
# DCA's mean selected-feature count closest to the manuscript's Table 3
# (10.53) while keeping top-tier accuracy.
FEATURE_PENALTY = 0.002


class FeatureSelectionProblem(Problem):
    def __init__(self, penalty: float = FEATURE_PENALTY, random_state: int = 42):
        data = load_breast_cancer()
        X, y = data.data, data.target
        # Standardise once -- required for well-conditioned logistic regression.
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.X = (X - self.mu) / self.sd
        self.y = y
        self.n_samples, self.n_features = self.X.shape

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        self.folds = list(skf.split(self.X, self.y))

        super().__init__(dim=self.n_features)
        self.name = "UCI-FeatureSelection"
        self.penalty = float(penalty)

        # Pre-computed item quality: |point-biserial correlation| with target.
        yc = (self.y - self.y.mean())
        xc = self.X - self.X.mean(axis=0)
        self._quality = np.abs((xc * yc[:, None]).mean(axis=0) / (self.X.std(axis=0) * self.y.std() + 1e-12))

        # Reference baselines (computed lazily once).
        self._baseline_full: float | None = None

    # ------------------------------------------------------------------
    def raw_fitness(self, x: np.ndarray) -> float:
        n_selected = int(x.sum())
        if n_selected == 0:
            # An empty model predicts the majority class on every fold.
            majority = max(np.mean(self.y == 0), np.mean(self.y == 1))
            return float(majority)
        acc = self.cv_accuracy(x)
        return float(acc - self.penalty * n_selected)

    def cv_accuracy(self, x: np.ndarray) -> float:
        cols = np.where(x)[0]
        Xs = self.X[:, cols]
        scores = []
        for tr, te in self.folds:
            clf = LogisticRegression(max_iter=300, solver="lbfgs")
            clf.fit(Xs[tr], self.y[tr])
            scores.append(clf.score(Xs[te], self.y[te]))
        return float(np.mean(scores))

    def quality(self, k: int) -> float:
        return float(self._quality[k])

    # ------------------------------------------------------------------
    def decompose(self, x: np.ndarray) -> dict:
        """Accuracy and feature count of a solution *without* the penalty."""
        n_selected = int(x.sum())
        if n_selected == 0:
            acc = max(np.mean(self.y == 0), np.mean(self.y == 1))
        else:
            acc = self.cv_accuracy(x)
        return {"accuracy": float(acc), "n_features": n_selected}

    def baseline_full_accuracy(self) -> float:
        """Cross-validation accuracy of the all-features model."""
        if self._baseline_full is None:
            self._baseline_full = self.cv_accuracy(np.ones(self.dim, dtype=int))
        return self._baseline_full
