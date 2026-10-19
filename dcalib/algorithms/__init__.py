"""Algorithm registry -- all ten optimizers with the manuscript's Table A2
hyper-parameters baked in as defaults."""

from __future__ import annotations

from .base import BinaryOptimizer, RunResult, sigmoid, binarize
from .dca import DCA
from .ga import GA
from .de_bpso import DE, BPSO
from .abc import ABC
from .gwo_hho import GWO, HHO
from .woa_mfo import WOA, MFO
from .hybrid_ga_dca import HybridGADCA

#: Display order used in every table / figure of the study.
ALGORITHM_ORDER = [
    "DCA",
    "Standard DE",
    "Standard GA",
    "GWO",
    "Binary PSO",
    "ABC",
    "HHO",
    "WOA",
    "Hybrid GA-DCA",
    "MFO",
]


def build_algorithms() -> dict[str, BinaryOptimizer]:
    """Instantiate all ten algorithms with Table A2 settings."""
    return {
        "DCA": DCA(p_prune=0.1, k_survey=4, p_trade=0.2),
        "Standard GA": GA(p_crossover=0.8, p_mutation=0.03, tournament_size=3),
        "Standard DE": DE(f=0.5, cr=0.7),
        "Binary PSO": BPSO(w=0.5, c1=1.5, c2=1.5),
        "ABC": ABC(limit=20),
        "GWO": GWO(),
        "HHO": HHO(),
        "WOA": WOA(b=1.0),
        "MFO": MFO(b=1.0),
        "Hybrid GA-DCA": HybridGADCA(p_crossover=0.8, tournament_size=3,
                                     local_search_steps=5, k_survey=4),
    }


__all__ = [
    "BinaryOptimizer", "RunResult", "sigmoid", "binarize",
    "DCA", "GA", "DE", "BPSO", "ABC", "GWO", "HHO", "WOA", "MFO",
    "HybridGADCA", "ALGORITHM_ORDER", "build_algorithms",
]
