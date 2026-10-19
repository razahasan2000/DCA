"""dcalib -- reference implementation of the Decorator Crab Algorithm (DCA)
and the full benchmarking suite used in the manuscript.

Modules
-------
problems    : knapsack + wrapper-based feature selection benchmarks
algorithms  : DCA and nine competitor metaheuristics
stats       : Friedman / Nemenyi protocol with Critical Difference diagrams
runner      : parallel benchmark execution with full provenance
"""

__version__ = "1.0.0"

from .problems import make_problem, KnapsackProblem, FeatureSelectionProblem
from .algorithms import build_algorithms, ALGORITHM_ORDER

__all__ = ["make_problem", "KnapsackProblem", "FeatureSelectionProblem",
           "build_algorithms", "ALGORITHM_ORDER", "__version__"]
