# Implementation Notes for the Ten Algorithms

All algorithms operate on binary strings (maximisation) and share the same
benchmark protocol (Appendix Table A1 of the manuscript: identical N,
MaxIter, runs and seeding per problem). All optimizers record the number of
true fitness evaluations (the evaluation cache in `dcalib/problems/base.py`
deduplicates repeated candidate vectors).

Continuous-domain algorithms are adapted with the **standard sigmoid
transfer function** described in Section 4.2 of the manuscript:
`P(bit = 1) = sigmoid(z)`, sampled stochastically each iteration.

| Algorithm | Continuous core | Key parameters (Table A2) | Binary adaptation |
|---|---|---|---|
| **DCA** | — (natively binary) | p_prune=0.1, k_survey=4, p_trade=0.2 | native; trading quality metric = value/weight (knapsack) or \|corr(feature, y)\| (FS) |
| GA | — (natively binary) | one-point crossover pc=0.8, per-bit mutation pm=0.03, tournament=3 | native; 1 elite retained |
| DE | DE/rand/1/bin | F=0.5, CR=0.7 | sigmoid binarization; continuous bounds (-20, 20), init U(-5, 5)* |
| BPSO | Kennedy–Eberhart | w=0.5 (constant), c1=c2=1.5, v clamp ±6 | velocity → sigmoid → Bernoulli bits |
| ABC | classic ABC | limit=20 | sigmoid binarization, fitness-shifted roulette onlookers |
| GWO | alpha/beta/delta | a: 2→0 linear | sigmoid binarization |
| HHO | Heidari et al. | E = 2·E0·(1−t/T) | sigmoid binarization |
| WOA | Mirjalili | a: 2→0, b=1 | sigmoid binarization |
| MFO | moth flame | flames N→1, b=1 | sigmoid binarization |
| Hybrid GA-DCA | GA + DCA local search | pc=0.8, tournament=3, 5 DCA steps after crossover | native; **no mutation** (per Table A2) → the documented diversity collapse |

\* The manuscript specifies only "a standard sigmoid transfer function";
the continuous bounds and init range are implementation choices we
calibrated (see the verification report). Other sigmoid-adapted algorithms
use bounds (-5, 5).

## DCA details (Algorithm 1, followed line by line)

* **Greedy Decoration** — survey `k_survey` random *available* components
  (bits = 0); adopt every surveyed component that gives an immediate fitness
  improvement (the pseudocode loop continues over the survey set).
* **Probabilistic Pruning** — with probability `p_prune`, remove one
  uniformly random *selected* component and re-evaluate (unconditional, per
  pseudocode lines 24–31).
* **Targeted Trading** — with probability `p_trade` per trial (N trials per
  generation), pick a random distinct pair; the fitter crab identifies its
  best *gettable* item (present in the weaker crab, absent from itself,
  best by the problem-specific quality metric) and its worst *giveable*
  item (present in itself, worst by the same metric); the swap is executed
  only if it improves the fitter crab.
* **Elitism** — GlobalBest is tracked and never degrades.

## Fairness notes

* All algorithms run with identical N / MaxIter / runs / seeds per problem
  (Table A1). Per-iteration evaluation counts differ *by design of each
  algorithm* and are reported in the results tables (`Median Evals`).
* The evaluation cache never leaks across runs; every run is seeded with
  `base_seed + run_id` (base 20260101), and the FS cross-validation folds
  are fixed once for the entire experiment.
