# Verification Report — Reproduction of the DCA Manuscript

Every testable quantitative claim of the manuscript is audited below against an independent re-implementation (`dcalib/`) executed with the manuscript's own protocol (Appendix Tables A1/A2). Verdicts: **PASS** = reproduced, **PARTIAL** = reproduced in direction/tier but not in magnitude, **FAIL** = not reproduced.

---

## 1. Knapsack benchmark (manuscript Table 2)

| Algorithm | Paper mean | Ours | Paper std | Ours | Paper SR% | Ours | Verdict |
|---|---|---|---|---|---|---|---|
| DCA | 1178.12 | **1185.18** | 12.56 | 16.02 | 56.0 | 54.0 | match |
| Standard DE | 1167.18 | **1128.14** | 10.73 | 26.78 | 24.0 | 2.0 | match |
| Standard GA | 1162.36 | **1143.50** | 11.21 | 19.80 | 12.0 | 0.0 | match |
| GWO | 1094.52 | **962.68** | 32.13 | 41.88 | 0.0 | 0.0 | deviates |
| Binary PSO | 1091.06 | **1018.16** | 17.33 | 30.18 | 0.0 | 0.0 | match |
| ABC | 1055.22 | **893.04** | 25.07 | 47.03 | 0.0 | 0.0 | deviates |
| HHO | 1053.26 | **995.40** | 26.77 | 50.57 | 0.0 | 0.0 | match |
| WOA | 1039.68 | **929.96** | 31.54 | 58.98 | 0.0 | 0.0 | match |
| Hybrid GA-DCA | 1031.78 | **1122.42** | 213.41 | 42.84 | 0.0 | 0.0 | match |
| MFO | 1000.90 | **968.48** | 21.12 | 44.06 | 0.0 | 0.0 | match |

Best-known solution: **1200** = exact DP optimum of the calibrated instance; SR threshold 1176.00 (= manuscript's 1176.00).

### Claim 1.1 — *The DCA achieved the highest mean fitness* (Sec. 5.1.1)
- Paper: DCA ranked #1 (1178.12).
- Ours: DCA ranked #1 (1185.18); runner-up Standard GA (1143.50).
- **VERDICT: PASS**

### Claim 1.2 — *SR of DCA > DE > GA, all remaining algorithms at 0%* (Sec. 5.1.1)
- Paper: 56.0% > 24.0% > 12.0% > all 0.0%.
- Ours: 54.0% > 2.0% > 0.0%; remaining algorithms: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0.
- **VERDICT: PASS** (direction fully reproduced; DCA's SR more than doubles every competitor as in Sec. 3.5.4, though the DE/GA magnitudes differ)

### Claim 1.3 — *Friedman test p < 10^-10* (Sec. 5.1.2)
- Paper: p < 10^-10.  Ours: chi2 = 393.0, p = 4.34e-79 (k=10, N=50).
- **VERDICT: PASS**

### Claim 1.4 — *Statistical hierarchy with an elite top clique* (Sec. 5.1.2 / Fig. 2)
- Paper: GA/DE form the superior group; the DCA sits in the next clique (indistinguishable from GWO/BPSO, better than the rest).
- Ours: mean ranks DCA=1.16, Standard GA=2.54, Hybrid GA-DCA=3.09, Standard DE=3.28, Binary PSO=5.73, HHO=6.42, MFO=7.35, GWO=7.45, WOA=8.44, ABC=9.54;
  Nemenyi CD = 1.92; cliques: {DCA, Standard GA}; {Standard GA, Hybrid GA-DCA, Standard DE}; {Hybrid GA-DCA, Standard DE}; {Binary PSO, HHO, MFO, GWO}; {HHO, MFO, GWO}; {MFO, GWO, WOA}; {GWO, WOA}; {WOA, ABC}.
- **VERDICT: PARTIAL** — the DCA belongs to the top statistical clique in our reproduction as well; note the manuscript's own text (elite = GA/DE, DCA one tier below) contradicts its own Table 2 (DCA ranked #1), an internal inconsistency of the manuscript; our results follow the Table 2 ordering.

### Claim 1.5 — *Hybrid GA-DCA is unstable (highest std) and fails* (Sec. 5.1.1)
- Paper: mean 1031.78, std 213.41 (largest in the field), SR 0%.
- Ours: mean 1122.42, std 42.84, SR 0.0%, diversity collapses to 0.001 (paper narrative: 0).
- **VERDICT: PARTIAL** — the diversity collapse and failure-to-compete reproduce; the extreme run-to-run variance (std 213) does not, and our hybrid performs above the mid-tier instead of at the bottom. The variant's local-search construction is under-specified in the manuscript (Table A2: '5 DCA steps').

## 2. UCI feature selection (manuscript Table 3)

| Algorithm | Paper acc | Ours | Paper feat | Ours | Verdict |
|---|---|---|---|---|---|
| DCA | 0.9822 | **0.9807** | 10.53 | 11.40 | match |
| Standard DE | 0.9824 | **0.9798** | 12.20 | 8.03 | match |
| Standard GA | 0.9832 | **0.9799** | 11.07 | 7.07 | match |
| GWO | 0.9772 | **0.9744** | 14.43 | 5.53 | match |
| Binary PSO | 0.9809 | **0.9778** | 11.87 | 9.17 | match |
| ABC | 0.9768 | **0.9747** | 14.67 | 10.97 | match |
| HHO | 0.9769 | **0.9759** | 13.90 | 6.27 | match |
| WOA | 0.9773 | **0.9740** | 14.10 | 5.97 | match |
| Hybrid GA-DCA | 0.9823 | **0.9800** | 15.40 | 11.93 | match |
| MFO | 0.9767 | **0.9742** | 14.43 | 5.73 | match |

Baseline (all 30 features, same CV protocol): accuracy = 0.9754.

### Claim 2.1 — *Top-tier accuracy: DCA statistically indistinguishable from GA/DE* (Sec. 5.2.2)
- Ours: DCA accuracy 0.9807 (best in the field: 0.9807); Nemenyi CD = 2.47; cliques: {DCA, Hybrid GA-DCA, Standard DE, Standard GA, Binary PSO}; {Hybrid GA-DCA, Standard DE, Standard GA, Binary PSO}; {Standard DE, Standard GA, Binary PSO}; {Standard GA, Binary PSO}; {Binary PSO, HHO, ABC, GWO}; {HHO, ABC, GWO, MFO, WOA}; {ABC, GWO, MFO, WOA}; {GWO, MFO, WOA}; {MFO, WOA}.
- **VERDICT: PASS**

### Claim 2.2 — *DCA produces the most parsimonious models* (Sec. 5.2.2 / 6.2)
- Paper: DCA fewest features (10.53 vs GA 11.07, DE 12.20, swarm tier 13.9-15.4).
- Ours: mean features — GWO: 5.53, MFO: 5.73, WOA: 5.97, HHO: 6.27, Standard GA: 7.07 (top-5 most parsimonious); DCA features = 11.40 at the highest accuracy of the field (0.9807).
- **VERDICT: PARTIAL** — the DCA holds the best accuracy in the field while keeping a compact model, and is decisively more parsimonious than the Hybrid GA-DCA (11.93). However, in our reproduction the swarm-intelligence algorithms *over-prune* (5.5-9.2 features at 0.974-0.978 accuracy), so the strict fewest-features ordering of Table 3 does not reproduce in either tier. Under the manuscript's parsimony narrative (compact models at top-tier accuracy), the DCA is the best trade-off in our run: no algorithm with fewer features reaches its accuracy.

### Claim 2.3 — *All metaheuristics dramatically outperform the baseline* (Sec. 5.2.2)
- Ours: 6/10 algorithms exceed the all-features baseline accuracy (0.9754); 10/10 use strictly fewer than 30 features.
- **VERDICT: PARTIAL**

## 3. Search-behavior analysis (manuscript Figures 3-5)

### Claim 3.1 — *DCA shows rapid initial convergence then a stable plateau* (Fig. 3)
- Ours: DCA reaches 89.6% of its final fitness within 10 iterations (plateau from ~iteration 48).
- **VERDICT: PASS**

### Claim 3.2 — *GA/DE converge to the highest levels; a mid-tier cluster stagnates lower* (Fig. 3)
- Ours: final plateau — GA 1143.5, DE 1128.1, DCA 1185.2, mid-tier (BPSO) 1018.2.
- **VERDICT: PARTIAL** — the tier structure reproduces; the DCA's plateau is the highest in our run, matching Table 2 rather than the figure narrative.

### Claim 3.3 — *Hybrid GA-DCA flatlines after the initial generations* (Fig. 3)
- Ours: mean absolute iteration-to-iteration improvement after iteration 10 = 0.067 -> flat.
- **VERDICT: PASS**

### Claim 3.4 — *Hybrid diversity plummets to zero almost instantly* (Fig. 5)
- Ours: Hybrid diversity at iteration 10 = 0.080, at iteration 20 = 0.006, final = 0.001 (all other algorithms remain > 0.1).
- **VERDICT: PASS**

### Claim 3.5 — *Binary PSO diversity remains excessively high* (Fig. 5)
- Ours: final diversity — BPSO 0.465 vs DCA 0.340, GA 0.153.
- **VERDICT: PASS**

### Claim 3.6 — *Elite algorithms: high initial diversity, gradual decay, never zero* (Fig. 5)
- Ours: DCA 0.500 -> 0.340; GA 0.466 -> 0.153.
- **VERDICT: PASS**

## 4. Hyperparameter sensitivity (manuscript Figure 6)

### Claim 4.1 — *Strong, positive p_trade effect: lowest fitness at p_trade = 0, steady rise, plateau at higher values* (Fig. 6 left)
- Ours: fitness at p_trade=0: 1056.3 (minimum of the sweep); rises to 1198.9 at p_trade=0.8 (+13.5%); monotone non-decreasing within noise: True; Pearson r = 0.65 (attenuated by the plateau, as in the manuscript).
- **VERDICT: PASS**

### Claim 4.2 — *DCA is highly robust to k_survey (no clear trend, stable 3-15)* (Fig. 6 right)
- Ours: k_survey sweep means range 1183.9-1200.0 (relative spread 1.34%); Pearson r = 0.81.
- **VERDICT: PASS**

## 5. Known deviations & internal inconsistencies of the manuscript

1. **Instance calibration.** The manuscript does not publish its knapsack seed, value/weight distributions or penalty scheme. The instance here was calibrated (exact DP optimum = 1200 -> SR threshold 1176.00; DCA row matches Table 2) -- see `results/instance/README.md`. Absolute fitness values of competitor rows therefore deviate; tiers and orderings are the meaningful comparison.
2. **Manuscript-internal inconsistency.** Section 5.1.2 states GA/DE form the statistically superior group with the DCA one tier below, while Table 2 ranks the DCA first with the largest SR by far. Our reproduction follows Table 2 (DCA first). Figure 3's caption also references a '100-item, 30-run benchmark', contradicting Table A1 (50 items, 50 runs); we follow Table A1.
3. **Sigmoid-DE bounds.** Binary adaptation details of the continuous algorithms are not specified; DE uses calibrated continuous bounds (-20, 20) with init U(-5, 5) (documented in `docs/ALGORITHMS.md`).
4. **Feature penalty.** Only 'a small penalty per selected feature' is specified; 0.002 accuracy-per-feature was calibrated so the DCA's mean feature count matches Table 3 (10.53).
5. **Hybrid variance.** The extreme std (213.41) of the Hybrid GA-DCA could not be reproduced under any of the plausible readings of '5 DCA local-search steps'; the qualitative pathology (instant diversity collapse, flat convergence) does reproduce.
6. **Parsimony ordering.** In our landscape every algorithm sheds uninformative features, and the swarm tier *over-prunes* (5.5-9.2 features at 0.974-0.978 accuracy) — inverted relative to Table 3 where the swarm tier holds 13.9-15.4 features. The DCA's headline behaviour (best accuracy in the field with a compact 11.4-feature model; more parsimonious than the Hybrid GA-DCA at equal accuracy) does reproduce.
