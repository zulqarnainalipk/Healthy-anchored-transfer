# Module 37: equivalence outcomes (primary contrast)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in under a second.

**Question.** Is HATC better than, equivalent to, or worse than the best simple baseline on the synthetic worlds? Three exclusive outcomes (proposal 7.6): **equivalent**, **different**, **inconclusive**. An inconclusive result is never reported as a tie.

**Rule.** The equivalence margin is ±2 accuracy points (registered). Alpha is 0.05, read from the 90% cluster-bootstrap interval of module 36 (two one-sided tests):
- **equivalent:** the 90% interval lies inside (−0.02, +0.02).
- **different:** the interval lies entirely above +0.02 (HATC better) or below −0.02 (HATC worse).
- **inconclusive:** anything else.

**Primary contrast.** HATC minus the best simple baseline, in each world and healthy-image cell. All other contrasts (against NA, MC, STD and CORAL) are **exploratory**; the proposal allows Holm correction or exploratory labelling (7.7), and exploratory labelling is used.

**Status.** Every cell has only 7 clusters, below the registered minimum of 8 (module 36). All outcomes below are therefore **exploratory** by the registered rule.

## Results (`results/37_equivalence/primary_outcomes.csv`; 20 cells)

| Outcome | Cells |
|---|---|
| different (HATC worse) | 12 |
| inconclusive | 8 |
| different (HATC better) | 0 |
| equivalent | 0 |

By world (healthy n = 50 / 200; difference from the best simple baseline in accuracy points):

| World | Best simple | Difference | Outcome |
|---|---|---|---|
| a (affine) | STD | −2.5 / +2.2 | inconclusive / inconclusive |
| a2, d, f (shifts) | MC | −20.4 / −12.5 | worse / worse |
| b (radial) | STD | −17.1 / −10.6 | worse / worse |
| b2 (non-composing) | NA | −9.3 / −4.4 | worse / inconclusive |
| c25, c50, c100 | STD | −7 to −5 / −4 to −3 | worse / inconclusive |
| e (rotation, shear) | STD | +3.9 / +4.8 | inconclusive / inconclusive |

**HATC against CORAL** (exploratory): equivalent in **18 of 20 cells**, inconclusive in 2. HATC and CORAL are practically the same method in this benchmark.

## Reading

- **HATC does not beat the best simple baseline in any world.** It is worse in 12 of 20 cells and inconclusive in the rest. In world e, the point estimate favours HATC by 4 to 5 points, but the interval includes zero and the margin.
- **At 200 healthy images HATC is closer to the best simple baseline** (most cells inconclusive), and at 50 images it is clearly worse in most worlds.
- The equivalence with CORAL means HATC adds nothing over a standard second-order alignment here. The proposal already reports a tie, an inconclusive result or a loss as a reportable result.
- With only 7 clusters, these outcomes are an indication, not a registered inference.

**How to re-run:**
```bash
python features/37_equivalence.py --intervals results/36_cluster_bootstrap/intervals.csv --output results/37_equivalence
```
