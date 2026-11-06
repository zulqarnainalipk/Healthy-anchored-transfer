# Module 42: D4, sequential agreement check with a refusal state

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in about 10 minutes.

**Question.** Can the device decide, after each batch of healthy captures, whether two sources agree well enough to trust the transfer, and enter a refusal state if it cannot decide within a capture budget (proposal section 6)?

**Setup.**
- After each batch of 25 healthy captures (a window of the most recent 200), the device recomputes the agreement bound between two sources' pushed predictions (module 18 setup). It **passes** when the bound is at or below a threshold tau, and stops at the first pass. If it never passes within 8 batches (200 captures), it enters the **refusal state**.
- Test worlds (operator strength 2, disjoint pools): a (affine), b2 (non-composing), e (rotation and shear), f (additive shift). 35 runs per world.
- **Threshold tau: 0.004.** It was set on a null case with **no target data**: the two sources are the same crop's class samples with no difference between them, so any disagreement is sampling noise. tau is the 95th percentile of the bound in that null.

## Results (`results/42_d4_sequential/sequential_results.csv`)

| World | Runs | Pass rate | Refusal rate | Captures to decision |
|---|---|---|---|---|
| a (affine) | 35 | 0.00 | 1.00 | 200 (budget exhausted) |
| b2 (non-composing) | 35 | 0.00 | 1.00 | 200 |
| e (rotation, shear) | 35 | 0.00 | 1.00 | 200 |
| f (additive shift) | 35 | 0.00 | 1.00 | 200 |

**The check refuses in every world, including the affine world where the transfer works.**

## This is a design flaw in the threshold, not a result about the method

- The threshold was calibrated on a null with **identical sources**. The bound in that null is about 0, so tau is 0.004.
- In the test worlds, source 2 is source 1 seen through a mild fixed look (the simulation of a second crop). That difference alone gives a bound far above 0.004 (about 0.35 to 0.5, module 18). **Any real difference between two sources exceeds a threshold calibrated on identical sources.** The check can therefore never pass.
- The proposal's design says the null hypothesis is "the two sources disagree beyond the registered margin", with a margin to be registered. A margin must be a **tolerated total-variation difference chosen by the team** (for example, the largest disagreement that still leaves the transfer reliable). A sampling-noise null is not that margin.
- Choosing that margin from the benchmark results would be post-hoc. In module 18 and module 30, the bound is only weakly related to failure, so there is no tolerated difference that separates good transfers from bad ones in these worlds.

## Consequence for the D4 claim

- The claims for D4 in proposal section 6 (captures to decision; false pass on injected mismatched crops) are **not supported as designed**. As specified, D4 cannot pass at all, so captures to decision equals the budget and a false pass cannot occur.
- A redesign needs two things: a margin chosen on principle (not from a noise null), and evidence that the bound separates good transfers from bad ones. Module 18 and module 30 give little support for the second.

## Limits

- The two-source setup is the synthetic simplification of module 18.
- Only the sequential structure (windowed recomputation, budget, refusal state) is tested here. The structure works as intended; the threshold does not.

**How to re-run:**
```bash
python features/42_d4_sequential.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/42_d4_sequential --scale 2
```
