# Module 18: first test of the agreement check (two sources, synthetic worlds)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in about 19 minutes.

**Question.** Does the disagreement between two sources' predictions for the same disease on the target predict transfer failure? This is the third research question (proposal section 2).

## Correction: the first version had a train-test leak

The first version of this module used source models and test samples drawn from the same pool of images, and an operator strength of 3. Both were corrected: pools are now disjoint (see `17_benchmark_hatc.md`), and the strength is 2. The first-version results (a bound near 0.53 everywhere, failure rates of 0 to 9% at 200 healthy images) were discarded. The numbers below are from the corrected run.

## Simplified two-source setup (a limitation)

Source 1 is a real source crop's class samples. Source 2 is the same classes seen through a fixed mild random affine "look" V, standing in for a second crop with the same diseases. Real data rarely has two crops with the same disease, so this is a synthetic stand-in. Each source is pushed to the target through its own affine map, fitted from its own healthy sample (module 11 and 13 logic).

## Measure

For each class, a logistic-regression discriminator separates the two sources' pushed samples (100 each). Its held-out accuracy gives a lower confidence bound on total variation: TV ≥ 2a − 1 (Wilson lower bound on a, 95%). The replicate's disagreement is the mean bound over classes. A replicate **fails** when the mean HATC accuracy of the two sources is below the no-adaptation accuracy by more than 0.02. Failure is now common, because HATC is often below no adaptation (module 17).

## Results (`results/18_agreement/agreement_summary.csv`; 117 replicates per cell)

| World | Healthy n | Failure rate | Mean bound | AUROC bound vs failure |
|---|---|---|---|---|
| a (affine) | 50 / 200 | 0.25 / 0.14 | 0.36 / 0.37 | 0.54 / 0.51 |
| a2 (shift) | 50 / 200 | 0.80 / 0.48 | 0.35 / 0.36 | 0.67 / 0.50 |
| b (radial) | 50 / 200 | 0.62 / 0.28 | 0.36 / 0.37 | 0.58 / 0.64 |
| b2 (non-composing) | 50 / 200 | 0.72 / 0.49 | 0.36 / 0.37 | 0.44 / 0.60 |
| c25 | 50 / 200 | 0.50 / 0.26 | 0.36 / 0.37 | 0.40 / 0.62 |
| c50 | 50 / 200 | 0.45 / 0.35 | 0.36 / 0.37 | 0.63 / 0.71 |
| c100 | 50 / 200 | 0.43 / 0.31 | 0.36 / 0.37 | 0.67 / 0.69 |
| d | 50 / 200 | 0.80 / 0.48 | 0.35 / 0.36 | 0.67 / 0.50 |
| e (rotation, shear) | 50 / 200 | 0.16 / 0.15 | 0.49 / 0.50 | **0.87 / 0.86** |
| f (known shift) | 50 / 200 | 0.90 / 0.60 | 0.35 / 0.36 | 0.64 / 0.50 |

## Reading

- **The check is informative in one world only.** In world e, the relation is a rotation plus shear and is not a transport map. The bound is clearly higher there (0.49 to 0.50 against about 0.36 elsewhere), and the AUROC is 0.86 to 0.87. The check therefore detects a relation that is not a transport map.
- **In every other world, the AUROC is close to chance** (0.40 to 0.71). The check does not predict whether HATC will fail to beat no adaptation. The bound is nearly constant (0.35 to 0.37) across those worlds.
- **The proposal's claim for the agreement check is conditional.** It is claimed only if it beats the label-free reliability scores (module 30). This evidence does not support a general claim. It supports at most the narrower claim that the check detects non-transport relations.

## Why the bound is similar in most worlds

The affine Monge map between two Gaussians is the one symmetric positive-definite map. The world operators (and the source-2 look V) are not symmetric positive-definite, so a map fitted from each source's healthy distribution differs from the true operator in a different way for each source. The two sources then disagree somewhat even when the true relation is affine. The disagreement is similar across worlds because it is dominated by this estimation difference, not by the world.

## Limits

- The two-source setup is synthetic and simplified.
- The failure label depends on the no-adaptation baseline and a margin of 0.02. A different definition (for example, failure against the best simple baseline, as proposal 7.7 specifies for the real-pair analysis) would give different rates. This is not yet done.
- No confidence intervals on the AUROC (module 36).
- World d is identical to a2 in this single-direction setup. A real common-mode test needs the two sources wrong in the same way, which needs a different construction.

**How to re-run:**
```bash
python features/18_agreement_benchmark.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/18_agreement --scale 2
```
