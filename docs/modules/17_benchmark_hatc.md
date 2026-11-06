# Module 17: first benchmark run on the synthetic worlds (affine branch)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in about 2.5 minutes.

**Question.** On worlds with a known relation between source and target crops, does the affine branch of HATC beat no adaptation and simple baselines?

**Methods compared** (single source crop; one replicate = one crop and one set of samples):
- **NA:** no adaptation. Source class Gaussians applied to the target directly.
- **MC:** mean-centring.
- **HATC (affine):** the affine map fitted from the source healthy sample to the target healthy sample, applied to the source class Gaussians (modules 11 and 13).

**Metric.** Accuracy of classifying the target disease samples among that crop's disease classes. 50 disease samples per class, healthy sample sizes 50 and 200, 117 replicates per cell.

## Correction: an earlier version of this module had a train-test leak

The first version drew the source models, the test samples and (in module 26) the probe's training samples from the **same pool of images**. The models and the probe could therefore see the test images. Symptoms: the probe reached HATC's accuracy with only 5 labels per class (see module 31), and HATC looked better than it should. The first results were discarded.

The fix: every crop-disease pool is split into three **disjoint** parts (50% for source models, 25% for target test samples, 25% for probe training). Groups smaller than 60 images are not used, so fewer replicates run (117 of 200 per cell).

## Operator strength (calibrated, post hoc)

The worlds' operators are scaled by a strength `s`. The strength is chosen on the **no-adaptation score only**, so HATC is not tuned in. World a, healthy n = 200, disjoint pools:

| Strength | No-adaptation accuracy |
|---|---|
| 1 | 0.72 |
| 2 | 0.51 |
| 3 | 0.35 |
| 4 | 0.28 |
| 6 | 0.24 |

Rule: the smallest strength where no adaptation scores between about 0.40 and 0.65. This gives strength **2**. (The earlier, leaky version had given strength 3.) The calibration was done after earlier results had been seen. It is recorded in the registration as post-hoc.

## Results at strength 2 (`results/17_benchmark_hatc/summary.csv`; mean accuracy, healthy n = 50 / 200)

| World | NA | MC | HATC | HATC − NA |
|---|---|---|---|---|
| a (affine) | 0.51 / 0.51 | 0.57 / 0.57 | 0.55 / 0.61 | +0.04 / +0.10 |
| a2 (additive shift) | 0.68 / 0.67 | 0.77 / 0.77 | 0.56 / 0.64 | −0.11 / −0.03 |
| b (radial) | 0.61 / 0.61 | 0.60 / 0.60 | 0.56 / 0.64 | −0.05 / +0.03 |
| b2 (non-composing) | 0.63 / 0.62 | 0.59 / 0.59 | 0.53 / 0.58 | −0.09 / −0.04 |
| c25 | 0.57 / 0.58 | 0.57 / 0.58 | 0.54 / 0.59 | −0.04 / +0.01 |
| c50 | 0.55 / 0.55 | 0.53 / 0.54 | 0.52 / 0.55 | −0.03 / +0.00 |
| c100 | 0.49 / 0.49 | 0.47 / 0.47 | 0.47 / 0.50 | −0.01 / +0.02 |
| d, f (shifts) | as a2 | as a2 | as a2 | −0.11 to −0.14 / −0.03 to −0.06 |
| e (rotation, shear) | 0.23 / 0.23 | 0.23 / 0.23 | 0.33 / 0.34 | +0.10 / +0.11 |

## Reading

- **HATC does not beat no adaptation in most cells at 50 healthy images.** With 50 images, the fitted map is noisy, and the noisy map costs more than it gains. It beats no adaptation at 200 images in the affine world (a), the radial world (b), the graded worlds (c) and world (e), by between 0.00 and 0.11.
- **In the additive-shift worlds (a2, d, f), mean-centring is far better.** HATC is below no adaptation there. This is expected: the shift is exactly what mean-centring corrects.
- **In the non-composing world (b2), HATC is below no adaptation at both sizes.**
- The gain from 50 to 200 healthy images is real and consistent, which supports the registered threshold of 200 for the affine map.
- The honest summary for the paper: HATC's affine branch helps when the relation is close to affine and at least 200 healthy images are available. It is not a general improvement.

## Limits

- Single source only. The agreement check and the common-mode world (d) need two source crops (module 18). World d is currently identical to a2.
- No confidence intervals yet (module 36). 117 of 200 replicates per cell.
- The worlds' operators are first-draft, and the strength was calibrated post hoc.

**How to re-run:**
```bash
python features/17_benchmark_hatc.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/17_benchmark_hatc --scale 2
```
