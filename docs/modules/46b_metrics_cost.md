# Modules 23, 27, 46b, 46c: extra baselines, per-class metrics, calibration and cost

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU, 31 seconds locally (`features/46b_metrics_cost.py`).

**Question.** (23) How do unlabelled-field adaptations compare with the healthy-only methods? (27) How does a pooled-source classifier with a few labelled target images compare? (46b) What do per-class recall, worst-class recall, Brier score and calibration show beyond accuracy? (46c) What does each method cost?

**Method.** Same worlds and pools as modules 17, 20, 43 (healthy n = 200, operator strength 2, 16 dimensions, 34 replicates per world, worlds a, c50, e, f).
- **ADABN-style** and **SHOT-style** are simplified reimplementations on frozen features, not the published methods (TENT and the original SHOT adapt network layers). They use 150 unlabelled target disease images per replicate. ADABN standardises them with their own mean and spread; SHOT starts from the mean-shift models and moves class means towards pseudo-labelled centroids for 3 rounds.
- **POOL5 and POOL25**: unaligned source class samples plus 5 or 25 labelled target images per class.
- Calibration uses a softmax over the class log-likelihoods with equal priors; ECE has 10 bins.

## Results (`results/46b_metrics_cost/n200/metrics.csv`; accuracy, worst-class recall, ECE)

| World | NA | MC | STD | CORAL | HATC | S50 | ADABN | SHOT | POOL5 | POOL25 |
|---|---|---|---|---|---|---|---|---|---|---|
| a, accuracy | 0.52 | 0.57 | 0.59 | 0.62 | 0.63 | 0.66 | 0.57 | 0.64 | 0.67 | 0.72 |
| a, worst-class recall | 0.17 | 0.19 | 0.23 | 0.23 | 0.26 | 0.30 | 0.24 | 0.31 | 0.33 | 0.46 |
| a, ECE | 0.40 | 0.33 | 0.31 | 0.27 | 0.27 | 0.24 | 0.32 | 0.28 | 0.24 | 0.17 |
| c50, accuracy | 0.55 | 0.54 | 0.62 | 0.60 | 0.59 | 0.62 | 0.56 | 0.59 | 0.66 | 0.72 |
| e, accuracy | 0.23 | 0.23 | 0.29 | 0.33 | 0.33 | 0.28 | 0.32 | 0.22 | 0.43 | 0.55 |
| f, accuracy | 0.71 | 0.77 | 0.76 | 0.66 | 0.67 | 0.76 | 0.59 | 0.73 | 0.75 | 0.76 |

## Reading

- **The worst class is far below the mean.** Even where mean accuracy is 0.62 to 0.66 (world a), the worst-class recall of HATC is 0.26 and of S50 is 0.30. In world e the worst class is near 0 for every healthy-only method. Mean accuracy hides this.
- **All methods are badly over-confident.** ECE is 0.12 to 0.76 for the healthy-only methods; the likelihood classifier with equal priors is not calibrated. Better accuracy brings better calibration, but none is calibrated. Reliability diagrams are not drawn.
- **The unlabelled-field adaptations do not beat the healthy-only maps** in most worlds: SHOT-style is similar to HATC in worlds a and f and worse in c50 and e (it can lock in wrong pseudo-labels); ADABN-style is worse than mean-centring in a and f. This supports the checklist claim that these baselines do not dominate, but these are simplified versions, so the comparison with the published methods is not made.
- **A few labels beat everything.** POOL5 matches or beats HATC in all four worlds, and POOL25 beats it by 5 to 21 points, consistent with module 31.
- **Cost.** All methods take 11 to 14 ms per replicate on one thread (fit plus classify of 50 images per class); SHOT-style takes about 27 ms because of its three pseudo-labelling rounds. HATC costs about 2 ms more than mean-centring, for the matrix square roots. Cost is not a reason to prefer or reject any method here. Peak memory and latency per image are not measured.

## Limits

Four worlds, 34 replicates, no intervals; simplified baselines; calibration by likelihood softmax only; timings include Python overhead and were taken in one process on a busy machine.

**How to re-run:**
```bash
python features/46b_metrics_cost.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/46b_metrics_cost --tag n200 --healthy-n 200 --worlds a c50 e f --reps 58
```
