# Module 41: D3, cached push-through with drift-gated refit

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in about 20 seconds.

**Question.** Can the device keep the pushed disease models cached and refit the host map only when the healthy captures drift, without losing accuracy (proposal section 6)?

**Setup (synthetic world a, strength 2, disjoint pools as in modules 17 and 20).**
- The target at time t is the world operator applied to source samples, plus a drift shift growing linearly to a final size (0, 1, 2 or 4 units) in a fixed random direction. 12 time steps; 25 new healthy captures per step; a window of the most recent 100.
- **Warm-up.** No policy runs until the healthy window is full (4 steps), so no policy starts from a small first fit. All policies are evaluated over the remaining 9 steps.
- **Drift statistic:** the Mahalanobis distance of the window mean from the profile mean, plus the relative Frobenius change of the window covariance. The profile is the window at the last refit.
- **Threshold: 1.30**, the 95th percentile of the same statistic on windows of **clean source-crop captures** (no drift). No target data sets it.
- **Policies:** *never* (fit once when the window is full), *gated* (refit when the statistic exceeds the threshold), *always* (refit at every step).
- Accuracy: classifying target disease samples drawn at each step with the same drift. 35 runs per drift level.

## Results (`results/41_d3_drift/drift_results.csv`)

| Final drift | Policy | Refits per run (of 9) | Share served from cache | Mean accuracy | Final-step accuracy |
|---|---|---|---|---|---|
| 0 | never | 1.0 | 0.89 | 0.587 | 0.586 |
| 0 | gated | 1.0 | 0.89 | 0.587 | 0.586 |
| 0 | always | 9.0 | 0.00 | 0.592 | 0.588 |
| 2 | never | 1.0 | 0.89 | 0.573 | 0.549 |
| 2 | gated | 3.3 | 0.64 | 0.589 | 0.590 |
| 2 | always | 9.0 | 0.00 | 0.589 | 0.589 |
| 4 | never | 1.0 | 0.89 | 0.551 | 0.502 |
| 4 | gated | 4.7 | 0.48 | 0.586 | 0.581 |
| 4 | always | 9.0 | 0.00 | 0.587 | 0.582 |

(Drifts of 1.0 are in the CSV: gated 2.4 refits, 0.592 accuracy.)

## Reading

- **No drift: no false triggers.** The gated policy refits once, the same as never-refit, and its accuracy is identical. Always-refit is within 0.005 (0.592 against 0.587), which is a small gain from the larger window.
- **With drift, gated matches always-refit.** At the strongest drift (4 units) its mean accuracy is 0.586 against 0.587, while it refits 4.7 times instead of 9 and serves 48% of steps from the cache. At drift 2 it refits 3.3 times and serves 64% from the cache, with the same accuracy as always-refit.
- **Never-refit loses accuracy as drift grows**: the final-step accuracy falls from 0.586 (no drift) to 0.502 (drift 4).
- This supports the D3 claim: refit only when the drift statistic fires. Most refits are avoided at no accuracy cost.

## An earlier version and its correction

The first version had no warm-up. The "never" baseline was then fitted from only 25 images at the first step, which handicapped it by about 0.12 accuracy even without drift. The version above fixes this. The conclusion about gated refitting is the same, with a more honest baseline.

## Limits

- The drift is a synthetic mean shift in one direction. Real drift (lighting, season, camera) may differ and may also change the covariance.
- 9 evaluated steps and 35 runs, with no intervals.
- The drift statistic and threshold are one choice among several. Only the mean-shift direction is tested.

**How to re-run:**
```bash
python features/41_d3_drift_refit.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/41_d3_drift --scale 2
```
