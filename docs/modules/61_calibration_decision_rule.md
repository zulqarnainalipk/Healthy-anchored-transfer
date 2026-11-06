# Module 61: calibration after alignment and a healthy-only decision rule

**Code:** `analysis/61_calibration_decision_rule.py`. **Compute:** CPU, about 7 minutes. Exploratory (registration section 9).

**(a) Calibration.** Tier-2 units (DINOv2, PCA), 5 splits per unit with the splits of module 53. Posterior of disease from the log-likelihood ratio under equal priors; ECE with 10 bins; Platt scaling fitted on 10 labelled target images per class (uses labels; diagnostic). Output `calibration.csv`.

| Method | ECE raw | ECE Platt | Brier raw | BA raw | BA Platt | AUROC raw |
|---|---|---|---|---|---|---|
| NA | 0.383 | 0.147 | 0.378 | 0.648 | 0.710 | 0.765 |
| MC | 0.296 | 0.141 | 0.289 | 0.716 | 0.737 | 0.811 |
| CORAL | 0.190 | 0.129 | 0.189 | 0.790 | 0.792 | 0.867 |
| HATC | 0.194 | 0.130 | 0.192 | 0.787 | 0.790 | 0.865 |

HATC minus MC: 7.1 points raw, 5.2 after Platt scaling; ranking unchanged.

**(b) Decision rule.** For each tier-2 unit, the Bures-Wasserstein distance between the shrunk healthy Gaussians of source and target in the 16-d projection, divided by the square root of the trace of the target covariance (`mean_part`, `cov_part`, `bw_total`). Spearman correlation with the gain of HATC (`decision_rule_correlations.csv`, with the range over leave-one-target-crop-out refits): DINOv2, distance versus gain over NA 0.595 (0.557 to 0.618), over MC 0.475 (0.429 to 0.533); BioCLIP 0.366 (0.338 to 0.405) and 0.153 (0.108 to 0.218). Terciles of the distance (DINOv2): gain over MC +3.1, +8.2, +10.1 points; (BioCLIP): +1.7, +2.6, +2.3. An association across overlapping units, not a validated threshold.

## How to re-run

```
python analysis/61_calibration_decision_rule.py --features-dino results/09_features_plantwild/remote/features_plantwild --features-bio results/10_features_plantwild_bioclip/remote/features_plantwild_bioclip --results results --output results/61_calibration_decision_rule
```