# Real-pair results (PlantWild v1) and their verification

Registered in `docs/registration.md`, section 7 (before any PlantWild feature was analysed); controls and verification checks are section 8 (post hoc). Code: `features/53_real_pair_test.py` (units, methods, primary outcome), `54_real_agreement.py` (agreement check), `analysis/55`, `56`, `57`, `58` (verification). Data audit: `docs/modules/08b_plantwild_audit.md`. Raw results: `results/53_real_pair_test/<tag>/units.csv` (per unit, mean of 20 splits) and `primary.csv`. Every number in the paper's real-pair sections is in this file.

**Setting.** 17,328 PlantWild images after the duplicate rule (distance 3), DINOv2 ViT-B/14 features (pinned revision), 16-dimensional projection fitted without the target crop (leave-one-crop-out PCA; LDA secondary), covariance shrinkage 0.1, map set = min(200, half of the target healthy images), 20 random splits per unit. Task: healthy versus disease D on the target crop; healthy model = target healthy Gaussian; disease model = source disease Gaussian moved by the method. Balanced accuracy (BA). Primary outcome: HATC − best simple method (best by the tier mean among NA, MC, STD; MC for DINOv2 PCA), two-way cluster bootstrap over target and source crops (10,000 resamples, 90% interval, ±2-point margin).

## 1. Primary result (tier 2: same disease type, different species; 80 units, 22 target crops, 22 source crops)

Mean balanced accuracy over the 80 units: NA 0.649, MC 0.716, STD 0.711, CORAL 0.790, **HATC 0.787**, S25 0.747, S50 0.771, S75 0.785, one-class healthy detector (OC) 0.749, probe 5 labels 0.799, probe 10 labels 0.831, probe 25 labels 0.856 (63 units; 16 target and 18 source crops). Mean AUROC: NA 0.764, MC 0.811, STD 0.799, CORAL 0.867, HATC 0.866, OC 0.841, probe 5 0.875.

| Contrast (minus MC) | Mean difference | 90% interval | Outcome |
|---|---|---|---|
| HATC | +0.0711 | 0.0453 to 0.0995 | different (HATC better) |
| CORAL | +0.0739 | 0.0471 to 0.1037 | different |
| S25 / S50 / S75 | +0.0307 / +0.0547 / +0.0694 | 0.0205 to 0.0426 / 0.0367 to 0.0752 / 0.0466 to 0.0947 | different |
| OC (no source disease) | +0.0331 | 0.0015 to 0.0717 | inconclusive |
| Probe 5 / 10 / 25 labels | +0.0831 / +0.1149 / +0.1476 | 0.0463 to 0.1213 / 0.0795 to 0.1530 / 0.1125 to 0.1827 | different |

Paired contrasts (two-way bootstrap, `results/56_primary_checks/pca_seeds/checks.json`): HATC − NA +0.138 (better in 73 of 80 units), HATC − MC +0.071 (67 of 80), HATC − STD +0.077 (76 of 80); HATC − OC +0.038 (interval 0.010 to 0.062; better in 64 of 80); HATC − CORAL −0.003 (−0.007 to +0.001); probe 5 − HATC +0.012 (−0.009 to +0.034); probe 10 − HATC +0.044 (0.023 to 0.065). Per unit, HATC minus the best of NA, MC, STD has mean +0.054; HATC is better by more than 1 point in 56 units, worse in 9, level in 15; HATC is more than 2 points below NA in 3 units.

Target healthy images per unit: 68 to 571 (median 235); source and target disease images per unit: 43 to 372 (median 151).

## 2. Verification (all agree)

| Check | Result |
|---|---|
| Reproducibility: first run and rerun with patched code | identical unit values (HATC 0.787) |
| Three split seeds (offsets 0, 100000, 200000) | HATC − MC +0.0711, +0.0711, +0.0719; intervals 0.045 to 0.100; correlation of unit values with the base seed 0.996; largest unit difference 0.028 |
| Bootstrap, 5 seeds, 20,000 resamples | 90% interval 0.0451 to 0.0999 in every case |
| Leave-one-target-crop-out / leave-one-source-crop-out | 0.0668 to 0.0748 / 0.0625 to 0.0773 (most influential: garlic, corn) |
| Cluster sign-flip permutation test, 20,000 permutations | p = 0.00005 (smallest value possible) |
| Crops with a positive mean difference | 20 of 22 target crops, 21 of 22 source crops |
| Independent re-implementation (own code, exact SVD PCA, Cholesky likelihoods), 15 random units × 5 methods | maximum absolute difference 0.0021, mean 0.0003, all within 0.01 (`results/55_independent_check/`) |
| Feature spot check: 40 random images recomputed locally (pinned DINOv2) | minimum cosine similarity 1.0000, largest relative difference 0.00022; 0 missing files; 0 rows whose class differs from the folder (`results/57_feature_spotcheck/`) |
| Numerical failures of any method in any replicate | none (all runs) |
| Supervised (LDA) projection | HATC − STD (best simple for LDA) +0.0666 (0.0388 to 0.1012); mean BA NA 0.618, MC 0.660, STD 0.707, CORAL 0.778, HATC 0.774, probe 5 0.818 |
| At most 50 healthy images for the map | HATC − MC +0.0621 (0.0382 to 0.0896); mean BA HATC 0.793, MC 0.731, OC 0.749 |
| Stricter duplicate rule (distance 8; 1,864 further images removed) | HATC − MC +0.0690 (0.0439 to 0.0969); same 80 units; mean BA HATC 0.782, MC 0.713 |
| Second encoder (BioCLIP, 512-d, same images) | **smaller**: HATC − MC +0.0218 (0.0054 to 0.0400; inconclusive); mean BA NA 0.764, MC 0.797, STD 0.794, CORAL 0.820, HATC 0.819, S50 0.824, OC 0.792, probe 5/10/25 0.833/0.861/0.889; HATC minus the best simple method per unit −0.003 (better in 34 units, worse in 26) |
| Tier 3 (generic leaf spot and leaf blight; 92 units, 11 target crops; DINOv2) | HATC − MC +0.0794 (0.0477 to 0.1115); mean BA NA 0.649, MC 0.735, CORAL 0.820, HATC 0.815, OC 0.794, probe 5 0.840 |

## 3. Controls: how much of the gain is specific to the disease?

- **One-class detector (no source disease at all).** OC (Mahalanobis distance to the target healthy Gaussian, threshold at the 95th percentile of the map images) has BA 0.749, above MC (0.716), STD (0.711) and NA (0.649), and 3.8 points below HATC. With 50 map images OC is 0.749, MC 0.731. With BioCLIP OC 0.792 is level with MC 0.797.
- **Wrong-disease control** (`--control wrong_disease`; units where the source crop has another disease class with at least 30 images; 56 units): source disease model replaced by another disease of the source crop. HATC 0.772 → 0.740 (−0.031), MC 0.697 → 0.680, NA 0.638 → 0.607; the right disease beats the wrong one in 41 of 56 units; with the wrong disease HATC is still +0.061 above MC (0.026 to 0.103).
- Reading: most of the benefit over simple alignments is generic detection of "diseased versus healthy" on the new crop; the disease-specific part of the transport adds about 3 to 4 points.

## 4. Disease types and the same-pathogen pair (DINOv2, PCA)

| Disease type | Units | NA | MC | STD | CORAL | HATC | OC | Probe 5 | HATC − best simple |
|---|---|---|---|---|---|---|---|---|---|
| rust | 30 | 0.626 | 0.728 | 0.724 | 0.827 | 0.821 | 0.779 | 0.852 | +0.079 |
| mosaic virus | 20 | 0.585 | 0.624 | 0.610 | 0.665 | 0.664 | 0.615 | 0.672 | +0.037 |
| downy mildew | 12 | 0.695 | 0.797 | 0.776 | 0.863 | 0.858 | 0.823 | 0.836 | +0.057 |
| powdery mildew | 6 | 0.717 | 0.763 | 0.780 | 0.807 | 0.813 | 0.755 | 0.782 | +0.031 |
| early blight | 4 | 0.719 | 0.814 | 0.850 | 0.921 | 0.918 | 0.891 | 0.891 | +0.069 |
| black rot | 2 | 0.716 | 0.685 | 0.617 | 0.730 | 0.742 | 0.702 | 0.813 | +0.026 |
| anthracnose | 2 | 0.603 | 0.673 | 0.677 | 0.726 | 0.733 | 0.830 | 0.852 | +0.049 |
| sheath blight | 2 | 0.788 | 0.553 | 0.607 | 0.766 | 0.745 | 0.746 | 0.839 | −0.043 |
| alternaria leaf spot | 2 | 0.873 | 0.864 | 0.863 | 0.882 | 0.875 | 0.857 | 0.820 | +0.002 |

Tier 1 (tomato and potato, same pathogen; 4 units: early and late blight in both directions; descriptive, 2 crops): mean BA NA 0.819, MC 0.797, STD 0.767, CORAL 0.885, HATC 0.868, S50 0.856, probe 5/10/25 0.860/0.896/0.917; HATC − NA +0.049 (0.033 to 0.065) in the primary form with the PCA projection; with the LDA projection HATC − MC is −0.013 (−0.015 to −0.010). Three-class task (healthy, early blight, late blight) with PCA: tomato → potato NA 0.616, MC 0.613, STD 0.657, CORAL 0.702, HATC 0.696, probe 5/10/25 0.604/0.633/0.679; potato → tomato NA 0.624, MC 0.570, STD 0.501, CORAL 0.644, HATC 0.620, probe 0.622/0.646/0.672. PlantVillage laboratory images (same two crops, same diseases; pilot, `results/53_real_pair_test/pilot_*`): potato → tomato early blight NA 0.600, MC 0.740, STD 0.739, CORAL 0.800, HATC 0.807, probe 5 0.879; late blight NA 0.642, MC 0.748, STD 0.800, CORAL 0.901, HATC 0.904, probe 5 0.846; tomato → potato both near 0.96 to 0.99 for all methods.

## 5. Agreement check on real pairs (`results/54_real_agreement/pca/`)

108 (target, disease, source pair) triples; HATC failed (more than 2 points below NA, mean over the two sources) in only 1, so AUROC for failure cannot be assessed (0.72 for the bound, 0.71 ensemble disagreement, 0.66 low confidence, from one failure). Spearman correlation with HATC's gain over NA: bound +0.242 (p = 0.012; the wrong sign), ensemble disagreement −0.401 (p < 0.001), low confidence −0.453 (p < 0.001). Mean bound 0.548 (sources remain distinguishable after transport). Per-image confidence of HATC detects its errors with AUROC 0.739 (mean over units).

## Domain-matched control (post hoc; `results/53_real_pair_test/pw_pca_domainmatch/run/`)

Only images from web domains that occur in both the healthy and the disease class of a crop are kept (at least 40 healthy and 25 disease images per crop after matching). Of 84 units that pass the count rule, 48 remain in tier 2 (11 target and 11 source crops). Mean balanced accuracy: NA 0.593, MC 0.658, CORAL 0.732, HATC 0.728, OC 0.675. HATC minus MC +7.0 points (90% interval 3.2 to 11.0); OC minus MC +1.6 (-2.2 to +5.8); HATC minus OC +5.3. On the same 48 units without matching: HATC minus MC +7.9. Command: `python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --output results/53_real_pair_test/pw_pca_domainmatch --domain-match <data>/plantwild/url_record.json`.

## Checks added in the second round of post-hoc checks (post hoc; registration section 9; modules 60 to 62)

Covariance estimators, HATC-S and the frame-only control: see `docs/modules/62_sensitivity_summary.md` and `results/62_sensitivity_summary/summary.csv`. Calibration and decision rule: `docs/modules/61_calibration_decision_rule.md`. Frame-only control: `docs/modules/60_background_control.md`.

## Checks added in the third round of post-hoc checks (post hoc; registration section 10; modules 63 and 64)

Learning curve (HATC minus MC: +1.0, +4.8, +6.2, +7.0, +7.1 points at 10, 25, 50, 100, 200 healthy images); STD minus MC -0.5 and CORAL minus MC +7.4 (the whole gain is covariance); EIG (eigenvector-matched rotation) 0.606 against HATC 0.787; UDA_CORAL 0.843 and UDA_BW 0.845 (unlabelled diseased target images, transductive); BioCLIP gains by disease type; label-free risk scores (false-alarm rate on unused healthy images: rho -0.42 with HATC accuracy); neighbour-disagreement label-noise proxy. Tables: `results/63_review4_summary/`, `results/64_risk_noise_release/`.
