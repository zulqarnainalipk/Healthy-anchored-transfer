# Analysis registration (local, time-stamped)

**Status:** draft for the team's sign-off. Not a public registry. Its integrity is recorded by the SHA-256 in `registration.sha256`, which is written when this file is frozen.

**Scope.** Numeric values and rules for the study in the proposal (section 7.11). Target-dataset results are not to be used to change any value below.

**Timing deviation.** Cached features for the target datasets were extracted before this registration was written (module 05). The source-only sweeps (modules 08 and 09) use source features only. No target result was used to set any value below.

## 1. Values set before the sweep (module 09)

| Item | Value | Source |
|---|---|---|
| Pair gate for real-pair inference | 25 qualifying pairs | proposal 7.1 |
| Gate counting rule | exact name matching; look-alike matches only with causal-agent evidence (module 02) | `docs/modules/02_pair_gate.md` |
| Reaching tolerance | 2 accuracy points | proposal 5 |
| Equivalence margin | ±2 accuracy points and ±2 macro-F1 points | proposal 7.6 |
| Significance level | alpha 0.05 | proposal 7.6 |
| Target power | 0.8 | proposal 7.4 |
| Bootstrap resamples | 10,000 | proposal 7.7 |
| Minimum clusters for an interval | 8 | proposal 7.7 |
| Copy threshold (near-duplicates) | 0.90 cosine on the class token, confirmed by a pixel check (module 04b) | `docs/modules/04_near_duplicates.md` |
| Projection | shrinkage LDA, fitted on source training data only | `docs/modules/08_projection.md` |
| Projection dimensions | main 16; ablations 4 and 8 | team decision |
| Encoder | DINOv2 ViT-B/14 (`facebook/dinov2-base`) | team decision |
| Encoder revision | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` (current main revision, last modified 2024-01-17). The 05 run did not pin a revision, so this is the revision it almost certainly used; to be confirmed by a pinned re-run | `docs/modules/05_features_dinov2.md` |

## 2. Values set after the sweep (module 09), written now

These were decided after the sweep was run. They are recorded as post-sweep and are open to challenge.

| Item | Value | Note |
|---|---|---|
| Healthy-image threshold for the affine map | 200 | the sweep shows the error at 200 is within about 3% of the error at 400 |
| Healthy-image threshold for the affine map, lower bound | 50 is too few | the error at 50 is more than twice the error at 400 |
| Stability rule | smallest n whose median error is within 5% of the error at n = 400 | written after seeing the sweep; flagged as post-hoc |

## 3. Frozen files

- Feature index: `results/05_features_dinov2/remote/features_dinov2/features_index.csv`
- Splits: `results/02b_splits/splits.csv` (SHA-256 in its manifest)
- Projection: `results/08_projection/projection_k16.npz`
- Sweep: `results/09_sweep_healthy_n/summary.json`

## 4. Open items before the study

1. Confirm the encoder revision by a pinned re-run of module 05 (or accept the caveat).
2. Confirm the healthy source for each source crop (section 4 of the proposal): 12 PlantSeg crops take healthy images from PlantVillage.
3. Choose the reaching rule and the cluster-bootstrap design details (proposal 7.7).
4. Decide whether the projection dimension in section 1 is final.

## 5. Sign-off

Filed by: [name]  Date: [date]  Frozen SHA-256: see `registration.sha256`.


## 6. Team decisions (recorded)

| Decision | Value | Basis |
|---|---|---|
| Minimum training images for a source disease model | 30 (blueberry anthracnose, with 29, is excluded) | kept at the value used in module 13; not changed after seeing results |
| Disease-name mapping for fig anthracnose | the *Colletotrichum* anthracnose diseases: blueberry, peach, banana and strawberry anthracnose | causal-agent check, `docs/modules/02_pair_gate.md` |
| Disease-name mapping for rice brown spot | rice brown spot only; soybean and tobacco brown spot are excluded | different causal fungi, `docs/modules/02_pair_gate.md` |
| Fig anthracnose evidence status | not testable at present: no pushed model reaches the fig class, because blueberry anthracnose is excluded by the minimum | consequence of the decisions above |

| Synthetic-world operator strength | 2 (calibrated on the no-adaptation score only, with disjoint pools) | post-sweep and post-first-run. An earlier strength of 3 came from a version with a train-test leak and was discarded; see `docs/modules/17_benchmark_hatc.md` |
| D1 enrolment gate | kept as specified (sample count, conditioning, spread); its inability to detect a different species is stated as a limit, not redesigned | team decision after module 39 |


| Class-name harmonisation inside a crop | adopted: names of the same disease in PlantVillage and PlantSeg are merged (90 crop-disease groups become 78); all synthetic benchmark modules rerun | post-sweep and post-first-run deviation, found by the failure-case module; first-pass results kept in `results/_first_pass_unharmonised/`; see `docs/modules/03b_label_harmonisation.md` and `docs/harmonised_results.md` |
| Second encoder as sensitivity check | BioCLIP (ViT-B/16, 512-d embedding), same index, 224 px full-frame resize, same projection and benchmark code | added after the first-pass results; run as a sensitivity check only, not as a change to the primary analysis |
| Design checks of the synthetic benchmark (modules 51 and 52) | leave-one-crop-out projection, per-projection strength calibration, shrinkage sensitivity, PlantVillage-only sources, Fisher-geometry baseline, per-image refusal | post-hoc, exploratory, run after the first full set of results; the primary analysis and all registered values are unchanged |

## 7. Real-pair test on PlantWild v1 (specified on 6 November 2026, before any PlantWild feature was analysed)

**Data.** PlantWild v1 (Wei et al. 2024; 89 classes, 30 healthy crop classes, 59 disease classes; CC BY-NC-ND 4.0). Audit: `docs/modules/08b_plantwild_audit.md`. Duplicate rule applied before any analysis: images that occur in more than one class are removed (both copies); within a class only the lexicographically first path of a duplicate group is kept (exact MD5 or 64-bit difference hash with Hamming distance at most 3). 17,328 of 18,542 images remain. Features: DINOv2 ViT-B/14, same recipe and pinned revision as for the synthetic benchmark.

**Units.** An ordered triple (source crop S, target crop T, disease type D), S different from T, where S and T each have a disease class of type D and T has a healthy class. Disease types: rust, mosaic virus, powdery mildew, downy mildew, early blight, late blight, black rot, anthracnose, sheath blight, alternaria leaf spot (tier 1: same pathogen species, only tomato and potato for early blight and late blight; tier 2: same disease type, different species, main analysis; tier 3: generic leaf spot and leaf blight, exploratory). Count rule: target healthy images at least 50, source disease images at least 30, target disease images at least 30 (after duplicate removal). The source crop must also have a healthy class with at least 50 images (its healthy images define the map); this was implicit in the first wording and is made explicit on 6 November 2026, before any PlantWild result; it removes classes of crops without a healthy class (zucchini, citrus, carrot) as sources. No unit is added or removed after the features are seen.

**Task.** Binary: healthy versus disease D on crop T. Healthy model: Gaussian fitted on the target healthy images used for the map. Disease model: the source disease Gaussian moved to the target by the method under test. Decision by the larger likelihood. A three-class version (healthy, early blight, late blight) is run for tomato and potato in both directions (tier 1).

**Methods.** The healthy-only arm as in the synthetic benchmark: NA, MC, STD, CORAL, HATC (covariance shrinkage 0.1, fixed), S25, S50, S75. Labelled arm: logistic probe on 5, 10, 25 labelled target images per class. Projection: 16 dimensions, fitted **without** the target crop (leave-one-crop-out): unsupervised PCA is primary; supervised LDA (class labels of the other crops) is secondary. Healthy images used for the map: a random min(200, half of the target healthy images); of the rest, up to 25 per class are held as the labelled pool of the probe (at most one third of the rest) and all others are test images; test images of the disease are all target disease images of type D not in the labelled pool. (Clarified on 6 November 2026 before any PlantWild result; the first wording left no healthy test images for crops with fewer than 400 healthy images.)

**Replicates.** 20 random splits per unit (seeds fixed in the code); unit value = mean over the 20 splits.

**Primary outcome.** Balanced accuracy of HATC minus balanced accuracy of the best of NA, MC, STD for the same unit (best chosen by the mean over all tier-2 units, not per unit). Secondary: AUROC of the likelihood difference; probe accuracy at 5, 10, 25 labels; the same differences for CORAL and the interpolated maps.

**Statistics.** Two-way cluster bootstrap over target crops and source crops (10,000 resamples), 90% interval; equivalence margin plus or minus 2 points (balanced accuracy), alpha 0.05; outcomes equivalent, different, inconclusive; clusters at least 8 for a non-exploratory label. Tiers are analysed and reported separately; tier 1 is descriptive (two crops). No Holm correction across tiers (one primary outcome).

**Agreement check.** For (T, D) with at least three crops sharing D (all tier-2 diseases except late blight, black rot, anthracnose, sheath blight, alternaria): for each pair of source crops, discriminator accuracy and Wilson bound on the pushed disease images of the two sources (as in the synthetic benchmark); AUROC of the bound for predicting failure (HATC more than 2 points below NA); compared with per-image confidence and ensemble disagreement.

**Not allowed after the results are seen:** changing the unit list, thresholds, primary outcome, projection or shrinkage; any such change is a deviation recorded here with its reason.

## 8. Controls and verification added after the primary real-pair results (post hoc)

The registered primary outcome (section 7) is unchanged. After seeing it, the following checks were added, all labelled post hoc: a one-class healthy detector (OC) and a wrong-disease control as controls on what the gain consists of; three split seeds, a second encoder (BioCLIP), a stricter duplicate rule (Hamming distance 8), at most 50 map images, an independent re-implementation, a feature spot check, bootstrap seeds, leave-one-crop-out and a cluster permutation test (`docs/real_pair_results.md`). Tier 3 (generic leaf spot and leaf blight) was run as registered (exploratory). The agreement check (module 54) was run as registered. Numerical failures are recorded, not hidden; none occurred in the final runs. The first PlantWild runs crashed because crops without a healthy class were used as sources; the explicit source-healthy rule (section 7) was added before any result existed.

## 9. Checks added in the second round of post-hoc checks (post hoc, 6 November 2026)

The registered primary outcome (section 7) and the shrinkage value (lambda 0.1) are unchanged. After the primary result and the controls of section 8 were known, a second round of checks covers the following; all are exploratory and none changes a registered quantity: (a) other covariance estimators for every Gaussian of the real-pair test (Ledoit-Wolf, OAS, lambda 0.03 and 0.3, rank-2/4/8 factor model plus isotropic noise; `53_real_pair_test.py --cov/--lam/--rank`); (b) HATC-S, the alignment chosen from the target healthy images of the map set only (`--select`); (c) a frame-only control, the real-pair test run on 20 colour statistics of the outer 12.5% of each picture (module 60); (d) calibration (ECE, Brier, Platt scaling with 10 labels per class) of the healthy-versus-disease decision, and (e) the association between the healthy-set Bures-Wasserstein distance and the gain of HATC as a candidate decision rule (module 61). Summary: `results/62_sensitivity_summary/summary.csv` (module 62).

## 10. Checks added in the third round of post-hoc checks (post hoc, 6 November 2026)

The registered primary outcome and the shrinkage value are unchanged. The third round of checks covers the following; all are exploratory: (a) a learning curve of the real-pair test with 10, 25, 50, 100 and 200 target healthy images in the map set (`53_real_pair_test.py --map-max`, folder `53_real_pair_test/curve` and `pw_pca_n50`); (b) the split of the gain into mean, per-coordinate scale and covariance parts by disease type for DINOv2 and BioCLIP (MC, STD, CORAL, HATC); (c) a rotation-aware alignment, EIG, with the rotation fixed by matching the eigenvectors of the two healthy covariances (`--extra`); (d) unlabelled-target reference arms UDA-CORAL and UDA-BW that pool healthy and diseased source images and the unlabelled target map and test images (`--extra`; transductive, not a healthy-only method); (e) three label-free risk scores (healthy-set distance, S25/S75 disagreement, false-alarm rate on unused healthy images) with a refusal of the third of units with the largest score; (f) a label-noise proxy from the disagreement of nearest neighbours in a crop; (g) release of the kept and removed image lists, the class-name table and the leave-one-crop-out projections. Not run: mixture models for the disease classes, a leaf segmentation, a supervised anchor, conformal refusal. Modules 63 and 64.
