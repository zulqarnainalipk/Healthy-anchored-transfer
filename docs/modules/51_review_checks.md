# Module 51: design checks of the synthetic benchmark (projection fairness, recalibration, shrinkage, source dataset, Fisher baseline)

**Status:** exploratory, run after the first full set of results as design checks; not part of the pre-specified plan (see `docs/registration.md`, section 6).
**Compute:** CPU. LOCO projection fits about 2 to 3 minutes per crop (LDA), 10 seconds (PCA); each benchmark run 25 to 30 minutes on one core. Code: `features/51_loco_projection.py` (modes `fit`, `calibrate`, `run`); shrinkage and source-dataset runs use `features/43_ablation_engine.py` with the environment variables `HATC_SHRINK` and `SOURCE_ONLY`.

## Questions and design

1. **Does the supervised projection favour the probe?** The projection of module 08 is fitted on source classes that include the classes of the crop used as target. Here it is refitted without the target crop (leave-one-crop-out, LOCO), as supervised LDA or as unsupervised PCA, 16 dimensions, one fit per target crop (cached in `results/51_review_checks/loco/`). Benchmark: 200 healthy images, ten worlds, seven target crops, 17 replicates per crop and world, methods NA, MC, STD, CORAL, HATC, S50, two Fisher-geometry variants and a logistic probe with 5, 10, 25 labels per class.
2. **Is the PCA exception an effect of shift strength?** Operator strength recalibrated per projection with the registered rule (smallest strength with mean no-adaptation accuracy over the worlds in 0.40 to 0.65; `results/51_review_checks/calib/`).
3. **How sensitive is HATC to covariance shrinkage?** λ in {0.01, 0.03, 0.1, 0.3, 0.5}; registered value 0.1; formula Σ̃ = (1 − λ) Σ̂ + λ (tr Σ̂ / d) I; worlds a, c50, e, f; 50 and 200 healthy images.
4. **Does mixing PlantSeg and PlantVillage matter?** Source crops restricted to PlantVillage (`SOURCE_ONLY=plantvillage_color`).
5. **ZFGA-style baseline** (our reading of the technical note arXiv:2609.37302; not the authors' code): Fisher information of the source class posterior with respect to the features, averaged over healthy source images (reference) and healthy target images (test); A = (I_te + εI)^(−1/2) (I_ref + εI)^(1/2); target features z' = A z ("ZFGA") or μ_A + A (z − μ_B) ("ZFGA_MC").

## Results

| Projection (strength) | Mean HATC − best simple (points) | Worlds below / above | HATC | Probe 5 | Probe 10 | Probe 25 |
|---|---|---|---|---|---|---|
| Original LDA (2) | −1.5 | 6 / 2 | 0.659 | 0.692 | 0.716 | 0.742 |
| LOCO LDA (1.5) | −1.9 | 9 / 1 | 0.658 | 0.625 | 0.661 | 0.691 |
| LOCO PCA (2) | −0.9 | 5 / 3 | 0.663 | 0.608 | 0.639 | 0.678 |
| Global PCA of module 08b (1.0; module 43) | −1.9 | 10 / 0 | 0.708 | | | |

Labels per class needed for the probe to come within 2 points of HATC (grid 5, 10, 25), per world: original projection 5 in six worlds and 10 in four; LOCO LDA 5 in two, 10 in three, 25 in one, more than 25 in four; LOCO PCA 5 in one, 25 in five, more than 25 in four.

- **The ranking of the healthy-only methods does not depend on the projection having seen the target classes** (HATC below the best simple method in 5 to 10 worlds in every projection).
- **The label-saving result does.** Leaving the target crop out of the projection lowers the 5-label probe by 7 to 8 points. "Five labels per class suffice" holds only for the original projection; 10 to more than 25 labels are needed otherwise.
- **PCA exception:** at the calibrated strength (1.0) HATC is below the best simple method in all ten worlds (0.6 to 2.4 points). The exception at strength 2 was an effect of shift magnitude (no-adaptation accuracy 0.41 there).
- **Calibrated strengths:** original 2 (mean NA 0.64), LOCO LDA 1.5 (0.65), LOCO PCA 2 (0.64), global PCA 1.0 (0.63).
- **Shrinkage** (mean HATC − best simple over worlds a, c50, e, f at n = 200): λ = 0.01: −5.1; 0.03: −3.2; **0.1: −0.6**; 0.3: −0.2; 0.5: −0.4 points. Too little shrinkage hurts (c50 at 0.01: −10.4); λ = 0.3 to 0.5 brings HATC level with the best simple method, not above it. The best λ was found on the benchmark, so this is not a healthy-only selection of λ (not tested).
- **PlantVillage-only sources:** HATC below the best simple method in 9 of 10 worlds (−1.7 to −10.5 points), above only in world e (+0.5).
- **ZFGA-style:** accuracy 0.43 to 0.52 against 0.64 for no adaptation, in all projections, directly or after mean-centring. Our adaptation to Gaussian class models in a discriminant space does not work; this is not a test of the method in its own setting.
- **Closed-form BW alignment:** HATC's map T is the exact solution of the Bures–Wasserstein objective that ITSPACE (arXiv:2606.30523) optimises iteratively under low-rank constraints, so no iterative variant was run.

## Limits

Seven target crops and 17 replicates per crop and world; no intervals; strengths differ between projections; the per-projection calibration uses 10 replicates per crop; ZFGA-style is an approximation; shrinkage chosen on the benchmark worlds.

**How to re-run:**
```bash
python features/51_loco_projection.py --mode fit --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --output results/51_review_checks/loco
python features/51_loco_projection.py --mode calibrate --features ... --labelled ... --output results/51_review_checks/calib --projections loco_lda loco_pca --scales 1 1.5 2 3 4 5 6 --reps-per-crop 10 --tag loco
python features/51_loco_projection.py --mode run --features ... --labelled ... --output results/51_review_checks/run --projections base --scale 2 --healthy-n 200 --reps-per-crop 17 --tag base2
python features/51_loco_projection.py --mode run --features ... --labelled ... --output results/51_review_checks/run --projections loco_lda --scale 1.5 --healthy-n 200 --reps-per-crop 17 --tag scale1.5
python features/51_loco_projection.py --mode run --features ... --labelled ... --output results/51_review_checks/run --projections loco_pca --scale 2 --healthy-n 200 --reps-per-crop 17 --tag pca2
# shrinkage (HATC_SHRINK=0.3) and PlantVillage-only (SOURCE_ONLY=plantvillage_color): features/43_ablation_engine.py, outputs in results/51_review_checks/shrinkage and plantvillage_only
```
