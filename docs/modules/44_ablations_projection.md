# Module 44: projection ablations (dimension and PCA against the supervised projection)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. About 1 minute per run, using the ablation engine (module 43).

**Question.** Does the number of projection dimensions matter, and what does the supervised projection hide compared with an unsupervised one? (Proposal section 8: 4, 8 and 16 dimensions; supervised projection against a healthy-only or unsupervised alternative.)

**Method.** The engine of module 43 is run with four projections of the same DINOv2 features, at operator strength 2, healthy n = 50 and 200, and worlds a, c50, e and f (58 replicates per cell):
- supervised shrinkage LDA, 4, 8 and 16 dimensions (module 08)
- PCA, 16 dimensions, fitted without labels on the same source training images (module 08b, components scaled to unit variance)

## Results (mean accuracy; `results/43_ablations/dim_k4`, `dim_k8`, `pca_k16` and `sweep_n_k16`)

World a (affine), healthy n = 50 / 200:

| Projection | NA | MC | SCALE | S50 | HATC |
|---|---|---|---|---|---|
| LDA 4 | 0.57 / 0.57 | 0.58 / 0.57 | 0.57 / 0.57 | 0.58 / 0.58 | 0.54 / 0.55 |
| LDA 8 | 0.54 / 0.54 | 0.57 / 0.58 | 0.57 / 0.58 | 0.60 / 0.62 | 0.56 / 0.61 |
| LDA 16 | 0.51 / 0.51 | 0.56 / 0.57 | 0.57 / 0.57 | 0.62 / 0.65 | 0.56 / 0.62 |
| PCA 16 | 0.35 / 0.34 | 0.64 / 0.65 | 0.65 / 0.66 | 0.67 / 0.68 | 0.65 / 0.68 |

## Reading

- **Dimension.** More dimensions help the aligned methods slightly in the affine world (HATC at n = 200: 0.55, 0.61, 0.62 for 4, 8, 16). The differences are small, and 16 is the best or tied. The ordering of the methods does not change with the dimension.
- **PCA versus the supervised projection.** In the unsupervised PCA space, the registered HATC does much better relative to the simple alignments: 0.65 / 0.68 in world a, level with the shrunk map and above mean-centring (0.64 / 0.65). In the supervised space, HATC was below the simple alignments.
- **Caution: the strengths are not comparable.** The operator strength (2) was calibrated on the no-adaptation score in the supervised space. In the PCA space the same strength disrupts the data more (no adaptation scores 0.35 in world a), so the two settings are not equally hard. The result shows that the conclusions depend on the projection, not that PCA is better.
- **What the supervised projection hides.** It concentrates the class information in a few directions, so a random operator in that space moves class means a lot relative to the within-class spread. The full map's estimation noise then costs more. This is one reason HATC loses to simple alignments in the main benchmark.

## Not done

- **Leave-one-source-dataset-out** is not possible: PlantSeg has no healthy images (module 01), so the two source datasets cannot be dropped one at a time for the host-map step.
- **Single Gaussian against mixture disease models** and **shrinkage-grid sensitivity of the class models** are not run. The benchmark uses single Gaussians for class models with a fixed shrinkage of 0.1.
- **Healthy-only PCA** (PCA on healthy images alone) is not run; the PCA here uses all source images.
- Only four worlds and one operator strength. No intervals.

**How to re-run:**
```bash
python features/08b_projection_pca.py --features results/05_features_dinov2/remote/features_dinov2 --output results/08b_projection_pca
python features/43_ablation_engine.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08b_projection_pca/projection_k16.npz --output results/43_ablations --tag pca_k16 --healthy-n 50 200 --worlds a c50 e f --reps 100
```
