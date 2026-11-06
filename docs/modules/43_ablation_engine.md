# Module 43: ablation engine and healthy-sample sweep

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. About 3 minutes for the sample-size sweep (four worlds, six healthy sizes, 58 replicates per cell).

**Question.** How many healthy images does each alignment need, and does shrinking the affine map help? (Proposal section 8, ablations: sample sweep; affine against other maps.)

**Method.** The engine runs the fixed alignments of modules 20 and 50 on the synthetic worlds, with the same disjoint pools and replicate structure: NA, MC (mean shift), SCALE (isotropic scaling), STD (per-dimension), S25, S50, S75 (the affine map shrunk toward the identity by 25%, 50%, 75%), CORAL and HATC (the full affine Monge map). Operator strength 2, 16 dimensions, 58 replicates per cell (crops with at least two disease classes of 60 or more images).

## Results (`results/43_ablations/sweep_n_k16/summary.csv`; mean accuracy)

World a (affine relation):

| Healthy n | NA | MC | SCALE | STD | S50 | S75 | HATC |
|---|---|---|---|---|---|---|---|
| 10 | 0.51 | 0.57 | 0.57 | 0.54 | 0.55 | 0.47 | 0.38 |
| 25 | 0.51 | 0.57 | 0.57 | 0.57 | 0.61 | 0.57 | 0.49 |
| 50 | 0.51 | 0.56 | 0.57 | 0.57 | 0.62 | 0.61 | 0.56 |
| 100 | 0.51 | 0.57 | 0.57 | 0.58 | 0.64 | 0.65 | 0.61 |
| 200 | 0.51 | 0.57 | 0.57 | 0.58 | 0.65 | 0.65 | 0.62 |
| 400 | 0.51 | 0.57 | 0.57 | 0.58 | 0.65 | 0.66 | 0.64 |

Other worlds, at n = 50 / 200: c50 (graded departure): SCALE 0.63 / 0.64 is best, HATC 0.54 / 0.59. e (rotation and shear): HATC 0.32 / 0.33 and CORAL 0.32 / 0.33 are best. f (shift): MC 0.77 / 0.77 is best, HATC 0.59 / 0.66.

## Reading

- **The full affine map needs many healthy images.** HATC is far below no adaptation at 10 images (0.38 against 0.51 in world a), overtakes it only at about 50 to 100 images, and keeps improving to 400. The registered threshold of 200 images for the affine map is a sensible minimum; 50 is too few.
- **Shrinking the map helps at every size in the affine world.** S50 and S75 beat the full map by 3 to 17 points between 10 and 100 images, and by about 3 points at 200. The shrinkage also costs nothing at large n.
- **No single method wins everywhere.** SCALE is best in the graded world, MC in the shift world, the full map or CORAL in the rotation world, and the shrunk maps in the affine world.
- This agrees with module 20 and module 50: the full affine map's weakness is estimation noise, and regularisation addresses it.

## Not done (stated as future work)

- **The Brenier (nonlinear) branch** for 200 or more healthy images is **not implemented**. The comparison "affine against Brenier" of proposal section 8 and the extension rule (the affine extension outside the healthy support) are therefore **not tested**.
- The sample-size sweep covers four worlds, not all ten.
- No intervals on these means (module 36 computes them for the registered contrasts only).

**How to re-run:**
```bash
python features/43_ablation_engine.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/43_ablations --tag sweep_n_k16 --healthy-n 10 25 50 100 200 400 --worlds a c50 e f --reps 100
```
