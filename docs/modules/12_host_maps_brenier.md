# Modules 12 and 44b: nonparametric host map, affine extension, diversified predictions

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU, about 2 minutes locally (`features/12_host_maps_brenier.py`).

**Question.** Does a nonparametric optimal-transport map between healthy images (the Brenier branch, proposal 5, used from 200 healthy images) improve on the affine map? Does a nonparametric correction on top of the affine map (the extension rule) help? Do diversified predictions (bootstrap maps; affine plus nonparametric) help? (Proposal 7.4 and 8.)

**Method.** Same worlds and pools as modules 17, 20, 43, 46b; healthy n = 200 and 400; operator strength 2; 16 dimensions; 34 replicates per world; worlds a, c50, e, f.
- **BRENIER:** exact empirical optimal transport (Hungarian assignment, squared Euclidean cost) between a source healthy sample and the target healthy sample, both of size n (the source sample is limited by the crop's healthy pool). It is extended to new points by inverse-distance barycentric interpolation over the 10 nearest source healthy points.
- **BRENIER_AFF:** the affine map plus the interpolated residual of the empirical map.
- **HATC_BOOT:** HATC fitted on 10 bootstrap resamples of the target healthy images, probabilities averaged. **ENS:** average of HATC and BRENIER_AFF probabilities.

## Results (`results/12_host_maps_brenier/k16/summary.csv`, accuracy)

| World | n | HATC | BRENIER | BRENIER_AFF | ENS |
|---|---|---|---|---|---|
| a | 200 | 0.626 | 0.214 | 0.614 | 0.623 |
| a | 400 | 0.637 | 0.210 | 0.623 | 0.631 |
| c50 | 200 | 0.592 | 0.226 | 0.578 | 0.586 |
| c50 | 400 | 0.602 | 0.208 | 0.581 | 0.593 |
| e | 200 | 0.332 | 0.262 | 0.330 | 0.330 |
| e | 400 | 0.332 | 0.242 | 0.334 | 0.334 |
| f | 200 | 0.670 | 0.207 | 0.656 | 0.662 |
| f | 400 | 0.681 | 0.213 | 0.664 | 0.674 |

(HATC_BOOT, in the csv, is 0.0 to 1.4 points below HATC in every cell, for example 0.614 against 0.626 in world a at n = 200.)

## Reading

- **The pure empirical map fails.** BRENIER is near chance in every world. The reason is the extension: in 16 dimensions with 200 to 400 healthy points, the 10 nearest healthy points are far from a disease image, and averaging their matched images pulls every point towards the target healthy centre, so the classes collapse onto each other. This is a failure of my simple interpolation, not proof that a better nonparametric map cannot work. I did not tune it (no kernel width search, no smoothing-potential fit), because tuning on these worlds would break the rule of fixing methods before reading results.
- **The affine extension is safe but not better.** BRENIER_AFF is within about 1 to 2 points below HATC in every cell: the residual adds noise rather than signal. The proposal's rule (use the nonparametric map only where the data are dense; otherwise the affine part) behaves as the safe fallback, which is what it is for.
- **Diversified predictions do not help.** ENS is 0 to 1 point below HATC and HATC_BOOT 0 to 1.4 points below. In these worlds, in which the true relation is affine or a simple shift, there is nothing for a second, nonlinear predictor to add.
- The synthetic worlds contain no strongly nonlinear host relation except world c, so a gain from the nonparametric branch is not expected here; the test of the branch on real nonlinear relations is not possible (module 32 and 33 infeasible).

## Limits

Naive nearest-neighbour extension; the source healthy sample is limited to the crop's pool (60 to 2,041 images, so for some crops fewer than n images are matched); four worlds; 34 replicates; no intervals. A proper Brenier implementation (smooth potential, for example by entropic OT or a learned potential network) is not tried.

**How to re-run:**
```bash
python features/12_host_maps_brenier.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/12_host_maps_brenier --tag k16 --healthy-n 200 400 --worlds a c50 e f --reps 58
```
