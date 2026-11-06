# Module 09: source-only sweep of the healthy-image sample size

**Compute:** CPU, about 12 seconds locally.

**Question.** How many healthy images does a crop need before its affine host map is stable? Source crops only, so no target result can influence the choice (an accepted timing deviation, see `docs/registration.md`).

**Method.** PlantVillage healthy images of crops with enough healthy images (7 crops, 42 ordered pairs). For each ordered pair and each n in 25, 50, 100, 200, 400 (5 repeats, seed 2026): sample n healthy training images of each crop, fit a shrunk Gaussian (shrinkage 0.1) in the 16-d supervised projection (module 08), map one crop to the other with the affine Monge map, and measure the Bures-Wasserstein distance to the held-out healthy validation images of the target crop.

**Result** (`results/09_sweep_healthy_n/summary.json`): the median error over pairs falls from 0.98 (n = 25) to 0.57 (50), 0.38 (100), 0.246 (200) and 0.240 (400). At n = 200 the error is 1.03 times the error at n = 400.

**Reading.** The map is close to stable from about 200 healthy images, and clearly unstable at 25. The two thresholds used in the benchmark, 50 and 200, bracket this.

**Limits.** Affine map only (the Brenier map was not part of this sweep); healthy images from one dataset (PlantVillage); 5 repeats.

**How to re-run:**
```bash
python features/09_sweep_healthy_n.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/09_sweep_healthy_n
```
