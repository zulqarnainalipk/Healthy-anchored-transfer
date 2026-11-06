# Module 10: Gaussianity of healthy features (source crops)

**Compute:** CPU. Runs locally in about 11 seconds.

**Question.** Is the healthy distribution of each source crop close to a single Gaussian in the 16-dimensional projection? The affine map assumes it is (proposal section 5, step 2). This check reports how far that holds. It does not change the registered rules.

**Method.**
- For each PlantVillage crop with at least 200 training and 50 validation healthy images (7 crops), fit (a) a single Gaussian with shrunk covariance and (b) a two-component Gaussian mixture, on the training split.
- Score both on the validation split. The gain is the mixture's mean log-likelihood per image minus the single Gaussian's.

**Results** (`results/10_gaussianity/`):

| Statistic | Value |
|---|---|
| Crops tested | 7 |
| Median gain (mixture minus single, nats per image) | 0.20 |
| Smallest gain | 0.006 |
| Largest gain | 0.59 |

**Reading.** A gain near 0 would mean the single Gaussian is adequate. The gains here are small but positive in every crop, so the healthy distribution has some structure a single Gaussian misses. The affine map is therefore a simplification, and the report must say so. The Brenier map is intended to capture more of this structure, which is what the ablation in module 43 tests.

**Limits.**
- PlantVillage has no plant IDs. The split is the dataset's own train and validation split, not a plant-level split.
- Only seven crops qualify (at least 200 training healthy images).

**How to re-run:**
```bash
python features/10_gaussianity.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/10_gaussianity
```
