# Module 11: affine host maps (source crop to target crop)

**Compute:** CPU. Runs locally in about 7 seconds.

**Question.** Does the affine Monge map between two crops' healthy Gaussians carry healthy images of one crop onto the other? This is step 2 of the method, for the affine branch (training size 50 to 199 in the registration; the Brenier branch is a separate module).

**Method.**
- Projection: 16 dimensions (module 08, main setting).
- For each source crop A (PlantVillage crops with at least 200 training and 50 validation healthy images) and each target crop B (rice, fig Original), the Gaussians of the healthy training features are fitted (shrunk covariance).
- The affine Monge map T between the two Gaussians is computed: x → m_B + T (x − m_A).
- **Quality** is measured on held-out images. The validation healthy images of A are mapped with T, and their Gaussian is compared with B's validation healthy Gaussian (Fréchet distance).
- **Noise floor**: the Fréchet distance between two random halves of B's training healthy images. A map is useful only if its error is near this floor.

**Note on an earlier version.** The first version compared the fitted Gaussian with itself, so every source gave the same error. This was a bug, found and fixed before the results below were written. The results come from the corrected check.

**Results** (`results/11_host_maps_affine/host_map_quality.csv`):

| Target | Noise floor | Source crops (held-out Fréchet error) |
|---|---|---|
| fig (Original) | 0.88 | tomato 1.66, soybean 1.77, pepper 1.82, blueberry 1.94, apple 2.12, cherry 2.48, corn 2.85 |
| rice | 1.21 | soybean 3.18, blueberry 4.10, apple 4.26, tomato 4.21, pepper 4.98, cherry 4.92, corn 5.47 |

**Reading.**
- Every map is above its noise floor. The affine maps carry healthy images only partly onto the target crop. This is consistent with the Gaussianity result (module 10): the healthy distribution has structure the affine map misses.
- The best source is tomato for fig and soybean for rice. Soybean has the most training images (4,083), but the error does not follow sample size alone.
- Because the errors are relative, the paper should report each map's error beside its floor, not as an absolute score.

**Limits.**
- Target healthy counts are small (fig 357 training, rice 599). The noise floor reflects this.
- Plant IDs are not available, so the held-out split is the dataset's own split.

**How to re-run:**
```bash
python features/11_host_maps_affine.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/11_host_maps_affine
```
