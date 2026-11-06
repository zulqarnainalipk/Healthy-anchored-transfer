# Module 08: supervised projection (source crops only)

**Compute:** CPU. Reads cached features only. Runs locally in about 2.5 minutes.

**Question.** Which low-dimensional projection of the DINOv2 features best separates the disease classes of the source crops? This is step 1 of the method (proposal section 5).

**Method.**
- Fitted only on the **training split of the source datasets** (PlantSeg and PlantVillage): 51,358 images, 137 classes. Target datasets (rice, fig, chili, peach) are never used.
- Shrinkage linear discriminant analysis (`solver = eigen`, `shrinkage = auto`). The centring is the overall training mean.
- The first 4, 8 and 16 discriminant directions are kept, as the proposal specifies.
- Accuracy is measured on the **validation split of the source datasets** (6,677 images), using the nearest class centroid in the projected space.

**Results** (`results/08_projection/`):

| Dimensions | Validation accuracy | Validation images | Images of unseen classes dropped |
|---|---|---|---|
| 4 | 0.507 | 6,677 | 0 |
| 8 | 0.723 | 6,677 | 0 |
| 16 | 0.805 | 6,677 | 0 |

Projection files: `projection_k4.npz`, `projection_k8.npz`, `projection_k16.npz`. Each contains the centring mean and the projection matrix.

**Choice of method.** Shrinkage LDA was chosen because it uses the disease labels and needs no target data. The proposal allows partial least squares as an alternative; it is not run here.

**Notes.**
- Some training classes have only one image. This gives a warning about the covariance estimate. The effect on the projection is small, but it should be reported.
- Accuracy rises with the number of dimensions. The registered choice of dimension is still open (see the decision below).

**Decision needed before registration.** Which dimension is the main setting: 16 (highest validation accuracy), or 8 (smaller, with the other two as ablations)? The proposal lists all three.

**How to re-run:**
```bash
python features/08_projection.py --features results/05_features_dinov2/remote/features_dinov2 --output results/08_projection
```
