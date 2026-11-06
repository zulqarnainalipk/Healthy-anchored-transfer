# Module 60: frame-only (background) control

**Code:** `analysis/60_background_control.py`. **Compute:** CPU, about 30 minutes for 17,328 images (image decoding). **Question:** does the gain of HATC over mean-centring come from photographic style and not from the leaves?

**Recipe.** Each image is reduced to 128 x 128 pixels; only the outer 16 pixels on every side (12.5%, mostly background) are described by the mean and standard deviation of R, G, B, H, S and V and an 8-bin hue histogram (20 values). The matrix `results/60_background_control/features_plantwild_border.npy` has the same rows as the DINOv2 features and is passed to module 53 (`--features-name features_plantwild_border.npy`, tag `border`). The script also writes `frame_vs_dinov2_auroc.csv`: per crop (at least 40 healthy and 40 diseased images), the 5-fold cross-validated AUROC of a logistic classifier separating healthy from diseased images from the frame and from DINOv2 features.

**Result.** Mean AUROC over 28 crops: 0.714 (frame) and 0.974 (DINOv2). Real-pair test, tier 2, 80 units: NA 0.663, MC 0.623, CORAL 0.609, HATC 0.607, one-class 0.503; HATC minus MC -1.6 points (90% interval -3.1 to -0.3), HATC minus NA -5.6. The frame carries some style information but the alignments do not exploit it. **Limits:** a proxy for the background, not a leaf segmentation.

**How to re-run:** see the docstring of the script.

## How to re-run

```
python analysis/60_background_control.py --data-root <data-root>/plantwild --features results/09_features_plantwild/remote/features_plantwild --output results/60_background_control
```

The test on the frame-only features is the module-53 command given in the recipe above (tag `border`).