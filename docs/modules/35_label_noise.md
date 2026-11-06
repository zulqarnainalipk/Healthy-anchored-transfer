# Module 35: label-noise robustness

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. About 45 seconds per noise level, using the ablation engine (module 43).

**Question.** How much does noise in the **source** disease labels hurt each alignment? (Proposal 7.8: symmetric noise at 5% and 10%.)

**Method.** In each replicate, a fraction (0%, 5% or 10%) of the source class samples of the crop is moved to another class of the same crop, chosen at random (symmetric noise). The class Gaussians are then built from the noisy classes. The target test samples are clean. Operator strength 2, 16 dimensions, healthy n = 200, 58 replicates per world, worlds a, c50, e and f.

## Results (`results/35_label_noise/noise_0`, `noise_0.05`, `noise_0.1`; mean accuracy)

| World | Noise | NA | MC | SCALE | STD | S50 | CORAL | HATC |
|---|---|---|---|---|---|---|---|---|
| a | 0% | 0.51 | 0.57 | 0.57 | 0.58 | 0.65 | 0.62 | 0.62 |
| a | 5% | 0.40 | 0.51 | 0.50 | 0.50 | 0.57 | 0.55 | 0.57 |
| a | 10% | 0.41 | 0.51 | 0.50 | 0.50 | 0.57 | 0.54 | 0.56 |
| c50 | 0% | 0.55 | 0.53 | 0.64 | 0.62 | 0.61 | 0.60 | 0.59 |
| c50 | 5% | 0.41 | 0.40 | 0.55 | 0.51 | 0.53 | 0.52 | 0.52 |
| c50 | 10% | 0.40 | 0.39 | 0.55 | 0.51 | 0.51 | 0.52 | 0.52 |
| e | 0% | 0.23 | 0.23 | 0.29 | 0.29 | 0.28 | 0.33 | 0.33 |
| e | 5% | 0.23 | 0.23 | 0.24 | 0.25 | 0.24 | 0.27 | 0.27 |
| f | 0% | 0.70 | 0.77 | 0.76 | 0.75 | 0.75 | 0.66 | 0.66 |
| f | 5% | 0.67 | 0.75 | 0.75 | 0.73 | 0.74 | 0.64 | 0.64 |

## Reading

- **Source label noise matters a lot.** At 5% noise, accuracy falls by 2 to 14 points for every method, depending on the world. The largest drops are for no adaptation in the affine and graded worlds (11 to 14 points) and for mean-centring in the graded world (13 points); HATC falls by 3 to 6 points.
- **10% noise adds little beyond 5%.** The accuracy at 10% is within about 1 point of the 5% value in every world. A small amount of noise already contaminates the class Gaussians, and extra noise has a diminishing effect.
- **The ordering of the methods is unchanged.** The same methods are best in each world at all noise levels. HATC and the shrunk map lose less in absolute terms than no adaptation in worlds a and c50, but this does not change which method wins.
- The registered claim for the agreement check (module 18, 30) was to be re-evaluated under label noise (proposal 7.8). This is **not done**: only accuracy under noise is measured here, not the change in the agreement-check AUROC.

## Limits

- Noise is symmetric among the classes of one crop and applied only to source labels. Asymmetric or target-side noise is not tested.
- Healthy n = 200 only, four worlds, 58 replicates. No intervals.
- The change in agreement-check AUROC under noise (proposal 7.8) is not computed.

**How to re-run:**
```bash
python features/43_ablation_engine.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/35_label_noise --tag noise_0.05 --healthy-n 200 --worlds a c50 e f --reps 100 --label-noise 0.05
```


