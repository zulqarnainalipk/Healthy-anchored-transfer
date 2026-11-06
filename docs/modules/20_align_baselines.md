# Module 20: alignment baselines on the synthetic worlds

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in about 3.5 minutes.

**Question.** On the same worlds as module 17, how does HATC compare with the other healthy-only alignment methods (proposal section 8, healthy-only arm)?

**Methods** (single source; the target's healthy images are the only target data):
- **NA:** no adaptation.
- **MC:** mean-centring.
- **STD:** per-dimension standardisation (AdaBN-like).
- **CORAL:** second-order alignment with the non-symmetric map C_B^{1/2} C_A^{-1/2}.
- **WHITEN:** whiten the target with its own healthy covariance and re-colour with the source healthy covariance. It gives accuracies identical to CORAL; it is the same alignment applied to the target samples instead of the source models.
- **HATC:** the affine Monge map (symmetric positive-definite), module 11 logic.

**Correction (see `17_benchmark_hatc.md`).** The first version of this module had a train-test leak and the operator strength of 3. Both are corrected: pools are now disjoint, and the strength is 2. The tables below are from the corrected run. 117 replicates per cell.

## Results (`results/20_align_baselines/summary.csv`; mean accuracy, healthy n = 50 / 200)

| World | NA | MC | STD | CORAL | HATC | Best |
|---|---|---|---|---|---|---|
| a (affine) | 0.51 / 0.51 | 0.57 / 0.57 | 0.58 / 0.59 | 0.54 / 0.60 | 0.55 / 0.61 | STD / HATC |
| a2 (shift) | 0.68 / 0.67 | 0.77 / 0.77 | 0.74 / 0.75 | 0.57 / 0.64 | 0.56 / 0.64 | MC / MC |
| b (radial) | 0.61 / 0.61 | 0.60 / 0.60 | 0.74 / 0.75 | 0.57 / 0.64 | 0.56 / 0.64 | STD / STD |
| b2 (non-composing) | 0.63 / 0.62 | 0.59 / 0.59 | 0.58 / 0.58 | 0.53 / 0.57 | 0.53 / 0.58 | NA / NA |
| c25 | 0.57 / 0.58 | 0.57 / 0.58 | 0.61 / 0.62 | 0.54 / 0.59 | 0.54 / 0.59 | STD / STD |
| c50 | 0.55 / 0.55 | 0.53 / 0.54 | 0.58 / 0.59 | 0.51 / 0.56 | 0.52 / 0.55 | STD / STD |
| c100 | 0.49 / 0.49 | 0.47 / 0.47 | 0.53 / 0.53 | 0.48 / 0.51 | 0.47 / 0.50 | STD / STD |
| d, f (shifts) | as a2 | as a2 | as a2 | as a2 | as a2 | MC / MC |
| e (rotation, shear) | 0.23 / 0.23 | 0.23 / 0.23 | 0.29 / 0.29 | 0.33 / 0.34 | 0.33 / 0.34 | HATC / HATC |

## Reading

- **HATC is the best method in only 3 of 20 cells** (world a at 200 images and world e). Per-dimension standardisation is best in 9 cells, mean-centring in 6, and no adaptation in 2.
- **HATC and CORAL are indistinguishable.** They differ by at most 0.01 in accuracy. HATC adds nothing over CORAL in this benchmark. The proposal already anticipates a tie with per-crop alignment baselines.
- **Simple alignment beats the map in most worlds.** Standardisation wins in the radial and graded worlds, and mean-centring wins in the shift worlds. The affine map pays an estimation cost that the simple methods avoid.
- **More healthy images help the map.** The gap between 50 and 200 images is consistent for HATC and CORAL, but not for NA, MC and STD.
- **No method does well in world e.** The best score is 0.34. The rotation is not identifiable from healthy distributions alone (see below).

## An expectation in the proposal does not hold

The proposal expected a general affine registration to win in world (e). With healthy distributions only and no paired points, the rotation part of an affine map is not identifiable: any map A with A C_A A′ = C_B carries the source covariance to the target covariance, for every rotation. No healthy-only method can recover the rotation. The proposal text is corrected (sections 7.4 and 8). A general affine registration needs paired points and is not part of the healthy-only arm.

## Limits

- Single source only. The agreement check is in module 18.
- The worlds' operator strength was calibrated post hoc, and the worlds are first-draft.
- No confidence intervals yet (module 36). 117 of 200 replicates ran per cell.
- Baselines from the other arms (unlabelled-field and labelled-target) are only partly covered (module 26 for the probes).

**How to re-run:**
```bash
python features/20_align_baselines.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/20_align_baselines --scale 2
```
