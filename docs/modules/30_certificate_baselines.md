# Module 30: label-free reliability scores against the agreement bound

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. About 7 minutes, run as four parallel jobs (worlds split into four groups on four CPU cores). The four AUROC tables were merged afterwards.

**Question.** Does the agreement bound predict transfer failure better than standard label-free reliability scores? The registered condition for claiming the agreement check is that it beats these scores (proposal sections 1 and 8).

**Setup.** The same two-source setup, worlds, disjoint pools (50% source models, 25% test, 25% probe) and failure label as module 18, at operator strength 2. A replicate **fails** when the mean HATC accuracy of the two sources is below the no-adaptation accuracy by more than 0.02. 117 replicates per cell.

**Scores** (all label-free):
- **bound:** the agreement bound of module 18 (higher predicts failure).
- **ens_disagree:** the fraction of target samples on which the two sources' classifiers disagree (ensemble disagreement across sources).
- **low_conf:** 1 minus the mean maximum posterior of the source-1 classifier on the target samples (low confidence predicts failure).

## Results (`results/30_certificate_baselines/auroc.csv`; AUROC against the failure label)

| Statistic | bound | ens_disagree | low_conf |
|---|---|---|---|
| Mean over 20 cells | **0.61** | 0.46 | 0.31 |
| Median | 0.63 | 0.44 | 0.28 |
| Cells with AUROC above 0.6 | 11 | 2 | 2 |
| Best of the three in the cell | **18** | 2 | 0 |

By world (healthy n = 50 / 200; AUROC):

| World | bound | ens_disagree | low_conf |
|---|---|---|---|
| a | 0.54 / 0.51 | 0.29 / 0.35 | 0.19 / 0.29 |
| a2, d | 0.67 / 0.50 | 0.42 / 0.33 | 0.17 / 0.18 |
| b | 0.58 / 0.64 | 0.45 / 0.37 | 0.29 / 0.08 |
| b2 | 0.44 / 0.60 | 0.56 / 0.55 | 0.43 / 0.33 |
| c25 | 0.40 / 0.62 | 0.55 / 0.36 | 0.46 / 0.26 |
| c50 | 0.63 / 0.71 | 0.49 / 0.49 | 0.37 / 0.26 |
| c100 | 0.67 / 0.69 | 0.46 / 0.55 | 0.36 / 0.29 |
| e | **0.87 / 0.86** | 0.70 / 0.78 | 0.68 / 0.76 |
| f | 0.64 / 0.50 | 0.43 / 0.31 | 0.23 / 0.19 |

## Reading

- **With the conventional direction for each score, the bound is the best in 18 of 20 cells**, with a mean AUROC of 0.61 against 0.46 and 0.31. It is clearly the best in world e (rotation and shear), where all three score above 0.5.
- **An AUROC well below 0.5 is not "no signal".** It means the score is related to failure with the opposite sign. The low-confidence score has a mean AUROC of 0.31, which means high confidence predicts failure in these worlds. Reversed, it scores about 0.69, above the bound's 0.61. Ensemble disagreement reversed scores about 0.54.
- The reversal has a likely cause. The affine map rescales covariances, which can make the transported classifier more confident while it is wrong. Failure here means HATC is worse than no adaptation, which is more common when the map is overconfident.
- **The registered condition is therefore only conditionally met.** The bound beats the label-free scores in their conventional direction. A confidence score used with the learned (reversed) direction would beat the bound. The claim for the agreement check cannot be made cleanly from this evidence.

## Limits

- The same simplified two-source construction as module 18 (source 2 is source 1 seen through a fixed affine look), so this says little about real source pairs.
- Only 7 crops per cell, below the registered minimum of 8 clusters. No confidence intervals on the AUROC. All comparisons are exploratory.
- The failure label depends on the no-adaptation baseline and a margin of 0.02. A different definition would change all three scores.
- The healthy-sample-size baseline (a constant per cell) has AUROC 0.5 by construction and is not tabulated.
- The disagreement discrepancy, agreement-on-the-line, calibration and OOD baselines of proposal section 8 are not yet implemented.

**How to re-run (one group):**
```bash
python features/30_certificate_baselines.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/30_certificate_baselines/g1 --scale 2 --worlds a a2 b
```
The four groups were: (a, a2, b), (b2, c25, c50), (c100, d), (e, f). Merge the four `auroc.csv` files.
