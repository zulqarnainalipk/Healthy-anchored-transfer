# Module 50: HATC-S, healthy-validated alignment selection (post-hoc, exploratory)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Status.** This is a **post-hoc extension**, designed after the registered benchmark (module 20) had been read. The registered method stays **HATC**, and every registered result is reported as it is. HATC-S is exploratory. Seeds 2026 and 9001 were used in the design; seed 9002 was untouched until one final run.

**Compute:** CPU. About 5 to 8 minutes per run, with two runs in parallel.

## Motivation

No single fixed alignment wins in every world (modules 17, 20 and 37): mean-centring wins for shifts, standardisation for scaling, and the affine map only for near-affine relations with plenty of healthy images. A device always has the target's healthy images, which need no labels. HATC-S uses them to validate the alignment:

1. **Candidates.** Iteration 1: NA, MC, STD, CORAL, HATC and a shrunk map S50 = 0.5 I + 0.5 T (T is the affine Monge map). Iteration 2 adds an isotropic scaling (SCALE) and two more shrinkage levels (S25, S75).
2. **Healthy validation.** Each candidate is fitted on the source healthy sample and half of the target healthy sample. The source healthy sample is transported, and its Fréchet distance to the other half of the target healthy sample is measured. The halves are swapped and the errors averaged.
3. **Selection.** The candidate with the lowest healthy transport error is chosen (iteration 2: the simplest candidate within 5% of the lowest error). It is refitted on all target healthy images and classifies the target disease samples.
4. **Certificate.** The lowest healthy error is a label-free reliability score.

No disease labels and no target disease images are used in the selection.

## Results (mean accuracy over 20 world-and-size cells; 117 replicates per cell)

| Method | Orig. (seed 2026), iter 1 | Fresh (9001), iter 1 | Orig., iter 2 | **Final (9002), iter 2, one-shot** |
|---|---|---|---|---|
| NA | 0.562 | 0.543 | 0.562 | 0.548 |
| MC | 0.584 | 0.598 | 0.584 | 0.613 |
| STD | 0.621 | 0.640 | 0.621 | 0.636 |
| SCALE | | | 0.637 | **0.656** |
| CORAL | 0.567 | 0.575 | 0.567 | 0.570 |
| **HATC (registered)** | 0.566 | 0.577 | 0.566 | 0.574 |
| S50 | 0.618 | 0.632 | 0.618 | 0.632 |
| **SELECT (HATC-S)** | 0.603 | 0.617 | 0.613 | 0.637 |

- **SELECT beats HATC** in 13 to 18 of the 20 cells in every set, and is worse in only 1 or 2. On the untouched final set (seed 9002) the mean is 0.637 against 0.574 for HATC: **about +6 points**, with SELECT better than HATC in 18 of 20 cells.
- **The gain comes mostly from not using the full map.** The shrunk maps (S25, S50, S75) and the one-parameter SCALE all beat the registered HATC by 3 to 8 points on average. The full affine Monge map is the weak part, because it pays a large estimation cost at 50 to 200 healthy images in 16 dimensions.
- **SELECT does not beat the best single fixed candidate.** It trails the best fixed candidate by about 4 points in every set. On the final set, the plain isotropic scaling (SCALE, 0.656) is better than SELECT (0.637). Against the best simple baseline (NA, MC or STD) SELECT is 1 to 3 points behind on average, ahead by more than 2 points in only 2 to 4 cells.
- **Selection frequency.** CORAL is chosen in 22% to 35% of replicates and HATC in about 1% to 11%, even though CORAL and HATC are among the weakest candidates in accuracy. The healthy transport error favours the most flexible alignments, because they fit a healthy distribution almost exactly. This is a bias in the selection criterion, and it is the main reason SELECT does not reach the best fixed candidate.
- **The certificate does not work in the conventional direction.** The healthy error has a mean AUROC against failure of 0.34 to 0.42, below chance, so a high error is not associated with failure. (Reversed it would score 0.58 to 0.66, similar to the agreement bound's 0.61 in module 30.) Both label-free certificates are weak.

## Reading

- **The extension helps, and it is honest to say why.** The registered HATC is hampered by the full affine map. A simpler or shrunk map is better, and healthy-validated selection recovers part of that gain automatically. It does not reach a well-chosen fixed simple alignment.
- **What this does not show.** It does not show that healthy-anchored selection is better than choosing a simple alignment in advance. In these worlds, an isotropic scaling plus a mean shift is the best single choice, and no healthy-anchored mechanism is needed to find it.
- **For the paper.** The honest contribution is the benchmark and the negative result for the full map, plus the observation that regularisation matters. HATC-S is a reasonable engineering fix, not a strong novelty claim.

## Limits

- Post-hoc design, tuned on seeds 2026 and 9001. Only seed 9002 is a clean one-shot test.
- The worlds' operators are mine, and simple alignments are well suited to them. This result may not carry to real crops.
- Single source only. 7 crops per cell; intervals (module 36) are not computed for HATC-S.
- The parsimony tolerance (0.05) and the candidate set are design choices not validated beyond these runs.

**How to re-run:**
```bash
python features/50_hatc_select.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/50_hatc_select/seed9002 --scale 2 --world-seed 9002 --tag seed9002
```
Iteration 1 used six candidates and no parsimony rule; iteration 2 is the version now in the repository. Iteration 1 outputs (first pass, unharmonised) are in `results/_first_pass_unharmonised/50_hatc_select/` (`original`, `fresh`, `v2_original`, `v2_final`). The harmonised rerun of iteration 2 is in `results/50_hatc_select/seed2026` (original world seed) and `seed9002` (fresh world seed); `analysis/rerun_all_harmonised.ps1` produces both.
