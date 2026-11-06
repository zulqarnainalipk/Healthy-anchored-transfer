# Modules 06 and 29: second-encoder sensitivity check (BioCLIP)

**Run:** features in the remote environment (about 110 minutes, 81,261 images, no missing or non-finite values); everything else locally (projection 30 seconds, main benchmark 4 minutes, probes 17 minutes in four parallel groups, statistics about 10 minutes).

**Question.** Do the conclusions of the DINOv2 benchmark depend on the encoder? (Proposal 7.7: second-encoder sensitivity check.)

**Method.**
- **Features (module 06):** BioCLIP (ViT-B/16, `imageomics/bioclip`, trained on biological images), 512-d image embedding, same image index and the same 224 × 224 full-frame resize as module 05, float16. Script: `features/06_features_bioclip.py` (see `06_features_bioclip.md`); output `results/06_features_bioclip/remote/features_bioclip/` with `manifest.json` (checksums per dataset).
- **Projection:** the same shrinkage LDA fitted on the source crops only (module 08), 16 dimensions. Validation accuracy 0.444, 0.709, 0.841 at 4, 8, 16 dimensions (DINOv2: 0.507, 0.723, 0.805).
- **Benchmark:** modules 20 (alignments), 36 and 37 (intervals, equivalence), 26 and 31 (probe and labelled saving), unchanged code, harmonised classes, operator strength 2, healthy n = 50 and 200. Results in `results/29_second_encoder/`.

## Results

**Main benchmark, healthy n = 200 (mean accuracy):**

| World | NA | MC | STD | CORAL | HATC | Best |
|---|---|---|---|---|---|---|
| a | 0.686 | 0.767 | 0.769 | 0.749 | 0.763 | STD |
| a2, d | 0.813 | 0.841 | 0.832 | 0.786 | 0.796 | MC |
| b | 0.759 | 0.762 | 0.832 | 0.786 | 0.796 | STD |
| b2 | 0.774 | 0.773 | 0.768 | 0.743 | 0.763 | NA |
| c25 | 0.756 | 0.773 | 0.783 | 0.754 | 0.770 | STD |
| c50 | 0.736 | 0.754 | 0.784 | 0.740 | 0.763 | STD |
| c100 | 0.689 | 0.689 | 0.774 | 0.717 | 0.733 | STD |
| e | 0.281 | 0.266 | 0.285 | 0.318 | 0.313 | CORAL |
| f | 0.828 | 0.841 | 0.832 | 0.786 | 0.796 | MC |

- **Same ordering as DINOv2.** Per-dimension standardisation or mean-centring is the best method in 17 of 20 cells (STD 9, MC 8), no adaptation in 1, CORAL in 2, and **HATC in none** (DINOv2: HATC best in 2).
- **HATC against the best simple method** (module 37, exploratory, 7 clusters): at n = 200 worse in 3 cells (a2, d, f), inconclusive in 7, none better; at n = 50 worse in 8, inconclusive in 2. Mean differences at n = 200 range from −4.5 to +2.7 points (world e: +2.7, interval −0.4 to +7.3 points).
- **Labelled saving:** a probe with 5 to 10 labelled images per class reaches HATC in all 20 cells: 5 labels per class in 16 cells and 10 in 4 cells (at n = 200: 5 labels in 6 cells, 10 in 4 cells; the 10-label cells are the pure-shift and scaling worlds, where HATC is relatively strong).
- Absolute accuracies are higher than with DINOv2 (about 0.69 to 0.84 against 0.62 to 0.73 outside world e) because the same operator strength is a milder shift in this projection.

## Reading

- The central conclusion survives a change of encoder: healthy-anchored alignment (HATC) does not beat simple alignments, and a handful of labels per class beat it.
- With BioCLIP the gap between HATC and the best simple method at n = 200 is 0.6 to 4.5 points below in nine worlds and 2.7 points above in world e, against 0.7 to 6.4 points below in nine worlds and 3.3 above in world e with DINOv2; so the gap is slightly smaller here, not qualitatively different.

## Limits

- The operator strength was **not recalibrated** for this encoder (the registered rule would choose a stronger operator, since no-adaptation accuracy is 0.69 to 0.83, above the 0.40 to 0.65 target band); absolute accuracies are not comparable between encoders.
- One second encoder only; same projection method; synthetic worlds only; no encoder fine-tuning (module 28 not run).
- The 0-label probe value (0.2895) is identical for both encoders: it is a feature-independent reference level, not an encoder result.

**How to re-run:**
```bash
python features/08_projection.py --features results/06_features_bioclip/remote/features_bioclip --output results/29_second_encoder/08_projection
python features/20_align_baselines.py --features results/06_features_bioclip/remote/features_bioclip --labelled results/01_crop_disease_table/labelled_images.csv --projection results/29_second_encoder/08_projection/projection_k16.npz --output results/29_second_encoder/20_align_baselines --scale 2
python features/26_linear_probes.py --features results/06_features_bioclip/remote/features_bioclip --labelled results/01_crop_disease_table/labelled_images.csv --projection results/29_second_encoder/08_projection/projection_k16.npz --output results/29_second_encoder/26_linear_probes/g1 --worlds a a2 b
python features/31_labelled_saving.py --hatc results/29_second_encoder/20_align_baselines/replicate_results.csv --probes results/29_second_encoder/26_linear_probes/probe_curve.csv --output results/29_second_encoder/31_labelled_saving
```
(Run the probe for the other world groups g2 to g4 and concatenate the group curves into `probe_curve.csv` as in `analysis/merge_groups.py`.)
