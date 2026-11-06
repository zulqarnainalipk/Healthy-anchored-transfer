# Module 04: near-duplicate check (proposal 7.9)

**Compute:** CPU. Uses the cached DINOv2 features from module 05. Runs locally in about 90 seconds.

**Question.** Does any target image (rice, fig Original, chili, peach) look like a copy of a source image (PlantSeg or PlantVillage)? A copy would inflate the evaluation.

**Method.** Similarity is the cosine of the class-token features (the first 768 values, L2-normalised). For each target image, the most similar source image is found.

**Results** (`results/04_near_duplicates/`):

| Target | Images | Highest similarity | At or above 0.90 | At or above 0.95 |
|---|---|---|---|---|
| rice_leaf_disease | 6,712 | 0.888 | 0 | 0 |
| fig_sartak_sfld (Original) | 2,037 | 0.796 | 0 | 0 |
| chili_yellow_leaf | 6,566 | 0.775 | 0 | 0 |
| peach_leaf_damage | 182 | 0.872 | 0 | 0 |

**Closest pairs.** The twelve closest pairs (all rice) score 0.87 to 0.89. Their labels show different diseases, for example rice brown spot against wheat septoria blotch, so they are coarse similarities, not copies.

**Verification (module 04b, `docs/modules/04b_near_duplicate_verify.md`).** The top 200 candidate pairs were checked at pixel level, which does not depend on the encoder. Each target was compared with its source under flips and 90-degree rotations, on 64 × 64 grayscale images. Result: **0 of 200 are pixel copies**. The highest pixel correlation is 0.80, far below the 0.9 copy threshold. The top 12 pairs were also shown side by side to the team.

**Threshold: 0.90 (proposed for registration).** At 0.90 nothing is removed by the embedding check, and the pixel check confirms that the closest pairs are not copies. The proposal's hand-checked sample requirement is met by the 12-pair sheet and the 200-pair pixel check. The team must confirm this before registration.

**Exact duplicates across datasets** were already checked by file hash in the audit: none is shared between a target and a source.

**Open point, important for the registration (proposal 7.11).** The proposal says the registration is fixed before any target feature is extracted. Target features were extracted in module 05 before the registration. The source-only sweeps (modules 08 and 09) remain valid only if they use source features alone, and no target result influences any choice. This timing deviation must be stated in the registration and in the paper.

**How to re-run:**
```bash
python features/04_near_duplicates.py --features results/05_features_dinov2/remote/features_dinov2 --output results/04_near_duplicates
```
