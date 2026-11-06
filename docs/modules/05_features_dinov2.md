# Module 05: cached DINOv2 features

**Run:** remote environment, one run of 2 h 03 min (7,375 s measured). Feature extraction; the slowest step of the secondary analysis.

**Question.** Extract one frozen-encoder feature vector per eligible image, so all later modules run on features only.

**Encoder.** DINOv2 ViT-B/14 (`facebook/dinov2-base`), chosen by the team. The manifest records the revision as "unknown". The current main revision is f9e44c814b77203eaa57a6bdbbd535f21ede1415 (last modified 2024-01-17), which is almost certainly the one used. A pinned re-run would confirm it.

**Input handling.**
- Each image is resized to 224 × 224 without cropping, so no leaf edge is cut off.
- Only eligible images are processed: 81,261 in total. PlantSeg masks are excluded, and fig augmented copies are excluded.

**Output features.** 1,536 values per image, float16: the class token (768) then the mean of the patch tokens (768).

**Results** (`results/05_features_dinov2/remote/features_dinov2/`):

| Dataset | Rows | File |
|---|---|---|
| chili_yellow_leaf | 6,566 | features_chili_yellow_leaf.npy |
| fig_sartak_sfld (Original only) | 2,037 | features_fig_sartak_sfld.npy |
| peach_leaf_damage (probe) | 182 | features_peach_leaf_damage.npy |
| plantseg_v6 | 11,458 | features_plantseg_v6.npy |
| plantvillage_color | 54,306 | features_plantvillage_color.npy |
| rice_leaf_disease | 6,712 | features_rice_leaf_disease.npy |
| **Total** | **81,261** | |

**Checks.**
- The row count of each file matches the index exactly. The file sizes match the row counts.
- No missing values in the first 200 rows of each file (checked).
- Each file has a SHA-256 checksum in the manifest.
- The index `features_index.csv` gives, for each row, the dataset, the path, the split, the class and the healthy flag.

**Open.**
- Record the exact encoder revision in the manifest, and in the registration.
- Check all rows for missing values, not only the first 200 (cheap to do locally).

**How to re-run:** the module code is `features/05_features_dinov2.py`. The remote run takes the dataset folder and the feature index as inputs.

