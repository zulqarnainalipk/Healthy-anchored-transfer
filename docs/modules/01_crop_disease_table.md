# Module 01: crop and disease table

**Compute:** CPU. Reads the audit inventory only (paths and metadata). No image is opened, so it runs locally.

**Inputs:** `results/code_audit_result/inventory.csv` from the dataset audit (module 00).

**Outputs** (in `results/01_crop_disease_table/`):
- `labelled_images.csv`: one row per image or mask, with role, crop, disease, healthy flag, evaluation eligibility, and the rule applied.
- `crop_disease_table.csv`: counts per dataset, crop, disease and healthy flag, over eligible images only.
- `label_rules.md`: each rule with the number of rows it touched.
- `manifest.json`: inputs, row counts, eligible images and masks excluded.

**Rules** (from `results/code_audit_result/audit_findings.md`):
- P1: PlantSeg masks (`annotations/`, PNG) are excluded from every table.
- P2: chili labels are read from file names (binary: healthy or yellow leaf). PlantSeg plant and disease are read from its `Metadata.csv` (one row per photo; the file name is the join key). The file-name fallback is used only for a photo missing from the metadata.
- P3: fig evaluation uses `SFLD/Original` only. Augmented copies are not eligible.
- Rice, peach, and PlantVillage keep their folder labels.

**Result (eligible images):**

| Dataset | Healthy | Disease | Notes |
|---|---|---|---|
| chili_yellow_leaf | 3,283 | 3,283 | binary target |
| fig_sartak_sfld | 513 | 1,524 | Original only (2,037 in total) |
| peach_leaf_damage | 107 | 75 | damage probe, not disease |
| plantseg_v6 | 0 | 11,458 | 114 diseases over 34 plants, from `reference/plantseg_v6/Metadata.csv`; no healthy images in v6 |
| plantvillage_color | 15,084 | 39,222 | 39 folders |
| rice_leaf_disease | 753 | 5,959 | 10 classes |

**Open point:** PlantSeg v6 has no healthy images in this download. The proposal (section 4) lists 8,000 healthy images for PlantSeg. Either the healthy images are in a different PlantSeg release, or the proposal's count is wrong. This must be resolved before the healthy-image counts for source crops (module 09) are registered.

**How to re-run:**
```bash
python data_prep/01_crop_disease_table.py --inventory results/code_audit_result/inventory.csv --output results/01_crop_disease_table
```
