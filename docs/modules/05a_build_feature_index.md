# Module 05a: feature index

**Compute:** CPU, 5 seconds locally.

**Question.** Modules 05 and 06 read an index of the images to encode. This module rebuilds that index from stored tables, so the index can be checked without the images.

**Method.** Join `results/02b_splits/splits.csv` (dataset, path, split) with `results/01_crop_disease_table/labelled_images.csv` (class = disease, healthy flag, eligibility), keep eligible rows in the order of `splits.csv`.

**Result.** 81,261 rows (PlantVillage 54,306; PlantSeg 11,458; rice 6,712; chili 6,566; fig originals 2,037; peach 182). The rebuilt file is byte-identical to the index used for the remote feature runs (SHA-256 `5026e5b5ba77...`). The module was written after the remote runs, as a record of how the index is obtained (no script for it existed before).

**Note.** `feature_index.csv` (12.8 MB) is excluded from git (see `.gitignore`); run this module to recreate it.

**How to re-run:**
```bash
python data_prep/05a_build_feature_index.py --splits results/02b_splits/splits.csv --labelled results/01_crop_disease_table/labelled_images.csv --output results/05_features_dinov2
```
