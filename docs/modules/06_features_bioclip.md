# Module 06: cached BioCLIP features (second encoder)

The method, result and limits of the BioCLIP check are in `29_second_encoder.md` (modules 06 and 29 are documented together). This page records how to run module 06 itself.

**Run:** remote environment, one run of about 110 minutes (81,261 images, no missing or non-finite values). Output: 512 values per image, float16, one `.npy` per dataset, plus `features_index.csv` and `manifest.json` with SHA-256 checksums (`results/06_features_bioclip/remote/features_bioclip/`).

**Code.** `features/06_features_bioclip.py`. The run used the same logic with the input and output folders fixed inside the script; the committed version takes them as arguments. Requires `open_clip_torch` (version 3.3.0 recorded in the manifest, see `requirements-gpu.txt`).

**How to re-run (needs the datasets; slow):**
```bash
python data_prep/05a_build_feature_index.py --splits results/02b_splits/splits.csv --labelled results/01_crop_disease_table/labelled_images.csv --output results/05_features_dinov2
python features/06_features_bioclip.py --dataset-root <data-root> --index results/05_features_dinov2/feature_index.csv --output results/06_features_bioclip/remote/features_bioclip
```
The cached features of the original run are published as an archive (see `docs/README.md`); a rerun is only needed to confirm them.
