# Module 02b: splits grouped by photo, duplicate or mosaic number

**Compute:** CPU, metadata and hashes only. Runs locally in seconds.

**Question.** Assign every eligible image to train, validation, test or probe so that no group (source photo, exact duplicate, or mosaic) spans two splits.

**Rules** (documented in the script header and in the registration):
- PlantSeg: the official split column of the PlantSeg metadata; each photo is its own group.
- PlantVillage, rice, chili: group = exact duplicate (MD5); groups assigned 80/10/10 with seed 2026. No plant identifiers exist, so no plant-level claim is made.
- Fig: Original folder only; group = mosaic number (a source photo and its augmented copies share it); 70/15/15 by group.
- Peach: a probe set, never trained on; all rows are `probe`.
- PlantSeg masks and fig augmented copies are not eligible.

**Output** (`results/02b_splits/`): `splits.csv` (81,261 rows: path, dataset, group, split), `split_summary.csv`, `manifest.json` (rule version `02b-1`, seed, SHA-256 of `splits.csv`, leakage check: passed, no group spans two splits).

**Inputs.** `results/local_audit/inventory.csv`, `results/01_crop_disease_table/labelled_images.csv`, and the PlantSeg `Metadata.csv` (not stored in the repository by default; download it from the PlantSeg record, see `docs/data_sources.md`).

**Limit.** The split is by image group, not by plant, because the datasets carry no plant identifiers.

**How to re-run:**
```bash
python data_prep/02b_splits.py --inventory results/local_audit/inventory.csv --labelled results/01_crop_disease_table/labelled_images.csv --plantseg-metadata reference/plantseg_v6/Metadata.csv --output results/02b_splits
```
