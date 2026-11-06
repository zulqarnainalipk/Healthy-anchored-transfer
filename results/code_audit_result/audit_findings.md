# Audit findings

Run: the dataset audit (`data_audit_code/audit_dataset.py`), completed on the full attached dataset. Log: `plant-data-audit.log`. All tables are in this folder.

## 0. Local confirmation

The full audit was also run locally on the complete dataset folder, with file hashing on (`results/local_audit/`). Its tables match the remote run: the same six datasets, the same image counts, the same 455 exact-duplicate groups, and no unreadable images. The one difference is the rice background sample (84 white against 29 in the remote run). Background samples are the first 200 images in file order, so they depend on the order files are read; they are a sample, not a count.

## 1. Totals

| Item | Value |
|---|---|
| Files | 100,941 |
| Images | 100,867 (including 11,458 PlantSeg masks) |
| Size | 9.17 GB |
| Unreadable images | 0 |
| Groups of exact duplicate files | 455 (`exact_duplicate_groups.csv`) |

## 2. Per dataset (`dataset_summary.csv`)

| Dataset | Files | Images | GB | Class folders | Augmented images | Licence or readme in folder |
|---|---|---|---|---|---|---|
| chili_yellow_leaf | 6,566 | 6,566 | 0.16 | 1 | 0 | no |
| fig_sartak_sfld | 10,189 | 10,185 | 4.17 | 4 | 8,148 | yes |
| peach_leaf_damage | 184 | 182 | 0.01 | 4 | 0 | yes |
| plantseg_v6 | 22,920 | 22,916 | 1.78 | 3 (train, val, test) | 0 | no |
| plantvillage_color | 54,370 | 54,306 | 0.91 | 39 | 0 | yes |
| rice_leaf_disease | 6,712 | 6,712 | 2.14 | 10 | 0 | no |

## 3. Problems found, and how they were resolved

**P1. PlantSeg contains lesion masks.** The 22,916 files are 11,458 JPG photos and 11,458 PNG masks. *Resolved:* module 01 excludes the masks (`docs/modules/01_crop_disease_table.md`).

**P2. Labels are in file names for two datasets.** Chili's labels are in its file names (3,283 `Health_Leaf`, 3,283 `Yellow_Leaf`). PlantSeg's disease name is in its file names, and its folders are only splits. *Resolved:* module 01 reads chili from file names (3,283 healthy and 3,283 infected) and PlantSeg from its `Metadata.csv` (34 plants, 114 diseases).

**P3. Fig augmented copies were counted.** Of 10,185 images, 8,148 are augmented and 2,037 are Original. *Resolved:* module 01 evaluates `SFLD/Original` only. Result: 513 healthy and 1,524 disease.

**P4. Peach background is almost all black.** 175 of 182 images have a black background. *Checked (module 03, `docs/modules/03_background_check.md`):* the background does not predict healthy versus damage. Shortcut accuracy is 54%, against a majority rate of 59%. So peach's black background carries no class information, but the uniform background still needs stating in the paper.

**P5. Chili background separates the classes.** *Checked (module 03):* mean corner brightness is 52 for healthy and 167 for yellow leaf. A single brightness threshold predicts healthy versus infected with 98% accuracy, against a 50% chance rate. So the chili binary target can be solved from the background alone. This is a serious shortcut. Any chili result must use a background-controlled input (for example a central crop or background masking) and report the shortcut accuracy beside it.

**P6. Rice contains a pest class.** `rice_hispa` is present (770 images). *Open:* the proposal reports rice with and without it.

**P7. Minor file issues.** PlantVillage has an odd file named `email (1)` and a few non-image files, which are excluded from image analysis. One PlantVillage image is palette mode and one is RGBA; both are converted to RGB before use. Rice has one palette-mode image.

**P8. PlantSeg has no healthy images in any release.** The proposal (section 4) lists 8,000 healthy PlantSeg images. Checked from each release's zip index (no full download):
- v6 (record 16665770, 1.57 GB): 22,916 images (11,458 photos and 11,458 masks), none named healthy.
- 2024 record (13762907, 2.83 GB, CC BY-NC-ND): 22,916 images, none named healthy.
- v7 (record 17719108, 0.98 GB, CC BY-NC): 15,548 images, none named healthy.

Decision: the project uses v6. The `Metadata.csv` for v6 (11,458 rows, 34 plants, 114 diseases) has no healthy entries either, so the 8,000 figure in the proposal is not supported. Its licence column is mixed per image: 10,325 CC0 and 1,133 CC-BY, which must be stated in the release plan. The proposal must be corrected, or healthy images for PlantSeg crops must come from another source, before module 09 registers the healthy thresholds.

## 4. Duplicates (`duplicate_summary.csv`)

- No exact duplicate is shared between a target dataset and a source dataset. Target groups: chili 8, fig 29, rice 1, peach 0. These are duplicates within a dataset.
- Two groups are shared between PlantSeg and PlantVillage (sources only).
- The near-duplicate check (proposal 7.9) is still required, because it finds near matches, not only exact ones.

## 5. Image properties (`resolution_summary.csv`)

- Chili: all 224 × 224 RGB. No resizing before caching at 224.
- Fig: all 512 × 512 RGB.
- Rice: width 183 to 4,000 px, median 1,600. Resizing to 224 is required.
- PlantVillage: all 256 × 256 (mostly RGB).
- PlantSeg: width 65 to 8,256 px; greyscale masks and RGB photos.
- Peach: width 76 to 1,953 px, median 535.

## 6. Checklist status

| Item | Status | Note |
|---|---|---|
| A1 dataset_summary | done | |
| A2 folder_tree | done | |
| A3 file_types | done | |
| A4 inventory | done | |
| A5 corrupt_files | done | 0 unreadable |
| A6 licence or readme | done | Checked on the record pages (step 4). Chili CC BY 4.0 (Zenodo record); peach CC BY 4.0 (Zenodo record); rice CC BY 4.0 (dataset page; its stated total of 6,758 is a typo, the category counts sum to 6,712); fig CC BY 4.0 (dataset page); PlantSeg v6 CC BY 4.0 (record), with per-image licences 10,325 CC0 and 1,133 CC-BY; PlantVillage: the record metadata says CC BY 4.0 but its text says CC BY-SA 3.0 (US). This conflict is open and must be resolved with the uploader before release |
| B1 class names | done | after module 01 (P2) |
| B2 class balance | done | after module 01 |
| B3 healthy and disease counts | done, with P8 open | PlantSeg has no healthy class (P8) |
| B4 chili binary | done | 3,283 and 3,283 |
| B5 peach is damage | done | four damage folders; `healthy` is one (107) |
| B6 fig Original and augmented | done | 8,148 augmented, 2,037 Original |
| C1 resolution distribution | done | |
| C2 colour mode | done | |
| C3 background type | done on a sample of 200 per dataset | peach and chili need the checks in P4 and P5 |
| D1 exact duplicates | done | |
| D2 duplicate counts per dataset | done | |
| D3 fig augmented copies of Original | done | `data_audit_code/fig_augmentation_check.py`: 8,104 of 8,148 augmented images match their Original (correlation at least 0.9, under rotation, flip and brightness change). Matches by class: Anthracnose 1,786/1,792, Cercospora 2,089/2,100, Healthy 2,044/2,052, Mosaic 2,185/2,204. The 44 unmatched score 0.78 to 0.87 and are most likely rotated copies. Evaluation uses only the 2,037 Original images, which is confirmed |
| E1 plant IDs in file names | partial | photo numbers only; no plant IDs for chili, rice or fig |
| E2 photo-level grouping | partial | fig `Mosaic_#_...` groups by source photo; others not yet checked |
| F1 crops and diseases | done | module 01, `crop_disease_table.csv` |
| F2 gate (pairs per disease) | done | module 02: 0 qualifying under exact names; 2 target-diseases (7 triples) under the crop-stripped rule. Both below 25. See `docs/modules/02_pair_gate.md` |
| F3 healthy count against 50 and 200 | partial | target-crop counts available; PlantSeg (P8) open |
| G1 rice audit | done | rice is in the attached dataset and is audited above |
| G2 rice acquisition setting | not done | from the rice documentation |

## 7. Next steps

1. Resolve P8: confirm where PlantSeg's healthy images are, or correct the proposal.
2. Run module 02 (pair gate) on the labelled table.
3. Run the background check on the photo subset (P4 and P5).
4. Check the per-copy fig augmentation (D3) and plant-level grouping (E2).
5. Confirm the licences on each record page (A6).
