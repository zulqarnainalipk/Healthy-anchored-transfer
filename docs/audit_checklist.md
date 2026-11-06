# Data audit checklist

Purpose: confirm what each source and target dataset contains before any method code is written. Each item below has a check and a result file. An item is ticked only when its result file has been pulled from the remote compute environment and read.

The audit runs in the remote compute environment against the dataset folder (about 9 GB). The dataset is never downloaded locally.

## A. Structure and inventory

| # | Check | Result file | Done |
|---|---|---|---|
| A1 | Top-level folders, one per dataset, and their file counts | `dataset_summary.csv` | [ ] |
| A2 | Folder tree to depth 4, to confirm nesting and naming | `folder_tree.txt` | [ ] |
| A3 | File types and counts per dataset (jpg, png, txt, csv, json, licence files) | `file_types.csv` | [ ] |
| A4 | Every image listed with path, size, dimensions and colour mode | `inventory.csv` | [ ] |
| A5 | Unreadable or corrupt images | `corrupt_files.csv` | [ ] |
| A6 | Licence or readme files present in each folder | `dataset_summary.csv` (column `has_licence_or_readme`) | [ ] |

## B. Labels and class structure

| # | Check | Result file | Done |
|---|---|---|---|
| B1 | Class folder names or label files per dataset | `class_counts.csv` | [ ] |
| B2 | Images per class and class balance | `class_counts.csv` | [ ] |
| B3 | Healthy vs diseased counts per dataset (needed for the 50 and 200 healthy thresholds) | `healthy_disease_counts.csv` | [ ] |
| B4 | Chili: binary labels only (healthy vs infected), confirmed | `class_counts.csv` | [ ] |
| B5 | Peach: foliar damage labels, not disease, confirmed | `class_counts.csv` | [ ] |
| B6 | Fig: Original and augmented folders, and which folder holds augmented copies | `inventory.csv` (column `top_folder`) | [ ] |

## C. Image properties

| # | Check | Result file | Done |
|---|---|---|---|
| C1 | Resolution distribution per dataset | `resolution_summary.csv`, `fig_resolution_hist.png` | [ ] |
| C2 | Colour mode (RGB, RGBA, greyscale) per dataset | `resolution_summary.csv` | [ ] |
| C3 | Background type from corner pixels (black, white, uniform, field) | `background_summary.csv` | [ ] |

## D. Duplicates and leakage risk

| # | Check | Result file | Done |
|---|---|---|---|
| D1 | Exact duplicates by file hash, within and across datasets | `exact_duplicate_groups.csv` | [ ] |
| D2 | Duplicate count per dataset and cross-dataset overlap | `duplicate_summary.csv` | [ ] |
| D3 | Fig augmented copies of Original images (the leakage risk in section 4 of the proposal) | `duplicate_summary.csv` | [ ] |

## E. Plant identity and grouping

| # | Check | Result file | Done |
|---|---|---|---|
| E1 | Whether file names or folders carry plant IDs (needed for plant-level splits) | `id_pattern_summary.csv` | [ ] |
| E2 | Photo-level grouping: files that look like crops of one photo | `id_pattern_summary.csv` | [ ] |

## F. Gate and pair counts (needs the crop-to-disease mapping, filled in after the first audit)

| # | Check | Result file | Done |
|---|---|---|---|
| F1 | Crops and disease names per dataset, from folder names | `class_counts.csv` | [ ] |
| F2 | Diseases labelled on at least two other crops (gate, proposal 7.1) | written after F1 | [ ] |
| F3 | Healthy image count per target crop against the 50 and 200 thresholds | `healthy_disease_counts.csv` | [ ] |

## G. Rice (manual step, not yet in the attached dataset)

| # | Check | Result file | Done |
|---|---|---|---|
| G1 | Rice archive audited the same way once added as version 2 of the dataset | same files as above | [ ] |
| G2 | Rice acquisition setting recorded from its documentation | `docs/data_sources.md` | [ ] |

## Sign-off

Audit complete only when A1 to F3 are ticked and the findings are written in `results/code_audit_result/audit_findings.md`. G items are ticked after the rice upload.

