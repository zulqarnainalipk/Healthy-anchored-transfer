# Documentation index

Author and contact: Zulqarnain Ali, Department of Information and Communication Engineering, Yeungnam University, 280 Daehak-ro, Gyeongsan 38541, Gyeongbuk, Republic of Korea; zulqar445ali@gmail.com; ORCID 0009-0009-0281-882X. Citation metadata: `../CITATION.cff`. Licence: pending (`../LICENSE.proposed`). PlantWild data (CC BY-NC-ND 4.0) are not redistributed.

## Start here

| Document | What it is |
|---|---|
| `../README.md` | purpose, data policy, environment, how to reproduce the key results |
| `real_pair_results.md` | **primary analysis**: real crop pairs from PlantWild v1 (HATC against simple alignments, controls, verification, agreement check). Every number of the paper's real-pair sections |
| `harmonised_results.md` | **secondary analysis**: final numbers of the synthetic benchmark (after label harmonisation). Where a synthetic-benchmark module document disagrees, this file is right |
| `registration.md` (+ `registration.sha256`) | the analysis rules fixed before the target data were used; the hash records that the file was frozen |
| `data_sources.md` | the seven datasets (six crop datasets and PlantWild v1): where to get them, licences, what the repository does and does not redistribute |
| `audit_checklist.md` | what the dataset audit establishes |
| `implementation_checklist.md` | working rules, module list with compute type, status table |

## Module documents (`modules/`)

One document per module, each with a "How to re-run" command. Documents with the banner "Numbers in this document are from the first pass" describe the run before label harmonisation (module 03b); the folder `results/_first_pass_unharmonised/` holds those results.

| Phase | Modules |
|---|---|
| Audit and preparation | `fig_augmentation_check`, `01_crop_disease_table`, `02_pair_gate`, `02b_splits`, `03_background_check`, `03b_label_harmonisation`, `05a_build_feature_index`, `08b_plantwild_audit` |
| Features | `04_near_duplicates`, `04b_near_duplicate_verify`, `05_features_dinov2`, `06_features_bioclip` (second-encoder results: `29_second_encoder`), `09_features_plantwild` (modules 09 and 10: PlantWild, DINOv2 and BioCLIP) |
| **Real-pair test (primary)** | `53_real_pair_test`, `54_real_agreement`, `55_to_59_verification` (55 independent check, 56 primary checks, 57 feature spot check, 58 stricter duplicates, 59 figures R7 and R8) |
| Projection, sweeps, maps (synthetic) | `08_projection` (also covers `features/08b_projection_pca.py`), `09_sweep_healthy_n`, `10_gaussianity`, `11_host_maps_affine`, `12_host_maps_brenier`, `13_push_forward` |
| Synthetic benchmark (secondary) | `16_world_generator`, `17_benchmark_hatc`, `18_agreement_benchmark`, `20_align_baselines`, `26_linear_probes`, `30_certificate_baselines`, `31_labelled_saving`, `35_label_noise`, `43_ablation_engine`, `44_ablations_projection`, `46b_metrics_cost`, `46d_failure_cases`, `50_hatc_select`, `51_review_checks` (`features/51_loco_projection.py`), `52_persample_refusal` |
| Real-data probes | `34_chili_peach_probes` |
| Statistics | `36_cluster_bootstrap`, `36b_bootstrap_coverage_power`, `37_equivalence`, `38_exact_test` |
| Device components | `39_d1_enrolment`, `40_d2_routing`, `41_d3_drift_refit`, `42_d4_sequential` |
| Post-hoc checks (exploratory) | `60_background_control`, `61_calibration_decision_rule`, `62_sensitivity_summary`, `63_review4_summary` (module name; third round of post-hoc checks), `64_risk_noise_release` |
| Tables, figures, rerun | `46_47_tables_figures`, `48_pipeline_figure` (`analysis/48_` and `48b_pipeline_panels*.py`), `rerun_harmonised` (`analysis/rerun_all_harmonised.ps1`, `analysis/merge_groups.py`) |

## What is in git and what is not (large-file policy)

The repository is about 105 MB as pushed (566 MB on disk before exclusion; measured 6 November 2026). `.gitignore` excludes the rest. No single tracked file exceeds 25 MB (GitHub's hard limit is 100 MB per file; a warning appears above 50 MB).

| Kept in git | Size | Why |
|---|---|---|
| code, docs, `requirements*.txt` | under 1 MB | |
| small result tables, `manifest.json` files, summaries, stdout logs, figures | about 17 MB | the evidence for every number in the paper |
| `results/01_crop_disease_table/labelled_images.csv` (19 MB), `results/02b_splits/splits.csv` (15 MB) | 34 MB | label and split tables that every module reads |
| `results/code_audit_result/inventory.csv`, `results/local_audit/inventory.csv` (24 MB each, identical content, different row order) | 48 MB | file inventory of the audit; one of the two could be dropped |

| Excluded from git | Size | Where it goes |
|---|---|---|
| `features_*.npy` (DINOv2 and BioCLIP on the six crop datasets, DINOv2 and BioCLIP on PlantWild; 14 files; largest 159 MB, over GitHub's 100 MB limit) | 404 MB | release archive on a data repository (archive host and DOI still to be decided; none is assigned yet); SHA-256 of each file is in the `manifest.json` that stays in git |
| `features_index.csv` (two copies) and `feature_index.csv` (the two PlantWild index files, 1.3 MB each, stay in git) | 41 MB | `feature_index.csv` is rebuilt by `data_prep/05a_build_feature_index.py`; the other two are written by the feature modules |
| `*.err`, `*.err.log` (warning streams with absolute local paths), raw logs of the remote runs, `results/local_audit/audit.log` (local data path) | 11 MB | not needed to reproduce anything |
| `reference/plantseg_v6/Metadata.csv` | 2.4 MB | third-party file; download from the PlantSeg record (`data_sources.md`) |

To run later modules without recomputing features, download the archive and unpack it so that the files sit in `results/05_features_dinov2/remote/features_dinov2/` and `results/06_features_bioclip/remote/features_bioclip/`; check them against the SHA-256 values in the manifests.

## Third-party material in the repository

- Figure 1 uses ten reduced-size leaf photographs (tomato from PlantVillage, CC BY-SA 3.0; fig from the Sartak Fig Leaf Dataset, CC BY 4.0) and twelve Lucide icons (ISC licence), see `data_sources.md`.

Module numbers are not unique in two places: `08b` is both the PlantWild audit (`data_audit_code/audit_plantwild.py`) and the PCA projection (`features/08b_projection_pca.py`), `09` and `10` name both the synthetic-benchmark modules (healthy-n sweep, Gaussianity) and the PlantWild feature extraction (`09_features_plantwild`, `10_features_plantwild_bioclip`), and `48` and `48b` share one document. The file name and the result folder name always disambiguate.
