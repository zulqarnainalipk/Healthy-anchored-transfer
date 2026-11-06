# Results folders

One folder per module, named `NN_module_name`, as in `docs/implementation_checklist.md`. Each holds the module's result tables and (for most) a `manifest.json` with inputs, versions, row counts and run time. Some folders have no manifest at their top level: they keep it in a subfolder (for example per-run folders), or use another summary file (the audits `code_audit_result` and `local_audit`, `03_background_check`, `04_near_duplicates`, `04b_verify`, `fig_augmentation_check`, the merged tables of modules 26 and 30, `46_tables`, and the verification modules 54 to 58, which write their own summary tables). The manifest, where it exists, lists inputs, versions, row counts and run time.

| Folder | Content |
|---|---|
| `code_audit_result/`, `local_audit/` | dataset audit (remote run and local run; `inventory.csv` has identical content in both, rows in a different order) |
| `01_` to `04b_` | label table, pair gate, splits, background check, near-duplicate check |
| `05_features_dinov2/`, `06_features_bioclip/` | feature manifests; the `.npy` arrays and index copies are not in git (see `docs/README.md`) |
| `08_`, `08b_`, `09_` to `13_` | projection, sweep, Gaussianity, host maps, push-forward |
| `16_worlds/` to `50_hatc_select/` | synthetic benchmark and its statistics; **final (harmonised) results** |
| `03b_harmonised/` | sensitivity run of modules 20, 26, 36, 37, 46b made when label harmonisation was first tested (`docs/modules/03b_label_harmonisation.md`). Its counts differ slightly from the final rerun (for example 5 instead of 6 cells where HATC is worse in module 37); the complete harmonised rerun is the one in the numbered folders. The `26_linear_probes/g1` and `g2` folders here hold no files and are not part of a clone |
| `_first_pass_unharmonised/` | the first-pass results (before label harmonisation), kept for the record. Their module documents carry a banner |
| `29_second_encoder/` | benchmark on BioCLIP features |
| `17_calibration/`, `17_calibration_v2/` | operator-strength calibration runs (strength 2 was chosen, see `docs/modules/17_benchmark_hatc.md`); the folders are not described individually in a module document |
| `fig_augmentation_check/` | check whether the augmented fig images are copies of the Original images (audit item D3) (`docs/modules/fig_augmentation_check.md`) |
| `46_tables/` | tables T1 to T8 (csv and markdown) |
| `09_features_plantwild/`, `10_features_plantwild_bioclip/` | PlantWild feature manifests (DINOv2 and BioCLIP); arrays and index copies are not in git |
| `plantwild_audit/` | PlantWild audit: class table, duplicate pairs, `dedupe_exclude.csv` (module 08b) |
| `53_real_pair_test/` | **primary analysis**: real source-target-disease units; one subfolder per run (`pw_pca` is the main run; `_seed1`, `_seed2`, `_n50`, `_lda`, `_bioclip_pca`, `_wrongdisease`, `_t3`, `_strictdup`, `_domainmatch`, `_3class`; `pilot_*` are PlantVillage pilots) |
| `54_real_agreement/` to `58_strict_duplicates/` | agreement check on real pairs, independent re-implementation, primary-outcome checks, feature spot check, strict duplicate list (`docs/modules/55_to_59_verification.md`) |
| `51_review_checks/`, `52_persample_refusal/` | design checks of the synthetic benchmark (projection fairness, shrinkage, per-sample refusal) |
| `*.log` at this level | stdout logs of each run (kept); `*.err`, `*.err.log` and `_rerun_*` are not committed |

Do not edit result files by hand. To regenerate them see the "How to re-run" command in each module document.

Post-hoc checks in the second round of post-hoc checks (registration section 9): `60_background_control/`, `61_calibration_decision_rule/`, `62_sensitivity_summary/`, and the `sens/` subfolder of `53_real_pair_test/` (covariance estimators, HATC-S, frame-only features).

Post-hoc checks in the third round of post-hoc checks (registration section 10): `63_review4_summary/` (learning curve, mean/covariance split, EIG and unlabelled-target arms), `64_risk_noise_release/` (label-free risk scores, label-noise proxy, `replication/` lists and projections), and the `curve/` and `sens/extra_*` subfolders of `53_real_pair_test/`. The `_run_*.log` files at this level are stdout logs of those runs.
