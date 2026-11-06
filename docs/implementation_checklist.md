# Implementation checklist

> The status table at the end of this file is the authoritative record. The plan tables above it describe the original design; module names in them may differ slightly from the file names.

Purpose: list every code module the proposal needs, the compute each one runs on, what it depends on, and what it produces. A module is ticked only when its output has been pulled back to the repository and read.

## Working rules

1. **One module, one run.** Each module is a separate script with its own run. A failed module is re-run alone; the others are not repeated.
2. **Each module is small and single-purpose.** It reads named inputs and writes named outputs.
3. **Dependencies are explicit.** A module reads only the outputs of the modules listed in its "Depends on" column.
4. **Decisions come from outputs.** A gate (G) is passed only after the output that sets it has been read and written down. The next module waits for the gate.
5. **Every module logs its steps and progress.** Each run prints a step number and progress for any long loop.
6. **The datasets never enter the repository.** The feature-extraction modules ran once in a remote environment against the attached datasets; later modules, and the audit and data-preparation modules, ran locally against a local copy kept outside the repository (see `docs/data_sources.md`). Only small outputs (tables, JSON, figures, reports) are kept in the repository; the cached feature arrays are published as an archive.
7. **Compute class is stated for each module.** Each module is marked **CPU** (cached features or metadata only) or **GPU** (image encoder). The class records what the module needs in principle, not the machine that ran it.
8. **Check what is already running before starting a run.** A run waits if another run writes to the same result folder.
9. **Run where it is cheapest and most reliable.** A short module (expected under 30 minutes) runs **locally**. Encoder runs and long runs may use the **remote compute environment**.
10. **Every run is documented and recorded, wherever it runs.** A run (local or remote) writes its log, its `manifest.json` (inputs, versions, row counts, run time, location) and its results. Its module document in `docs/modules/` is updated with the run details.
11. **The repository must be publishable at any time.** Every module has a code file, a document and a manifest. `README.md` lists every module. No dataset files, tokens, passwords, account names or platform names appear in the repository. Results contain only small files.

## Repository layout for each module

```
<folder>/NN_name.py                 the module (data_audit_code/, data_prep/, features/ or analysis/)
docs/modules/NN_name.md             what it does, inputs, outputs, compute, run time
results/NN_name/                    outputs pulled back (small files only)
    manifest.json                   inputs used, versions, row counts, run time
    <outputs>
```

## Module list

### Phase 0. Audit

| Module | Compute | Inputs | Outputs | Status |
|---|---|---|---|---|
| 00 `audit_dataset` | CPU, about 1.5 h | dataset folder | inventory, counts, duplicates, figures | done |

**G0. Audit gate.** The audit must show all six sources (chili, peach, PlantSeg v6, PlantVillage, fig, rice) with correct per-dataset counts before any module below runs.

### Phase 1. Data preparation

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 01 `crop_disease_table` | CPU | 00 inventory | crop and disease table; healthy and disease counts per crop | 00 |
| 02 `pair_gate` | CPU | 01 table | pair count per disease target (proposal 7.1), before and after harmonisation | 01 |
| 03 `label_harmonisation` | CPU | 01 table | harmonised label map; counts before and after merges | 01 |
| 04 `near_duplicates` | GPU | 05 features | source-target near-duplicate pairs; threshold sweep | 05 |

**G1. Pair gate.** If fewer than 25 qualifying pairs remain (proposal 7.1), the real-pair comparison is reported as a case study. Modules 32 and 33 then run as case studies only.

### Phase 2. Features (one pass, cached)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 05 `features_dinov2` | GPU | dataset images at 224 px | cached DINOv2 ViT-B/14 features per image, with labels and file ids | 00 |
| 06 `features_second_encoder` | GPU, optional | dataset images | CLIP or BioCLIP features (sensitivity check, proposal 7.7) | 00 |
| 07 `zero_label_clip` | GPU class, optional | 06 features | zero-shot CLIP or BioCLIP scores (baseline) | 06 |

Features are written one file per dataset, so a failure in one dataset does not repeat the others.

### Phase 3. Source-only sweeps (fixed before any target data)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 08 `projection` | CPU | 05 features, source crops only | supervised projection (PLS or regularised LDA), 4, 8 and 16 dimensions | 05, 02 |
| 09 `sweep_thresholds` | CPU | 08 projection, source crops only | healthy thresholds 50 and 200; condition threshold; shrinkage grid; bandwidth | 08 |
| 10 `gaussianity` | CPU | 05 features, plant-level split | per-crop Gaussianity result (reported; does not change the rules) | 05 |

**G2. Registration gate.** The values from module 09 are written to the registration (proposal 7.11) before any target feature is used. Modules 11 onward read the registered values only.

### Phase 4. Method core (HATC)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 11 `host_maps_affine` | CPU | 05, 08, 09 | affine Gaussian map per source-target pair | G2 |
| 12 `host_maps_brenier` | CPU | 05, 08, 09 | Brenier map for 200 or more healthy images, with affine extension | G2 |
| 13 `push_forward` | CPU | 11 or 12; source mixtures | pushed disease mixtures on the target; sampled density | 11, 12 |
| 14 `classify_agreement` | CPU | 13 | predictions; held-out discriminator lower bound; withholding decisions | 13 |
| 15 `hypothesis_N` | CPU | 13; target diseased images | discriminator test of hypothesis N, with power per world | 13 |

The method core runs on the synthetic worlds first (Phase 5), and on real pairs only after G1 passes.

### Phase 5. Synthetic benchmark (primary)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 16 `world_generator` | CPU | 05 feature statistics | generator for worlds (a) to (f) | 05 |
| 17 `benchmark_a_b` | CPU | 16, 14 | results for worlds (a), (a′), (b), (b′) | 16, 14 |
| 18 `benchmark_c_d` | CPU | 16, 14 | results for worlds (c), (d); miss rate for (d) | 16, 14 |
| 19 `benchmark_e_f` | CPU | 16, 14 | results for worlds (e), (f) | 16, 14 |

Worlds run as three modules so a failed world does not repeat the others. Each cell uses at least 200 replicates.

### Phase 6. Baselines (healthy-only arm)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 20 `align_baselines` | CPU | 05, 08 | no adaptation; mean-centring; CORAL; whitening; AdaBN; Gaussianisation | 05, 08 |
| 21 `ot_baselines` | CPU | 05, 08 | healthy-only OT map; map alone; MLP and MMD map; affine registration; mean shift | 05, 08 |
| 22 `caster_reimplementation` | CPU | 05, 08 | CASTER transport step, reimplemented; compared with its published summary | 05, 08 |

### Phase 7. Baselines (unlabelled-field arm)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 23 `tent_shot` | CPU | 05 | TENT and SHOT-style pseudo-labelling on frozen features | 05 |
| 24 `mdfan_style` | GPU | 05 | MDFAN-style unlabelled-field adaptation at matched cost | 05 |
| 25 `published_gmm_ot` | CPU | 05 | Montesuma et al. 2024 as published; data used stated | 05 |

### Phase 8. Baselines (labelled-target arm)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 26 `linear_probes` | CPU | 05 | few-shot linear probe on the label grid (0, 5, 10, 25, 50, 100, 200) | 05 |
| 27 `pooled_source` | CPU | 05 | pooled source plus target supervised baseline | 05 |
| 28 `finetune_lora` | GPU | dataset images | target-only fine-tune (LoRA) on the label grid; secondary reference | 00 |
| 29 `probe_stronger_encoder` | CPU | 06 | stronger-encoder linear probe (CLIP or BioCLIP) | 06 |

### Phase 9. Certificate baselines and labelled-data saving

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 30 `certificate_baselines` | CPU | 14 | ensemble disagreement; disagreement discrepancy; agreement on the line; confidence; calibration; OOD scores | 14 |
| 31 `labelled_saving` | CPU | 26, 14 | labelled-saving grid; reaches rule; censoring count | 26, 14 |

### Phase 10. Real pairs (primary only if G1 passes)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 32 `real_pairs_rice` | CPU | 13, 14, 15 | real-pair results with rice as target | G1 |
| 33 `real_pairs_fig` | CPU | 13, 14, 15 | real-pair results with fig as target | G1 |
| 34 `chili_peach_probes` | CPU | 05 | chili binary metrics; peach probe; background shortcut check | 05 |

### Phase 11. Robustness and statistics

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 35 `label_noise` | CPU | 14, 05 | 5% and 10% symmetric label noise on source labels | 14 |
| 36 `cluster_bootstrap` | CPU | 32, 33, 20 to 27 | two-way cluster bootstrap intervals; 10,000 resamples | 32, 33 |
| 37 `tost_holm` | CPU | 36 | equivalent, different and inconclusive outcomes; Holm correction | 36 |
| 38 `exact_test_descriptive` | CPU | 36 | exact test, descriptive only | 36 |

### Phase 12. Device workflow (synthetic)

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 39 `d1_enrolment` | CPU | 11, 12 | enrolment gate: false enrolment rate; rollback success | 11, 12 |
| 40 `d2_routing` | CPU | 14, 13 | routing: share resolved per route; accuracy at matched coverage | 14, 13 |
| 41 `d3_drift_refit` | CPU | 13 | drift statistic; refits per crop; accuracy drift | 13 |
| 42 `d4_sequential` | CPU | 14 | sequential check: captures to decision; false pass on mismatched crops | 14 |

### Phase 13. Ablations

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 43 `ablations_maps` | CPU | 11, 12, 13 | affine vs Brenier; sample sweep; extension rule; smoothing | 11, 12, 13 |
| 44 `ablations_rules` | CPU | 14, 08 | supervised vs PCA; dimensions; single source; mixture; leave-one-source-out; threshold | 14, 08 |
| 45 `ablations_device` | CPU | 39 to 42 | device ablations (D1 to D4 against their baselines) | 39 to 42 |

### Phase 14. Tables and figures

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 46 `tables` | CPU | all result folders | every table in the paper, with on-topic counts | all results |
| 47 `figures` | CPU | all result folders | figures in black and white, research style | all results |

### Phase 15. Release

| Step | Compute | Notes |
|---|---|---|
| Repository check: docs, README, manifests | local | every module has a document and a manifest |
| Licence check | local | proposal section 4 |
| Publish the code and small results | local | code and results only; the dataset is not included |

## Coverage check against the proposal

A second pass compared every proposal section with the module list. Gaps are closed by the modules below. Suffixes (02b, 15b and so on) keep the existing numbers stable.

| Proposal item | Covered by | Gap found and closed |
|---|---|---|
| 4 Datasets: fig Original-only evaluation | 01 | 01 must exclude the augmented folder from all evaluation sets |
| 4 Grouping by photo and plant; split files named by SHA-256 (7.11) | **02b** (new) | added |
| 4 Annotation protocol and kappa (7.10) | **03b** (new, manual input) | added |
| 4 Rice hispa with and without (pest vs disease) | 32 | variant run noted in 32 |
| 4 Second encoder on all real pairs (7.7, 7.11) | 06, then 32, 33 | variant run noted in 32 and 33 |
| 5 Saturation rule; acquisition-setting probe; random-label control (7.5) | **15b** (new) | added |
| 7.4 Minimum detectable difference and power per world | 16 | output added to 16 |
| 7.4 Synthetic-world oracle baseline | 16, 17 | added to 16 and 17 |
| 7.7 Sliced-Wasserstein distance (descriptive) | 13 | output added to 13 |
| 7.7 Coverage check of the cluster bootstrap at 25 pairs (7.1) | **36b** (new) | added |
| 7.7 Real-pair power as a function of pair count | **36b** | added |
| 8 Per-class recall, worst-disease, worst-pair accuracy | **46b** (new) | added |
| 8 ECE, Brier score, reliability diagrams | **46b** | added |
| 8 Fit time, memory, latency for every method | **46c** (new) | added; run in one environment for all methods |
| 8 Failure-case section | **46d** (new) | added |
| 8 Ablation: single Gaussian vs mixture; shrinkage sensitivity | 44 | shrinkage sensitivity added to 44 |
| 8 Ablation: diversified predictions (bootstrap maps) | **44b** (new) | added |
| 8 Withholding threshold from source pairs only | 44 | already covered |
| 6 Device components D1 to D4 | 39 to 42, 45 | covered |
| 10 Pre-registration on OSF with identifier (7.11) | Phase 2 gate G2 | a registration document is a manual step, listed under Phase 15 |
| 4 Encoder pre-training corpus description (section 9) | manual document | listed under Phase 15 |

### Added modules

| Module | Compute | Inputs | Outputs | Depends on |
|---|---|---|---|---|
| 02b `splits` | CPU | 01 table | split files grouped by source photo and plant; SHA-256 of each split file | 01 |
| 03b `annotation_agreement` | CPU; manual sample | annotator sheets for a random sample per target | Cohen's kappa per target; class-drop rule input | 03 |
| 15b `saturation_controls` | CPU | 13, 05 | saturation rule per pair; acquisition-setting probe; random-label negative control | 13, 05 |
| 36b `bootstrap_coverage_power` | CPU | 36; synthetic worlds | coverage of the cluster bootstrap at 25 pairs; power against pair count | 36, 16 |
| 44b `ablations_diversity` | CPU | 11, 12, 14 | diversified predictions (affine vs nonlinear; bootstrap maps) | 11, 12, 14 |
| 46b `metrics` | CPU | 17 to 19, 32, 33 | per-class recall; worst-disease and worst-pair accuracy; ECE; Brier score; reliability diagrams; sliced-Wasserstein | 17 to 19, 32, 33 |
| 46c `cost_profile` | CPU | 11 to 14, 20 to 31 | fit time, peak memory and latency per method, all run in one environment | all methods |
| 46d `failure_cases` | CPU | 14, 17, 32, 33 | most confidently wrong pairs; most abstained pairs; example images; per-class breakdown | 14, 17, 32, 33 |

Phase 14 order: 46b, 46c and 46d run before 46 (tables) and 47 (figures).

## Figure inventory

Every figure in the paper is listed here with the module that produces it. All are black and white.

| Figure | Content | Produced by |
|---|---|---|
| Fig. 1 | HATC pipeline (done, in repo) | drawn manually |
| Fig. 2 | Device workflow D1 to D4 (done, in repo) | drawn manually |
| Fig. 3 | Study timeline (done, in repo) | drawn manually |
| Fig. 4 | Dataset audit: samples per dataset and healthy/disease split | 00 (done in audit) |
| Fig. 5 | Image resolution distributions | 00 (done in audit) |
| Fig. 6 | Synthetic worlds (a) to (f): accuracy by method with cluster intervals | 17 to 19, 47 |
| Fig. 7 | Power and minimum detectable difference against replicates | 16, 36b, 47 |
| Fig. 8 | Bootstrap coverage at 25 pairs (synthetic) | 36b, 47 |
| Fig. 9 | Hypothesis N: discriminator lower bound per pair, by world | 15, 47 |
| Fig. 10 | Real pairs: accuracy against the best simple baseline, with TOST outcomes | 32, 33, 37, 47 |
| Fig. 11 | Labelled-data saving curves on the label grid (0 to 200), with censoring | 31, 47 |
| Fig. 12 | Accuracy against coverage (value of abstaining) | 14, 47 |
| Fig. 13 | AUROC of disagreement against failure labels across cutoffs | 14, 30, 47 |
| Fig. 14 | Reliability diagrams (ECE) for real pairs | 46b, 47 |
| Fig. 15 | Per-class recall and worst-disease accuracy | 46b, 47 |
| Fig. 16 | Failure cases: example images of the most wrong and most abstained pairs | 46d, 47 |
| Fig. 17 | Ablations: affine vs Brenier; sample sweep; extension rule | 43, 47 |
| Fig. 18 | Device: enrolment false-enrolment rate; routing shares; drift refits; sequential captures | 39 to 42, 47 |
| Fig. 19 | Label-noise robustness (5% and 10%) | 35, 47 |
| Fig. 20 | Cost: fit time, memory and latency per method | 46c, 47 |
| Fig. 21 | Leave-one-source-dataset-out result | 44, 47 |

## Statistical test inventory

Every test the paper reports, with its module. Each is a pre-registered decision (section 7).

| Test or analysis | Purpose | Produced by |
|---|---|---|
| Two-way cluster bootstrap, 10,000 resamples | intervals for every per-pair contrast | 36 |
| Bootstrap coverage check at 25 pairs | confirms the pair threshold (7.1) | 36b |
| Power and minimum detectable difference per world | stated before running (7.4) | 16 |
| Real-pair power against pair count | reported with the cluster bootstrap (7.7) | 36b |
| TOST at alpha 0.05, margin ±2 points | equivalent / different / inconclusive (7.6) | 37 |
| Holm correction across other contrasts | family-wise error (7.7) | 37 |
| Exact paired test, descriptive only | smallest achievable p-value (7.7) | 38 |
| Hypothesis N: held-out discriminator lower bound | test of hypothesis N (5) | 15 |
| Saturation rule and controls | valid two-sample tests (7.5) | 15b |
| Labelled-saving reach rule and censoring | saving claim (5) | 31 |
| Failure definition relative to best simple baseline | primary failure label (7.7) | 37 |
| AUROC of disagreement against failure labels | certificate evaluation (8) | 30 |
| Sequential test error control (D4) | device claim (6) | 42 |
| Drift statistic threshold | refit and rollback (6) | 41, 39 |

Analyses reported descriptively (not tests): sliced-Wasserstein distance (13), ECE and Brier score (46b), per-class recall (46b), cost profile (46c), failure cases (46d).

## Run location

Default location for each module, under rule 9. The estimate is replaced by the measured run time after the first run, and the module document is updated.

| Location | Modules | Reason |
|---|---|---|
| **Local** (CPU, expected under 30 min) | 01, 02, 02b, 03, 08, 09, 10, 11, 13, 14, 15, 15b, 16, 30, 31, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 44b, 45, 46, 46b, 46d, 47 | small inputs (cached features or metadata); fast on a CPU |
| **Remote** (GPU) | 04, 05, 06, 07, 24, 28 | need a GPU for speed |
| **Remote** (CPU, expected over 30 min) | 00 (audit on the full dataset when the local run is not used), 12, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 29, 32, 33, 36, 36b, 46c | longer runs, or many replicates, or the full dataset |

Cached features (module 05) are produced once in the remote environment and downloaded. The 150 MB feature files are small enough to run the CPU modules locally afterwards.

Local runs use the same code and write the same manifest as remote runs, so the results are directly comparable.

## Encoder run times

Planned and measured run times of the encoder modules.

| Module | Estimated hours | Status |
|---|---|---|
| 05 DINOv2 features | 2.05 h measured (7,375 s) | done |
| 06 second-encoder features | 1.85 h measured (6,648 s) | done |
| 07 zero-label CLIP | to be measured | not started (not done) |
| 04 near-duplicates | ran locally in 83 s | done |
| 24 MDFAN-style | to be measured | not started |
| 28 LoRA fine-tune | to be measured | not started |

## Sign-off

A module is complete only when its output is in `results/NN_name/`, its `manifest.json` is written, its document is in `docs/modules/`, and its result has been read. A gate (G0 to G2) is passed only when the result that sets it has been read and written down.

## Status (6 November 2026)

Rebuilt during the repository audit from the result folders and module documents (the earlier table had duplicate and contradictory rows), updated after the later audits. **The primary analysis is now the real-pair test on PlantWild (modules 08b, 09, 10, 53 to 59); its numbers are in `docs/real_pair_results.md`.** The synthetic benchmark (phases 5 to 13 above) is the secondary analysis; its final numbers are in `docs/harmonised_results.md`. Module documents that carry a banner contain first-pass numbers (unharmonised class names, module 03b). The phase headings above ("Synthetic benchmark (primary)", "Real pairs (primary only if G1 passes)") record the original plan: G1 was not met on the six crop datasets, and the real-pair question was answered with PlantWild instead.

| Module | Status | Document |
|---|---|---|
| 00 audit | done (remote run `results/code_audit_result/`, local run `results/local_audit/`) | `docs/audit_checklist.md`, `results/code_audit_result/audit_findings.md` |
| 00 fig augmentation check (audit item D3) | done | `docs/modules/fig_augmentation_check.md` |
| 01 crop and disease table | done | `docs/modules/01_crop_disease_table.md` |
| 02 pair gate | done: gate not met (case study) | `docs/modules/02_pair_gate.md` |
| 02b splits | done | `docs/modules/02b_splits.md` |
| 03 background check | done | `docs/modules/03_background_check.md` |
| 03b label harmonisation | done: propagated by the full harmonised rerun (all benchmark modules except 13, 34, 39, 40) | `docs/modules/03b_label_harmonisation.md`, `docs/modules/rerun_harmonised.md` |
| 04, 04b near duplicates | done: no near-duplicates | `docs/modules/04_near_duplicates.md`, `04b_near_duplicate_verify.md` |
| 05 DINOv2 features, 05a feature index | done (05 remote GPU; 05a added later, reproduces the index byte for byte) | `docs/modules/05_features_dinov2.md`, `05a_build_feature_index.md` |
| 06 BioCLIP features, 29 second-encoder benchmark | done: same conclusions as DINOv2 | `docs/modules/06_features_bioclip.md`, `29_second_encoder.md` |
| 08, 08b projection | done | `docs/modules/08_projection.md`, `44_ablations_projection.md` |
| 09 healthy-sample sweep | done (affine only) | `docs/modules/09_sweep_healthy_n.md` |
| 10 Gaussianity | done | `docs/modules/10_gaussianity.md` |
| 11 affine host maps | done | `docs/modules/11_host_maps_affine.md` |
| 12 Brenier maps, 44b diversity | done: naive extension fails; affine extension about 1 to 2 points below HATC | `docs/modules/12_host_maps_brenier.md` |
| 13 push-forward | done (real data; not rerun with harmonised classes) | `docs/modules/13_push_forward.md` |
| 14 classify and agreement (real data), 15 hypothesis N, 15b saturation controls | not done | |
| 16 world generator | done | `docs/modules/16_world_generator.md` |
| 17 benchmark, 18 agreement benchmark, 19 (worlds e and f) | done (leak fixed; harmonised). Module 19 was not built as a separate script: all ten world variants a to f run inside modules 17 and 18 (`WORLDS` list in `features/17_benchmark_hatc.py`). Number 49 was never used |  `docs/modules/17_benchmark_hatc.md`, `18_agreement_benchmark.md` |
| 20 alignment baselines (covers part of 21) | done | `docs/modules/20_align_baselines.md` |
| 21 OT baselines (other than 20), 22 CASTER, 24 MDFAN, 25 published GMM-OT, 28 LoRA fine-tune, 07 zero-label CLIP | not done | |
| 23 TENT and SHOT (simplified AdaBN and SHOT styles), 27 pooled source | done inside module 46b | `docs/modules/46b_metrics_cost.md` |
| 26 linear probes, 31 labelled saving | done (leak fixed; harmonised) | `docs/modules/26_linear_probes.md`, `31_labelled_saving.md` |
| 30 certificate baselines | done (partial baselines; harmonised) | `docs/modules/30_certificate_baselines.md` |
| 32, 33 real pairs | not feasible: no pushed model reaches the rice or fig classes | `docs/modules/02_pair_gate.md`, `13_push_forward.md` |
| 34 chili and peach probes | done (not rerun with harmonised classes) | `docs/modules/34_chili_peach_probes.md` |
| 35 label noise | done (accuracy only) | `docs/modules/35_label_noise.md` |
| 36 cluster bootstrap, 37 equivalence, 38 exact test | done (7 clusters, exploratory) | `docs/modules/36_cluster_bootstrap.md`, `37_equivalence.md`, `38_exact_test.md` |
| 36b coverage and power | done: equivalence margin of 2 points is unreachable at any cluster count tried | `docs/modules/36b_bootstrap_coverage_power.md` |
| 39 D1, 40 D2, 41 D3, 42 D4 | done (39 and 40 not rerun with harmonised classes) | `docs/modules/39_d1_enrolment.md` to `42_d4_sequential.md` |
| 43, 44 ablations | done for the listed ablations; the rest is future work | `docs/modules/43_ablation_engine.md`, `44_ablations_projection.md` |
| 45 device ablations | partly covered by the baselines in modules 39 to 42 | |
| 46b metrics and cost (46c is inside 46b), 46d failure cases | done | `docs/modules/46b_metrics_cost.md`, `46d_failure_cases.md` |
| 46 tables, 47 figures | done (8 tables, figures R1 to R6, harmonised results) | `docs/modules/46_47_tables_figures.md` |
| 48 pipeline figure (paper Figure 1) | done | `docs/modules/48_pipeline_figure.md` |
| 50 HATC-S (post-hoc extension) | done, exploratory | `docs/modules/50_hatc_select.md` |
| Harmonised rerun driver | done | `docs/modules/rerun_harmonised.md` |
| 51 projection fairness, recalibration, shrinkage, PlantVillage-only, Fisher-geometry baseline | done, exploratory | `docs/modules/51_review_checks.md` |
| 52 per-image refusal | done, exploratory | `docs/modules/52_persample_refusal.md` |
| 08b PlantWild audit | done (data suitable; duplicates removed by rule) | `docs/modules/08b_plantwild_audit.md` |
| 09 PlantWild DINOv2 features, 10 BioCLIP features | done (remote CPU; 17,328 images; verified) | `docs/real_pair_results.md` |
| 53 real-pair test (primary analysis) | done, registered; verified (3 seeds, 2 projections, 2 encoders, controls); includes the PlantVillage tomato-potato pilot | `docs/modules/53_real_pair_test.md` |
| 54 real-pair agreement check | done | `docs/modules/54_real_agreement.md` |
| 55 to 59 verification and figures | done | `docs/modules/55_to_59_verification.md` |
| 08b projection by PCA (`features/08b_projection_pca.py`), 51 LOCO projection (`features/51_loco_projection.py`) | done (08b: PCA projection for the synthetic ablations; 51: leave-one-crop-out projection, calibration and runs of the review checks) | `docs/modules/44_ablations_projection.md`, `51_review_checks.md` |
| 60 frame-only (background) control | done | `docs/modules/60_background_control.md` |
| 61 calibration and healthy-set decision rule | done | `docs/modules/61_calibration_decision_rule.md` |
| 62 summary of covariance, selection and frame-only runs | done | `docs/modules/62_sensitivity_summary.md` |
| 63 learning curve, mean/covariance split, EIG and unlabelled-target arms (summary) | done | `docs/modules/63_review4_summary.md` |
| 64 label-free risk scores, label-noise proxy and replication lists | done | `docs/modules/64_risk_noise_release.md` |
