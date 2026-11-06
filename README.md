# healthy-anchored-transfer

Code, results and registered analysis rules for a benchmark of **healthy-anchored transfer of plant-disease models across crops (HATC)**: can healthy leaf images of a new crop alone anchor the transfer of disease models from known crops, and does an agreement check tell us when the transfer will fail?

## Purpose and headline result

The **primary analysis** is a registered test on real crop pairs from PlantWild v1 (80 units of tier 2, 22 target and 22 source crops, frozen DINOv2 features): HATC beats mean-centring by 7.1 points of balanced accuracy (90% interval 4.5 to 10.0), but matches the simpler CORAL alignment (difference -0.3 points), and a one-class healthy detector with no source disease recovers about half of the gain over mean-centring (`docs/real_pair_results.md`). The **secondary analysis** is a benchmark on synthetic target worlds built from real image features (PlantVillage and PlantSeg crops), plus real-data audits and probes; there HATC does not beat the best simple method (`docs/harmonised_results.md`). The repository keeps the negative results, the failure modes found on the way (a label-naming error, a train-test leak) and the corrected analysis. Checks added after the primary result are labelled post hoc or exploratory in the documents.

Companion paper: *Can Healthy Leaves Alone Transfer Plant Disease Models Across Crops? A Benchmark of Healthy-Anchored Alignment* (MDPI format; journal and DOI to be added). Status: pre-release.

## Author and contact

Zulqarnain Ali, Department of Information and Communication Engineering, Yeungnam University, 280 Daehak-ro, Gyeongsan 38541, Gyeongbuk, Republic of Korea.
Email: zulqar445ali@gmail.com. ORCID: [0009-0009-0281-882X](https://orcid.org/0009-0009-0281-882X).

## Layout

```
healthy-anchored-transfer/
├── README.md                     this file
├── LICENSE.proposed              proposed licence text; the licence is pending and this file is not in force
├── CITATION.cff                  citation metadata (author, ORCID; DOI, release date and licence still open)
├── requirements.txt              base environment, pinned
├── requirements-gpu.txt          extra packages for the encoder (feature extraction) modules
├── .gitignore                    large-file and privacy policy
├── docs/
│   ├── README.md                 documentation index and large-file policy
│   ├── real_pair_results.md      PRIMARY: real-pair test on PlantWild, controls, verification
│   ├── harmonised_results.md     SECONDARY: final numbers of the synthetic benchmark
│   ├── registration.md           analysis rules fixed before target data were used (+ .sha256)
│   ├── data_sources.md           datasets, licences, what is and is not redistributed
│   ├── audit_checklist.md        what the dataset audit establishes
│   ├── implementation_checklist.md   working rules, module list, status table
│   └── modules/                  one document per module, each with a "How to re-run" command
├── data_audit_code/              dataset audit, PlantWild audit (08b), fig augmentation check
├── data_prep/                    01 crop-disease table, 02 pair gate, 02b splits, 03 background check, 05a feature index
├── features/                     05/06/09 feature extraction, 04/04b, 08 to 13, 16 to 18, 20, 26, 30, 31, 34 to 52, 53 real-pair test, 54 real-pair agreement
├── analysis/                     46 tables, 47 figures, 48 pipeline panels, 55 to 58 verification, 59 real-pair figures, 60 to 64 post-hoc summaries, merge_groups.py, rerun_all_harmonised.ps1
├── figures/                      result figures R1 to R8 (png, pdf; R7 and R8 are the real-pair figures) and the pipeline figure (fig4_pipeline/)
├── reference/                    (not in git) third-party PlantSeg metadata, see docs/data_sources.md
└── results/                      one folder per module; see results/README.md
```

Module numbers are those of the implementation checklist (`docs/implementation_checklist.md`); script names carry the module number.
Module numbers are not unique in two places: `08b` is both the PlantWild audit (`data_audit_code/audit_plantwild.py`) and the PCA projection (`features/08b_projection_pca.py`), `09` and `10` name both the synthetic-benchmark modules (healthy-n sweep, Gaussianity) and the PlantWild feature extraction (`09_features_plantwild`, `10_features_plantwild_bioclip`), and `48` and `48b` share one document. The file name and the result folder name always disambiguate.


## Data availability and licences

- **No dataset image is stored in this repository** (with the exception of ten reduced-size leaf photographs in Figure 1, listed in `docs/data_sources.md`). The seven public datasets (PlantWild v1 for the primary analysis; PlantVillage colour, PlantSeg v6, Sartak fig leaf dataset, rice leaf disease, chili yellow leaf, peach leaf damage for the secondary analysis and the audit) are downloaded by the user from their records.
- **PlantWild is licensed CC BY-NC-ND 4.0 (non-commercial, no derivatives) and is not redistributed here.** Only per-image tables (relative paths, labels), derived result tables and the code are included. Whether the derived PlantWild feature arrays may be released in the archive is an open decision (`docs/data_sources.md`, open point 6).
- What is stored: per-image tables (relative paths, labels, splits), small result tables, manifests, logs and figures. The cached feature arrays (14 files, 404 MB) are **not** in git; they are to be published as a release archive with SHA-256 checksums (see `docs/README.md`).
- **Licence of this repository: pending.** The owner has not yet decided. `LICENSE.proposed` holds a proposal (MIT for code, CC BY 4.0 for documentation and results); until it is adopted, no licence is granted. Third-party datasets and the photographs in Figure 1 keep their own licences (`docs/data_sources.md`).

## How to obtain the data

Follow the records and folder names in `docs/data_sources.md`. Put the dataset folders (including `plantwild`) under one data root (any location). Also download `Metadata.csv` of PlantSeg v6 into `reference/plantseg_v6/`.

## Environment

Python 3.12 (the pinned versions in `requirements.txt` were installed under Python 3.12.10 on Windows 10). Base environment: `python -m pip install -r requirements.txt` (numpy, scipy, scikit-learn, matplotlib, pillow; versions pinned). Feature extraction (modules 05, 06, 09) and the feature spot check (`analysis/57_feature_spotcheck.py`) additionally need `python -m pip install -r requirements-gpu.txt` (torch, transformers, open_clip_torch). The torch and transformers pins are the versions installed locally and are not a record of the original feature run (see the header of that file). Every script logs its steps and progress with time stamps, and every script takes `--help`.

## Sanity check

No unit-test suite is included. A quick check that the code is intact and the environment is complete (writes no files, takes a few seconds, needs only `requirements.txt`):

```
python -B -c "import pathlib; ps=[p for p in pathlib.Path('.').rglob('*.py') if 'results' not in p.parts]; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in ps]; print(len(ps), 'scripts compile')"
python -B -c "import numpy, scipy, sklearn, matplotlib, PIL; print('imports ok')"
python -B features/53_real_pair_test.py --help
```

At the last audit all 60 scripts compiled and the `--help` of 57 of 60 scripts printed their arguments (the three feature-extraction and spot-check scripts need torch; `figures/fig4_pipeline/build_pipeline.py` builds the figure and has no options). This checks syntax and arguments only; it does not reproduce a result.

## Reproduce the key results

All commands run from the repository root. Run times are indicative only: they depend on the machine, and the measured run time of each module is in its document in `docs/modules/` and in the `manifest.json` of its result folder. Feature extraction is by far the slowest step (1 to 2 hours for PlantWild; the six-dataset extraction is longer).

0. **Primary analysis (real pairs, PlantWild).** Audit: `python data_audit_code/audit_plantwild.py --data-root <data-root>/plantwild --output results/plantwild_audit`; features: `python features/09_features_plantwild.py --data-root <data-root>/plantwild --exclude results/plantwild_audit/dedupe_exclude.csv --output results/09_features_plantwild/remote/features_plantwild` (or use the archive); test: `python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag pw_pca --output results/53_real_pair_test --tiers 1 2 --reps 20`; agreement check: `features/54_real_agreement.py`; verification: `analysis/55` to `58`; figures: `analysis/59_real_pair_figures.py`. All commands and expected numbers are in `docs/modules/53_real_pair_test.md`, `54_real_agreement.md`, `55_to_59_verification.md` and `docs/real_pair_results.md`.
1. **Features of the secondary analysis (skip if you use the archive).** `python features/05_features_dinov2.py --dataset-root <data-root> --index results/05_features_dinov2/feature_index.csv --output results/05_features_dinov2/remote/features_dinov2` and `python features/06_features_bioclip.py` (arguments in `docs/modules/06_features_bioclip.md`). The index is rebuilt by `python data_prep/05a_build_feature_index.py --splits results/02b_splits/splits.csv --labelled results/01_crop_disease_table/labelled_images.csv --output results/05_features_dinov2`. **Easier:** download the feature archive and unpack it into `results/05_features_dinov2/remote/features_dinov2/` and `results/06_features_bioclip/remote/features_bioclip/`.
2. **Tables of labels and splits** (already in the repository; to regenerate): `data_audit_code/audit_dataset.py --input <data-root> --output results/local_audit`, then `data_prep/01_crop_disease_table.py`, `data_prep/02_pair_gate.py`, `data_prep/02b_splits.py` (commands in `docs/modules/`).
3. **Projection and host maps:** `python features/08_projection.py --features results/05_features_dinov2/remote/features_dinov2 --output results/08_projection` (a few minutes), then modules 10, 11, 13.
4. **Secondary analysis: synthetic benchmark, statistics, device components, tables and figures in one go:** `powershell -File analysis/rerun_all_harmonised.ps1` (several hours; archives the old results first). It runs modules 17, 18, 20, 36 to 38, 26, 30, 31, 35, 41 to 43, 50, 12, 46b, 46d, then `analysis/46_tables.py` and `analysis/47_figures.py`.
5. **Compare with the stored numbers** in `docs/real_pair_results.md` (primary), `docs/harmonised_results.md` and `results/46_tables/` (secondary). Module documents carrying the banner "first pass" describe the run before label harmonisation; `docs/harmonised_results.md` is authoritative.
6. **Second encoder (BioCLIP):** commands in `docs/modules/29_second_encoder.md`.
7. **Post-hoc checks (exploratory):** modules 60 to 64, commands in their module documents.

Each module is one script with one purpose and one document; a failed module is re-run alone. The module list, run location and status are in `docs/implementation_checklist.md`. The registration of the analysis rules is `docs/registration.md`; its SHA-256 is in `docs/registration.sha256` (check with `Get-FileHash docs/registration.md -Algorithm SHA256`).

## Status

Dataset audit, the real-pair test on PlantWild (modules 08b, 09, 10, 53 to 59, with post-hoc modules 60 to 64) and the synthetic benchmark are complete (see the status table in `docs/implementation_checklist.md`). Not done and stated as such: modules 14, 15, 15b, 21 (beyond 20), 22, 24, 25, 28 and 07. Real-pair modules 32 and 33 (rice and fig as targets) are not feasible because the pair gate is not met on the six crop datasets; the real-pair question is answered with PlantWild instead (module 53).

Open decisions for the owner before release: the licence (see above), the archive host and its DOI, and the dataset-licence points in `docs/data_sources.md`.

## Citation

Please cite the paper and the archived release. Machine-readable metadata is in `CITATION.cff`. The DOI, release date and journal are not yet assigned and are deliberately left open.

```bibtex
@software{ali2026healthyanchored,
  author  = {Ali, Zulqarnain},
  title   = {healthy-anchored-transfer: code, results and registered analysis rules for a benchmark of healthy-anchored alignment of plant-disease models across crops},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/zulqarnainalipk/Healthy-anchored-transfer},
  note    = {Pre-release. DOI to be added.}
}

@unpublished{ali2026healthyleaves,
  author = {Ali, Zulqarnain},
  title  = {Can Healthy Leaves Alone Transfer Plant Disease Models Across Crops? A Benchmark of Healthy-Anchored Alignment},
  year   = {2026},
  note   = {Manuscript in preparation. Journal and DOI to be added.}
}
```
