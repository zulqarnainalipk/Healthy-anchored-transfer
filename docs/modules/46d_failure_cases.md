# Module 46d: failure cases on the synthetic worlds

**Compute:** CPU, about 9 seconds locally (worlds a and e, healthy n = 200, 58 replicates).

**Question.** Where does HATC (and the shrunk map) fail? This module pooled the target test predictions and exposed the label-naming problem that led to module 03b (`03b_label_harmonisation.md`).

**Method.** Per method: recall for every (crop, disease) class with the worst ten; the ten most frequent confusions inside a crop; the most confidently wrong predictions (share of wrong predictions with probability above 0.9). No images are shown, because the synthetic worlds act on cached features.

**Outputs** (`results/46d_failure_cases/n200/`; the first-pass copy is in `results/_first_pass_unharmonised/46d_failure_cases/n200/`): `per_class_recall.csv`, `top_confusions.csv`, `confident_errors.csv`, `manifest.json`. The harmonised rerun is the one in `results/46d_failure_cases/`.

**Reading.** The most frequent confusion of HATC in world a was `potato Early_blight` to `potato early blight` (147 cases): the same disease under two names, one from each source dataset. Merging such names is module 03b. Numbers in the first-pass folder are superseded by `docs/harmonised_results.md`.

**How to re-run:**
```bash
python features/46d_failure_cases.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/46d_failure_cases --tag n200 --healthy-n 200 --worlds a e --reps 58
```
