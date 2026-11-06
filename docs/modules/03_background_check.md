# Module 03: background shortcut check (audit items P4 and P5)

**Compute:** CPU. Reads the peach and chili images (6,748 files). Runs locally.

**Question.** Can the background alone predict the class? If it can, a model can learn the background instead of the leaf, and the result would not mean what it claims.

**Method.**
- For each image, the mean brightness of the four corner patches (16 × 16 pixels) is measured.
- A background label is given: black (below 30), white (above 225), or other.
- A shortcut test uses one brightness threshold, learned on four folds and tested on the fifth. It predicts healthy versus everything else. Folds are stratified by class, with a fixed seed.

**Results** (`results/03_background_check/summary.json`):

| Dataset | Class | Images | Black | Other | Mean corner brightness |
|---|---|---|---|---|---|
| Peach | healthy | 107 | 99 | 8 | 11.4 |
| Peach | mechanical stress | 44 | 40 | 4 | 9.9 |
| Peach | abiotic stress | 24 | 22 | 2 | 11.6 |
| Peach | chewing insect | 7 | 6 | 1 | 5.9 |
| Chili | healthy | 3,283 | 1,178 | 2,105 | 52.4 |
| Chili | yellow leaf | 3,283 | 0 | 3,283 | 166.8 |

**Shortcut test:**
- **Peach:** 54% accuracy, against a majority rate of 59%. The background carries no class information. The uniform black background is still stated as a limit.
- **Chili:** 98% accuracy, against a 50% chance rate. The background alone solves the binary task.

**Consequence for chili.** The chili binary target is not valid as it stands. Any chili result must use a background-controlled input (for example, a central crop or background masking) and must report the 98% shortcut accuracy beside its result. The chili metric is reported only after that control.

**How to re-run:**
```bash
python data_prep/03_background_check.py --labelled results/01_crop_disease_table/labelled_images.csv --data-root "<data-root>" --datasets peach_leaf_damage chili_yellow_leaf --output results/03_background_check
```
