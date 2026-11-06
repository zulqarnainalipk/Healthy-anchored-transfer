# Module 26: few-shot linear probes on the label grid (synthetic worlds)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. About 21 minutes, run as four parallel jobs (worlds split into four groups, 4 CPU cores). The four result tables were merged into one file afterwards.

**Question.** How accurate is a classifier trained on labelled **target** disease images, as a function of the number of labels per class? This is the primary labelled-target baseline (proposal section 5, 8), and the basis for the labelled-saving measurement (module 31).

**Method.**
- Same worlds, replicates and test samples as modules 17 and 20 (single source crop, 16-dimensional projection from module 08).
- A logistic-regression classifier is trained on labelled target samples at 0, 5, 10, 25, 50, 100 and 200 labels per class, and tested on held-out target samples. At 0 labels the score is chance (1 / number of classes).
- The probe does not use healthy images; it classifies among disease classes.

## Correction: the first version had a train-test leak

The first version of this module drew the probe's training samples and the test samples from the **same pool** of images, so the probe could be tested on images it had trained on. The leak was found when module 31 showed the probe reaching HATC's accuracy with only 5 labels per class. All first-version numbers were discarded.

The fix: each crop-disease pool is split into three **disjoint** parts (50% for source models, 25% for target test samples, 25% for probe training). The operator strength was then recalibrated from 3 to 2 on the no-adaptation score alone (see `17_benchmark_hatc.md`). The numbers below are from the corrected run, strength 2.

## Results (`results/26_linear_probes/probe_curve.csv`; mean accuracy over replicates)

| Labels per class | World a | World c50 | World e |
|---|---|---|---|
| 0 (chance) | 0.22 | 0.22 | 0.22 |
| 5 | 0.66 | 0.66 | 0.61 |
| 10 | 0.70 | 0.70 | 0.65 |
| 25 | 0.73 | 0.73 | 0.70 |
| 50 | 0.73 | 0.73 | 0.70 |
| 100 | 0.74 | 0.74 | 0.72 |
| 200 | 0.74 | 0.73 | 0.71 |

The curves are nearly the same in every world: a probe trained on target labels does not depend on the relation between source and target. It levels off at about 0.73 to 0.74, from about 25 labels per class.

## Limits and notes

- **Pairing.** The probe's test samples equal HATC's only for healthy n = 50, because the random stream aligns for that cell. For healthy n = 200 the probe and HATC use different draws from the same distribution. Module 31 therefore compares means over replicates, not paired replicates.
- **The projection favours the probe.** The 16-dimensional projection (module 08) was fitted to separate exactly these source disease classes, and the worlds reuse those classes. A probe working in a space built to separate its classes has an easy task. With real target diseases that the projection has never seen, the probe's accuracy would be lower. This makes the synthetic probe a favourable baseline, and the labelled-saving result in module 31 correspondingly conservative for HATC.
- **Parallel run.** The module ran as four jobs, each on a group of worlds, with the same seeds. Only the order of execution differs from a single run; results do not depend on the split.
- 117 replicates per world; each replicate is one crop.

**How to re-run (one group):**
```bash
python features/26_linear_probes.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/26_linear_probes/g1 --scale 2 --worlds a a2 b
```
The four groups used were: (a, a2, b), (b2, c25, c50), (c100, d), (e, f). Merge the four `probe_curve.csv` files.
