# Module 31: labelled-data saving on the synthetic worlds

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in under a second.

**Question.** How many labelled diseased images per class does HATC save a new crop? HATC uses none. The baseline is a few-shot linear probe (module 26), which "reaches" HATC at the first grid point where its mean accuracy is within the registered tolerance (0.02) of HATC's. That grid point is the saving. If the probe never reaches HATC by 200 labels, the cell is censored and the saving is at least 200 (a lower bound).

## Correction: the first version had a train-test leak

The first version showed the probe reaching HATC with 5 labels per class. That was partly due to a leak: the probe, the source models and the test samples were drawn from the same pool of images. The leak was fixed with disjoint pools, and the operator strength was recalibrated from 3 to 2 (see `17_benchmark_hatc.md` and `26_linear_probes.md`). **The result held after the fix**, so it is not an artefact of the leak.

## Results (`results/31_labelled_saving/labelled_saving.csv`)

In **all 20 cells** (10 worlds × healthy n of 50 and 200), the probe reaches HATC's accuracy at the **first grid point, 5 labels per class**. No cell is censored.

| World | HATC accuracy (n = 50 / 200) | Probe at 5 labels | Probe at 25 labels |
|---|---|---|---|
| a (affine) | 0.55 / 0.61 | 0.66 | 0.73 |
| a2, d, f (shifts) | 0.56 / 0.64 | 0.67 | 0.74 |
| b (radial) | 0.56 / 0.64 | 0.67 | 0.73 |
| b2 | 0.53 / 0.58 | 0.67 | 0.74 |
| c25, c50, c100 | 0.47 to 0.59 | 0.66 | 0.72 to 0.73 |
| e | 0.33 / 0.34 | 0.61 | 0.70 |

## Reading

- **The labelled-saving claim is not supported in this synthetic benchmark.** Five labelled target images per class already beat HATC's zero-label accuracy in every world. The true crossover lies between 0 and 5 labels, below the smallest non-zero grid point, so HATC saves at most 5 labels per class here.
- The probe is helped by the projection, which was trained to separate exactly these classes (see `26_linear_probes.md`). The saving for real targets, with disease classes the projection has not seen, may differ. This benchmark cannot answer that.
- The proposal's wording for the saving claim is conditional ("claimed only if the interval over non-censored pairs excludes zero"). That condition is not met on this evidence.
- The question that remains for the real data (rice, fig) is open: only a real-target comparison can show a saving, and the real-pair gate was not met (module 02). The labelled-saving result therefore cannot be established from the real data either, and the paper must say so.

## Limits

- The grid has no point between 0 and 5 labels. A finer grid (1, 2, 3 labels) would sharpen the estimate.
- Means over replicates, not paired replicates. No intervals yet (module 36).
- The probe is a favourable baseline (projection fitted to the classes).

**How to re-run:**
```bash
python features/31_labelled_saving.py --hatc results/20_align_baselines/replicate_results.csv --probes results/26_linear_probes/probe_curve.csv --output results/31_labelled_saving
```
