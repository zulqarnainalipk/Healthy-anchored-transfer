# Module 52: per-sample refusal criteria

**Compute:** CPU, about 40 seconds (`features/52_persample_refusal.py`). **Status:** exploratory, added as a design check after the first full set of results.

**Question.** The agreement check of modules 18 and 30 decides per replicate. Would a per-image refusal rule detect the classifier's own errors better? (Design question 5.)

**Method.** Same worlds, pools and replicates as module 46b (worlds a, c50, e, f; 200 healthy images; operator strength 2; 100 replicates), methods MC and HATC. Three per-image scores for detecting an error of the classifier:
- **conf:** the maximum posterior probability of the pushed class models (equal priors);
- **density:** the maximum log-likelihood of the image under the pushed class models;
- **disagree:** the classifier built from a second source (the same classes through a fixed mild random affine look, pushed through its own healthy-anchored map) predicts a different class.
Metrics: AUROC for detecting errors (pooled over all images) and the accuracy when the 20% of images with the least trustworthy score are refused (coverage 0.8).

## Results (`results/52_persample_refusal/refusal.csv`; HATC)

| World | Accuracy | AUROC conf | AUROC density | AUROC disagree | Accuracy at 80% coverage (conf) |
|---|---|---|---|---|---|
| a | 0.668 | 0.68 | 0.55 | 0.56 | 0.715 |
| c50 | 0.648 | 0.67 | 0.54 | 0.56 | 0.698 |
| e | 0.298 | 0.52 | 0.51 | 0.50 | 0.293 |
| f | 0.688 | 0.72 | 0.55 | 0.57 | 0.754 |

## Reading

- **Confidence is the best per-image refusal score** (AUROC 0.67 to 0.72 where the relation is identifiable; refusing the 20% least confident images raises accuracy by 5 to 7 points). Density and the two-source disagreement are weak (0.50 to 0.57).
- In world e, where no healthy-only method can recover the relation, no per-image score helps (0.50 to 0.52).
- So the plain confidence of the classifier is a better refusal rule than the registered agreement check, which fits the replicate-level result of module 30 (ensemble disagreement and confidence beat the bound).

## Limits

Four worlds; the second source is a mild affine copy; MC and HATC only; no intervals; cycle-consistency of the transport was not tested (for an affine map the cycle is exact, so it adds no information).

**How to re-run:**
```bash
python features/52_persample_refusal.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/52_persample_refusal --worlds a c50 e f --healthy-n 200 --reps 100
```
