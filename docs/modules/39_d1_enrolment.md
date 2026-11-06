# Module 39: D1, enrolment gate with refusal

**Compute:** CPU. Runs locally in about 5 seconds.

**Question.** Does the D1 enrolment gate (proposal section 6) refuse a new crop whose healthy captures are mixed with a different species, while accepting clean captures?

**Gate.** The device captures healthy leaves of a new crop and builds a profile in the 16-dimensional projection. It enrols the crop only if:
- there are at least 50 captures,
- the condition number of the shrunk covariance is at most a threshold, and
- the spread (trace of the shrunk covariance) is at most a threshold.

**Thresholds** are set on **source crops only**: the 95th percentile of each statistic over clean captures of the source crops, at each sample size. No target data is used. (`results/39_d1_enrolment/thresholds.csv`)

**Mismatch set.** A capture of n healthy images in which a fraction (10%, 25% or 50%) comes from a different crop. 200 trials per cell, crops with at least 250 healthy training images.

## Results (`results/39_d1_enrolment/enrolment_rates.csv`)

| Captures | Share from the wrong crop | False enrolment, gate | False enrolment, no gate | False refusal of clean captures |
|---|---|---|---|---|
| 50 | 10% / 25% / 50% | 0.91 / 0.91 / 0.86 | 1.00 | 0.12 |
| 100 | 10% / 25% / 50% | 0.92 / 0.93 / 0.87 | 1.00 | 0.12 |
| 200 | 10% / 25% / 50% | 0.95 / 0.95 / 0.93 | 1.00 | 0.16 |

## Reading

- **The gate does not detect a different species.** It enrols 86 to 95% of mixed captures, against 100% with no gate at all. The false refusal rate on clean captures is 12 to 16%, so the gate costs more clean enrolments than it blocks mismatched ones.
- **Why.** In the projected space, a mix of two healthy crops is not more spread out or worse conditioned than a single crop's healthy sample. The statistics the gate uses do not separate the two. A first-time enrolment also has no reference profile to compare against.
- **Consequence for the device claim.** The proposal's D1 claim, a refusal of mismatched crops using these statistics, is **not supported** by this simulation. A species check would need a different test, for example comparing the new profile with the profiles of known crops, or a multimodality test. That would be a change to the design and needs a team decision.

## Limits

- The mismatch set is a mixture with one other crop; the real mismatch (a whole capture of a different species) is not simulated.
- The thresholds use the 95th percentile of clean source captures, which sets the false refusal near 5% by design on the source; the observed 12 to 16% is higher because the clean captures here include crops outside the threshold draw.
- Only the projection from module 08 is used. A gate on raw features might differ.

**How to re-run:**
```bash
python features/39_d1_enrolment.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/39_d1_enrolment
```
