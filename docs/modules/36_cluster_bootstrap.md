# Module 36: cluster-bootstrap intervals

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in about 4.5 minutes (10,000 resamples per interval).

**Question.** How uncertain are the paired accuracy differences between HATC and the baselines on the synthetic worlds (proposal 7.7)?

**Method.**
- Input: `results/20_align_baselines/replicate_results.csv` (corrected run, strength 2, disjoint pools).
- Each replicate is one crop. Replicates of the same crop share source samples, so the bootstrap resamples **crops** (clusters) with replacement, not replicates.
- For each world and healthy-image count, the paired difference HATC minus each baseline (NA, MC, STD, CORAL) and minus the **best simple baseline** (the best of NA, MC and STD by mean accuracy in that cell) is computed per replicate.
- Intervals are percentile intervals from 10,000 cluster-bootstrap resamples: 90% (for the equivalence test, module 37) and 95%.

**Results** (`results/36_cluster_bootstrap/intervals.csv`): 20 cells × 5 contrasts = 100 intervals.

## Two limits stated by the registration

1. **Only 7 clusters (crops) per cell, below the registered minimum of 8.** Seven crops have at least two disease classes with at least 60 images. By the registered rule, every interval from this benchmark is **exploratory**. It is reported as such, and the paper must say so.
2. **One cluster dimension.** The proposal's bootstrap resamples the target crop and the source crop (two-way). In the synthetic worlds each replicate uses one crop as both, so the bootstrap reduces to one dimension. For real pairs (rice and fig as targets) the two-way structure would apply, but that comparison is a case study.

**Other notes.**
- Intervals with few clusters are wide. A 90% bootstrap interval from 7 clusters is a rough guide only.
- The best simple baseline is chosen in each cell by its mean accuracy on the same data. This favours the baseline slightly, which is conservative for HATC.

**How to re-run:**
```bash
python features/36_cluster_bootstrap.py --replicates results/20_align_baselines/replicate_results.csv --output results/36_cluster_bootstrap
```
