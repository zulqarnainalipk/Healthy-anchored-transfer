# Module 36b: coverage and power of the cluster bootstrap

**Compute:** CPU. About 20 seconds locally.

**Question.** What does the cluster bootstrap of modules 36 and 37 deliver at 7 clusters (our synthetic worlds) and at the registered 25 pairs? (Proposal 7.7 and 8.)

**Method.** Variance components come from the stored paired differences (HATC minus best simple method, healthy n = 200, module 20; rerun on the harmonised replicates): median between-crop SD 0.089, within-crop SD 0.046, 17 replicates per crop. Simulation, 2,000 runs per row: cluster effect = delta + N(0, between), observation = effect + N(0, within). The 90% percentile bootstrap uses 1,000 resamples (10,000 in module 36). Equivalence margin ±2 points, alternative difference 3 points.

## Results (`results/36b_bootstrap_coverage_power/coverage_power.csv`)

| Clusters | Coverage of 90% interval (true diff 0) | Coverage (true diff −3 pts) | Power to show a 3-point difference | Power to show equivalence (margin ±2 pts) |
|---|---|---|---|---|
| 7 | 0.83 | 0.82 | 0.32 | 0.001 |
| 8 | 0.85 | 0.83 | 0.31 | 0.001 |
| 12 | 0.85 | 0.85 | 0.38 | 0.001 |
| 25 | 0.89 | 0.87 | 0.55 | 0.002 |
| 50 | 0.89 | 0.88 | 0.78 | 0.035 |

## Reading

- **Intervals are too narrow at few clusters.** At 7 clusters the nominal 90% interval covers the truth about 83% of the time; at 25 clusters it is about 89%. The percentile bootstrap is known to under-cover with few clusters. Our exploratory intervals are therefore somewhat optimistic.
- **The registered design cannot show equivalence at ±2 points.** The between-crop SD (0.089) is much larger than the margin. Even at 25 clusters the power to show equivalence is 0.2%, and at 50 clusters 3.5%. With this variability, the equivalence outcome will almost always be "inconclusive" or "different". The margin or the design (more replicates cannot help; only more clusters or a smaller between-crop spread) needs rethinking before any real-data claim of equivalence.
- **Power to show a 3-point difference** at 25 clusters is 0.55, below the registered 0.8. About 50 clusters would be needed.
- This is a consequence of the synthetic worlds' between-crop variability, which may be larger or smaller on real pairs.

## Limits

Normal random effects, equal cluster sizes, one cluster dimension; variance components from the synthetic worlds only; 1,000 resamples per simulation.

**How to re-run:**
```bash
python features/36b_bootstrap_coverage_power.py --replicates results/20_align_baselines/replicate_results.csv --output results/36b_bootstrap_coverage_power
```
