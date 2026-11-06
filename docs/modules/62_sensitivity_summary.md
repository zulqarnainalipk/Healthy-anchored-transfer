# Module 62: summary of post-hoc covariance, selection and frame-only runs

**Code:** `analysis/62_sensitivity_summary.py` (reads `results/53_real_pair_test/sens/*`, writes `results/62_sensitivity_summary/summary.csv`). Runs, all tier 2, DINOv2, PCA, 80 units; HATC minus MC (points, 90% two-way cluster bootstrap): registered lambda 0.1 +7.1 (4.5 to 10.0); lambda 0.03 +6.0 (3.3 to 8.9); lambda 0.3 +6.2 (4.0 to 8.7); Ledoit-Wolf +6.4 (3.9 to 9.3); OAS +6.3 (3.7 to 9.3); rank 8 +5.1 (2.2 to 8.4); rank 4 +5.4 (2.7 to 8.4); rank 2 +3.2 (0.6 to 5.9); frame-only -1.6 (-3.1 to -0.3). HATC-S (`--select`): SELECT minus MC +7.4 (4.7 to 10.3); choices over 1,680 splits: CORAL 1,300, S75 227, HATC 132, S50 21.

**Commands** (one per run; tags as in the folder names):
```bash
python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag cov_lw --output results/53_real_pair_test/sens --cov lw
python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag cov_rank4 --output results/53_real_pair_test/sens --cov lowrank --rank 4
python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag select --output results/53_real_pair_test/sens --select
python analysis/62_sensitivity_summary.py --results results
```
