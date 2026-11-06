# Modules 55 to 59: verification of the real-pair result

All are CPU modules; results in `results/<module>/`; summary in `docs/real_pair_results.md`.

- **55 independent re-implementation** (`analysis/55_independent_check.py`): re-implements, without importing the benchmark code, the covariance shrinkage, the Bures-Wasserstein map, NA/MC/STD/CORAL/HATC, the Gaussian likelihoods (Cholesky) and an exact-SVD leave-one-crop-out PCA, and compares the balanced accuracy on 15 random units × 5 methods with `units.csv` of module 53. Result: maximum absolute difference 0.0021, mean 0.0003, all within 0.01.
- **56 primary checks** (`analysis/56_primary_checks.py`): bootstrap with five seeds and 20,000 resamples, leave-one-target-crop-out and leave-one-source-crop-out, cluster sign-flip permutation test, per-crop summaries, paired contrasts (HATC − OC, HATC − CORAL, probe − HATC) and agreement across split seeds. Result: interval 0.045 to 0.100 for every bootstrap seed; leave-one-out ranges +6.7 to +7.5 (targets) and +6.3 to +7.7 (sources); permutation p = 0.00005; 20 of 22 target and 21 of 22 source crops positive; seed correlation 0.996.
- **57 feature spot check** (`analysis/57_feature_spotcheck.py`): recomputes 40 random DINOv2 feature vectors locally with the pinned revision and compares them with the stored ones, and checks that every listed file exists and that each row's class equals its folder. Result: minimum cosine similarity 1.0000, largest relative difference 0.00022, 0 missing files, 0 class mismatches (needs `torch` and `transformers`).
- **58 stricter duplicates** (`analysis/58_strict_duplicates.py`): groups images whose 64-bit difference hashes differ by at most 8 bits; a group spanning more than one class is removed, within a class one image is kept. Removes 1,864 further images (281 groups); module 53 with `--exclude-file` gives HATC − MC +6.9 points (4.4 to 9.7).
- **59 figures** (`analysis/59_real_pair_figures.py`): figures R7 (real-pair methods and per-unit differences) and R8 (controls).

## How to re-run

All commands run from the repository root; the module-53 run `pw_pca` must exist. `<data-root>` is the folder that holds the `plantwild` dataset folder.

```
python analysis/55_independent_check.py --features results/09_features_plantwild/remote/features_plantwild --units results/53_real_pair_test/pw_pca/units.csv --output results/55_independent_check
python analysis/56_primary_checks.py --units results/53_real_pair_test/pw_pca/units.csv --other-seeds <units.csv of the two other seeds> --tag pca_seeds --output results/56_primary_checks
python analysis/57_feature_spotcheck.py --data-root <data-root>/plantwild --features results/09_features_plantwild/remote/features_plantwild --output results/57_feature_spotcheck
python analysis/58_strict_duplicates.py --data-root <data-root>/plantwild --features results/09_features_plantwild/remote/features_plantwild --output results/58_strict_duplicates
python analysis/59_real_pair_figures.py --results results --output figures
```

Module 57 imports torch and transformers and takes a long time to start; the others need only the base environment. The exact arguments of each run are recorded in the stdout logs next to the results.