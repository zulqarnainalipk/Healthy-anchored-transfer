# Module 63: summary of the post-hoc checks in the third round of post-hoc checks

`analysis/63_review4_summary.py` reads the unit tables of module 53 and writes `results/63_review4_summary/`:

- `learning_curve.csv`: tier 2, DINOv2, 10/25/50/100/200 target healthy images in the map set (folders `curve/n10`, `n25`, `n100`, `pw_pca_n50`, `pw_pca`); mean balanced accuracy of NA, MC, CORAL, HATC, one-class and the 5-label probe, and HATC minus MC / CORAL with the 90% two-way cluster-bootstrap interval. The test set shrinks as the map set grows, so levels are not comparable between rows.
- `by_disease.csv`: per disease type and encoder, MC, STD, CORAL, HATC and the differences from MC (mean part, scale part, covariance part).
- `extra_baselines.csv`: EIG, UDA_CORAL, UDA_BW and the 5-label probe against HATC for both encoders, with intervals.

Key results: HATC minus MC +1.0 / +4.8 / +6.2 / +7.0 / +7.1 points at 10 / 25 / 50 / 100 / 200 images; STD minus MC -0.5 and CORAL minus MC +7.4 (DINOv2); EIG 0.606 against HATC 0.787; UDA_BW 0.845 against HATC 0.787 (DINOv2).

How to re-run (CPU, minutes; the module-53 runs take 10 to 40 minutes each):

```bash
python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag extra_dino --output results/53_real_pair_test/sens --tiers 2 --reps 20 --extra
python features/53_real_pair_test.py --dataset plantwild --features results/10_features_plantwild_bioclip/remote/features_plantwild_bioclip --features-name features_plantwild_bioclip.npy --projection pca --tag extra_bioclip --output results/53_real_pair_test/sens --tiers 2 --reps 20 --extra
python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag n10 --output results/53_real_pair_test/curve --tiers 2 --reps 20 --map-max 10
python analysis/63_review4_summary.py --results results
```
