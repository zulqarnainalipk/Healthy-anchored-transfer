# Module 64: label-free risk scores, label-noise proxy and replication lists

`analysis/64_risk_noise_release.py` writes `results/64_risk_noise_release/`:

- `risk_proxy_units.csv`, `risk_proxy_correlations.csv`: per tier-2 unit (DINOv2, 3 splits) the Bures-Wasserstein distance of the healthy sets, the S25/S75 prediction disagreement and the false-alarm rate of HATC on unused healthy target images; Spearman correlation with the HATC balanced accuracy and with HATC minus MC (range over leave-one-target-crop-out), and the mean outcome on the two thirds of units with the smaller score.
- `label_noise_proxy.csv`: share of the 10 nearest neighbours (cosine, DINOv2, same crop, leave-one-out) that have the other role, by disease type. A proxy for hard or mislabelled images, not an annotator study.
- `replication/`: `kept_images.csv` (the feature index: 17,328 images with harmonised crop, role, disease, class), `removed_near_duplicates.csv` (1,214 files), `class_name_harmonisation.csv`, and `loco_pca_projection_{dinov2,bioclip}.npz` (mean and 16 PCA axes for each held-out crop, keys `<crop>__mean` and `<crop>__axes`). The lists contain file names only.

How to re-run (CPU, about 4 minutes):

```bash
python analysis/64_risk_noise_release.py --features-dino results/09_features_plantwild/remote/features_plantwild --features-bio results/10_features_plantwild_bioclip/remote/features_plantwild_bioclip --results results/53_real_pair_test --output results/64_risk_noise_release
```
