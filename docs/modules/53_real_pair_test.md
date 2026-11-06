# Module 53: real-pair test (PlantWild v1; PlantVillage tomato-potato pilot)

**Specification:** `docs/registration.md`, sections 7 and 8. **Code:** `features/53_real_pair_test.py`. **Results and verification:** `docs/real_pair_results.md` (every number of the paper's real-pair sections). **Compute:** CPU; one run of 84 units with 20 splits takes 10 to 40 minutes depending on the projection and the machine load.

**Design.** A unit is (source crop S, target crop T, disease type D). Task: healthy versus disease D on crop T (three-class for tomato and potato with `--multiclass`). Healthy model = Gaussian of the target healthy images used for the map; disease model = the source disease Gaussian moved by NA, MC, STD, CORAL, HATC, S25, S50, S75; one-class healthy detector (OC) and a logistic probe with 5, 10, 25 labelled target images per class as references. Projection: 16 dimensions fitted without the target crop (leave-one-crop-out; PCA primary, LDA secondary). 20 random splits per unit (map set min(200, half) of the target healthy images; up to 25 labelled images per class; the rest is test). Primary outcome: HATC minus the best simple method (best by the tier mean among NA, MC, STD), two-way cluster bootstrap over target and source crops (10,000 resamples, 90% interval, ±2-point margin).

**Options used for the controls and checks:** `--seed-offset` (split seeds), `--control wrong_disease` (source disease replaced by another disease of the source crop), `--exclude-file` (stricter duplicate rule, module 58), `--map-max 50`, `--features-name` (BioCLIP), `--tiers 1 2 3`, `--domain-match <url_record.json>` (keep only images from source domains shared by the healthy and the disease class of a crop). Numerical failures are recorded in the run log (none occurred in the final runs).

**Headline (DINOv2, PCA, tier 2: 80 units, 22 target and 22 source crops):** HATC − MC +7.1 points (90% interval 4.5 to 10.0); CORAL +7.4; one-class detector +3.3; probe with 5 labels +8.3. With BioCLIP: HATC − MC +2.2 (0.5 to 4.0). See `docs/real_pair_results.md` for the full tables and the verification.

**Pilot on PlantVillage (tomato and potato, descriptive):** used to test the code before the PlantWild features existed; potato → tomato early blight HATC 0.807 against MC 0.740, late blight 0.904 against STD 0.800; tomato → potato near 0.96 to 0.99 for all methods (`results/53_real_pair_test/pilot_*`).

**How to re-run:**
```bash
python features/53_real_pair_test.py --dataset plantwild --features results/09_features_plantwild/remote/features_plantwild --projection pca --tag pw_pca --output results/53_real_pair_test --tiers 1 2 --reps 20
python features/53_real_pair_test.py --dataset plantwild --features ... --projection lda --tag pw_lda --output results/53_real_pair_test --tiers 1 2
python features/53_real_pair_test.py --dataset plantwild --features ... --projection pca --tag pw_pca_seed1 --output results/53_real_pair_test --tiers 1 2 --seed-offset 100000
python features/53_real_pair_test.py --dataset plantwild --features results/10_features_plantwild_bioclip/remote/features_plantwild_bioclip --features-name features_plantwild_bioclip.npy --projection pca --tag pw_bioclip_pca --output results/53_real_pair_test --tiers 1 2
python features/53_real_pair_test.py --dataset plantwild --features ... --projection pca --tag pw_pca_wrongdisease --output results/53_real_pair_test --tiers 2 --control wrong_disease
```

**Options added in the second round of post-hoc checks (post hoc, registration section 9):** `--cov {shrink,lw,oas,lowrank}` with `--lam` and `--rank` (covariance estimator for every Gaussian), `--select` (HATC-S: the alignment chosen from the target healthy images of the map set only; writes `select_counts.json`). Results are summarised by module 62.

**Option added in the third round of post-hoc checks (post hoc, registration section 10):** `--extra` adds EIG (rotation fixed by eigenvector matching of the healthy covariances) and the unlabelled-target reference arms UDA_CORAL and UDA_BW (pooled healthy and diseased source images against the pooled unlabelled target map and test images; transductive). Runs: `sens/extra_dino`, `sens/extra_bioclip`; the learning curve uses `--map-max 10 25 100` (folder `curve/`) with `pw_pca_n50` and `pw_pca`. Summarised by module 63.
