# Harmonised results (final numbers of the synthetic benchmark; secondary analysis)

**Role in the paper.** The primary analysis is the registered real-pair test on PlantWild (`docs/real_pair_results.md`: HATC − best simple alignment +7.1 points on 80 real units). The synthetic worlds below are the secondary analysis and answer a different question (known truth, controlled shift); on them HATC does not beat the best simple method. The two results are not in conflict: they use different data, different units and different baselines (in the real-pair test CORAL matches HATC and the one-class detector recovers about half of HATC's gain over mean-centring).

All benchmark modules were rerun with harmonised class names (module 03b: the same disease named differently by PlantVillage and PlantSeg inside a crop is one class; 90 crop-disease groups become 78). The module documents written before this rerun contain first-pass numbers; the first-pass result folders are kept in `results/_first_pass_unharmonised/`. **Where a module document and this file disagree, this file is right.** Everything is exploratory (7 crop clusters, registered minimum 8). Tables are also in `results/46_tables/`.

## What changed in the conclusions

| Claim (first pass) | Harmonised result |
|---|---|
| HATC worse than the best simple method in 12 of 20 cells, inconclusive in 8 | **Worse in 6, inconclusive in 14**, none equivalent, none better (exploratory) |
| HATC best in only 3 of 20 cells (module 20) | Best method in **2 of 20** cells (n = 50 and 200 pooled: MC 9, NA 5, STD 4, HATC 2); mean-shift or standardisation or no adaptation wins most cells |
| World e: only place the agreement bound is informative (AUROC 0.86) | **No longer true.** Agreement bound AUROC averages 0.46 (below chance); in world e it is 0.23 |
| Ensemble disagreement 0.46, low confidence 0.31 (worse than the bound) | Ensemble disagreement **0.57**, low confidence 0.38 (0.62 if the sign is flipped); the conventional ensemble score is the best of the three |
| A probe needs 5 labels per class to reach HATC in all 20 cells | **Still 5 labels per class in all 20 cells** (probe at 5 labels: 0.65 to 0.72 against HATC 0.38 to 0.70); the probe is helped by a projection fitted on the same classes |
| D4 refuses in every world (rate 1.00) | Refusal 0.86 to 0.94 (pass 6 to 14%); still refuses almost always |
| D3 gated matches always-refit | Same: gated 0.641 to 0.645 against always-refit 0.648 to 0.649 under drift, with 2.3 to 4.7 refits against 9; never-refit falls to 0.610 at drift 4 |
| Shrunk map S50 and HATC-S beat HATC | Same direction; HATC-S minus HATC (seed 9002, nine candidates): above HATC in 19 of 20 cells, up to +8.3 points, mean +5.7 at n = 50 and +1.5 at n = 200 |

## Main benchmark, healthy n = 200 (mean accuracy; module 20)

| World | NA | MC | STD | CORAL | HATC | Best |
|---|---|---|---|---|---|---|
| a | 0.623 | 0.676 | 0.681 | 0.654 | 0.675 | STD |
| a2, d, f (pure shifts: MC, STD, CORAL, HATC identical across the three) | 0.705 / 0.730 for f | 0.761 | 0.758 | 0.688 | 0.697 | MC |
| b | 0.656 | 0.610 | 0.758 | 0.688 | 0.697 | STD |
| b2 | 0.699 | 0.692 | 0.683 | 0.665 | 0.680 | NA |
| c25 | 0.656 | 0.675 | 0.672 | 0.647 | 0.662 | MC |
| c50 | 0.646 | 0.637 | 0.648 | 0.620 | 0.634 | STD |
| c100 | 0.613 | 0.572 | 0.606 | 0.580 | 0.593 | NA |
| e | 0.331 | 0.307 | 0.346 | 0.377 | 0.379 | HATC |

## HATC against the best simple method (modules 36 to 38)

At n = 200: HATC worse in 1 cell (b), inconclusive in 9. At n = 50: worse in 5 (a2, b, b2, d, f), inconclusive in 5. Mean differences range from −12.8 to +3.3 points; the only positive cells are world e (+2.5 and +3.3 points, intervals above zero but inside the ±2-point margin's reach, so "inconclusive"). The coverage and power check (36b, rerun on the harmonised replicates): between-crop SD 0.089, within-crop SD 0.046; the 90% bootstrap interval covers about 83% at 7 clusters and 89% at 25; power to show a 3-point difference 0.32 / 0.55 / 0.78 at 7 / 25 / 50 clusters; equivalence within ±2 points 0.001 / 0.002 / 0.035, so the registered equivalence test cannot succeed at the planned design.

## Healthy-sample sweep, world a (module 43)

| n | NA | MC | STD | S50 | CORAL | HATC |
|---|---|---|---|---|---|---|
| 10 | 0.620 | 0.668 | 0.657 | 0.637 | 0.491 | 0.486 |
| 50 | 0.619 | 0.670 | 0.682 | 0.707 | 0.649 | 0.659 |
| 200 | 0.617 | 0.672 | 0.679 | 0.707 | 0.683 | 0.694 |
| 400 | 0.618 | 0.671 | 0.678 | 0.712 | 0.689 | 0.697 |

HATC needs about 100 healthy images to overtake mean-centring; the shrunk map S50 is best from n = 25.

## Projection and dimension ablations, n = 200 (HATC minus best simple method)

| World | 4-d | 8-d | 16-d (supervised) | 16-d (PCA) |
|---|---|---|---|---|
| a | −0.013 | 0.000 | −0.007 | **+0.048** |
| c50 | −0.026 | −0.057 | −0.014 | +0.002 |
| e | −0.004 | +0.017 | +0.033 | −0.016 |
| f | −0.019 | −0.049 | −0.064 | −0.024 |

In PCA space HATC beats the simple methods in world a and ties in c50, as in the first pass; strengths are not comparable between projections.

## Label noise (healthy n = 200, 0%, 5%, 10% symmetric source-label noise; HATC / S50 / best simple)

| World | 0% | 5% | 10% |
|---|---|---|---|
| a | 0.694 / 0.707 / 0.679 | 0.648 / 0.678 / 0.618 | 0.627 / 0.643 / 0.588 |
| c50 | 0.677 / 0.684 / 0.672 | 0.627 / 0.643 / 0.629 | 0.605 / 0.611 / 0.608 |
| e | 0.363 / 0.330 / 0.320 | 0.315 / 0.282 / 0.288 | 0.328 / 0.290 / 0.288 |
| f | 0.714 / 0.752 / 0.757 | 0.705 / 0.741 / 0.748 | 0.687 / 0.732 / 0.735 |

Source label noise at 10% costs 6 to 12 points in worlds a and c50 and 1 to 5 points in worlds e and f; the top and bottom of the ranking are unchanged, neighbouring methods swap places. The change in agreement-check AUROC under noise is not computed.

## Agreement check against other failure scores (healthy n = 200; AUROC for failure; module 30)

Mean over the 20 cells: agreement bound 0.46, ensemble disagreement 0.57, low confidence 0.38 (0.62 if the sign is flipped). Per world (n = 200): bound 0.17 to 0.64; best single value of any score is 0.70 (ensemble disagreement, world b).

## Device components (modules 39 to 42)

D1 enrolment: false enrolment 0.86 to 0.94 with the gate, 1.00 without (unchanged). D2 routing: diseased leaves routed to disease 0.93 (82% correct class); healthy leaves answered 0.99; unenrolled-crop leaves answered as healthy 0.76 (unchanged, PlantVillage only). D3: see the table above. D4: refusal 0.86 to 0.94.

## Metrics beyond accuracy (module 46b; n = 200; worlds a, e)

| World | HATC acc / worst class / ECE | S50 | POOL5 | POOL25 |
|---|---|---|---|---|
| a | 0.688 / 0.44 / 0.22 | 0.705 / 0.48 / 0.20 | 0.673 / 0.39 / 0.24 | 0.743 / 0.51 / 0.16 |
| e | 0.373 / 0.10 / 0.57 | 0.336 / 0.06 / 0.64 | 0.542 / 0.16 / 0.44 | 0.628 / 0.33 / 0.31 |

The worst class is 20 to 30 points below mean accuracy; all methods are over-confident (ECE 0.12 to 0.70). SHOT-style and AdaBN-style are within a few points of HATC or below it.

## Brenier branch and diversified predictions (module 12; HATC, BRENIER, BRENIER_AFF, ENS)

n = 200, world a: 0.688, 0.295, 0.670, 0.677; world e: 0.373, 0.327, 0.354, 0.365. The naive empirical map is near chance (a failure of the interpolation); the affine extension is 1 to 2 points below HATC; bootstrap maps and the ensemble do not help.

## Not rerun with harmonised classes

Modules 13 (push-forward on real data, PlantVillage and PlantSeg models per class), 34 (chili and peach probes), 39 and 40 (PlantVillage only or healthy only). Real-data counts in module 13 still use unharmonised class names.

## Second encoder (BioCLIP; module 29)

Same conclusions: HATC is the best method in 0 of 20 cells (STD 9, MC 8, CORAL 2, NA 1); worse than the best simple method in 3 cells at n = 200 and 8 at n = 50 (rest inconclusive, none better); a probe reaches HATC with 5 labels per class in 16 cells and 10 in 4. Operator strength was not recalibrated for this encoder, so absolute accuracies are not comparable. See `docs/modules/29_second_encoder.md`.

## Design checks of the synthetic benchmark (modules 51 and 52; exploratory)

Details and commands: `docs/modules/51_review_checks.md`, `docs/modules/52_persample_refusal.md`.

- **World e excluded** (unidentifiable by design; 18 cells): HATC best in 0 cells (MC 9, NA 5, STD 4); equivalence outcomes 6 different (HATC worse), 12 inconclusive; mean HATC − best simple −6.1 points; probe with 5 labels reaches HATC in all 18; AUROC of the bound, ensemble disagreement, low confidence 0.47, 0.59, 0.38.
- **Projection fitted without the target crop** (LOCO): the ranking of the healthy-only methods is unchanged (HATC below the best simple method in 9 of 10 worlds with LDA, 5 with PCA), but the labels needed to reach HATC rise from 5 to 10 per class (original) to 5 to more than 25 (LOCO LDA) and 25 to more than 25 (LOCO PCA).
- **PCA exception:** disappears when the operator strength is recalibrated for the PCA projection (1.0): HATC below the best simple method in all ten worlds.
- **Shrinkage:** mean HATC − best simple over worlds a, c50, e, f at n = 200: −5.1, −3.2, −0.6, −0.2, −0.4 points for λ = 0.01, 0.03, 0.1, 0.3, 0.5.
- **PlantVillage-only sources:** HATC below the best simple method in 9 of 10 worlds by 1.7 to 10.5 points.
- **ZFGA-style Fisher-geometry baseline** (our reading of the technical note): 0.43 to 0.52 accuracy against 0.64 for no adaptation.
- **Per-image refusal:** classifier confidence AUROC 0.67 to 0.72 (worlds a, c50, f) against 0.54 to 0.57 for likelihood and two-source disagreement; refusing the 20% least confident images adds 5 to 7 points.
- **Encoder revisions:** DINOv2 pinned to f9e44c814b77203eaa57a6bdbbd535f21ede1415 in `features/05_features_dinov2.py` (the original run did not pin it; the weights are unchanged since 2023-09-04); BioCLIP loaded from commit ce901ab3c6a913f9e9ef94ce6d27761069f4f01c (26 February 2026).
