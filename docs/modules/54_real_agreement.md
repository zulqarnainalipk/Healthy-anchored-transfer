# Module 54: agreement check on the real PlantWild pairs

**Specification:** `docs/registration.md`, section 7. **Code:** `features/54_real_agreement.py`. **Compute:** CPU, about 12 minutes.

For each target crop T and disease type D that at least three crops share, and each pair of source crops (S1, S2): both source disease sets are moved to T by the healthy-anchored map, a logistic discriminator is trained on half of the pushed samples and tested on the other half, and the 95% Wilson lower bound of its accuracy q gives the bound TV >= 2q - 1 (as in module 18). Scores compared: the bound, ensemble disagreement and low confidence; outcome: failure of HATC (mean balanced accuracy over S1 and S2 more than 2 points below NA) and the continuous gain over NA (Spearman correlation).

**Result (108 triples):** only 1 failure, so the AUROC for failure cannot be assessed. Spearman correlation with the gain: bound +0.242 (p = 0.012, wrong sign), ensemble disagreement −0.401, low confidence −0.453 (both p < 0.001); mean bound 0.548.

**How to re-run:** `python features/54_real_agreement.py --features results/09_features_plantwild/remote/features_plantwild --units results/53_real_pair_test/pw_pca/units.csv --projection pca --tag pca --output results/54_real_agreement`
