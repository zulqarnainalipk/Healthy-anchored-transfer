# Module 38: exact sign test (descriptive only)

> **Numbers in this document are from the first pass (unharmonised class names).** The final numbers are in [harmonised_results.md](../harmonised_results.md); where they disagree, that file is right.


**Compute:** CPU. Runs locally in under a second.

**Question.** In how many crops is HATC better or worse than the best simple baseline, and what would an exact test say? The proposal reports the exact test as **descriptive only** (7.7), because it assumes independent pairs, and crops that share source samples are not independent. No inference is drawn from it.

**Method.** For each world and healthy-image cell, the mean accuracy difference HATC minus the best simple baseline is computed per crop. An exact two-sided sign test across crops (ties dropped) gives a p-value. The **smallest achievable p-value** at 7 crops is 2/2⁷ = **0.0156**, so the test can never give a p-value below 0.0156 at this sample size.

**Results** (`results/38_exact_test/sign_tests.csv`; 7 crops per cell):

| Pattern | Cells |
|---|---|
| HATC worse in all 7 crops (p = 0.0156, the smallest possible) | a2, b, d, f (both sizes); c25, c50 at n = 50, 10 cells |
| HATC worse in 6 of 7 crops (p = 0.125) | b2 (both sizes), c100 at n = 50, 3 cells |
| HATC worse in 5 of 7 crops (p = 0.45) | c100 and c50 at n = 200; e at n = 200 better in 5, 3 cells |
| Split between better and worse (p = 1.0) | a (both sizes), c25 at n = 200, e at n = 50, 4 cells |

(Counts of cells: 10 + 3 + 3 + 4 = 20.)

**Reading.** Where the sign test has any power, it points the same way as the bootstrap (module 37): HATC is worse than the best simple baseline in the shift, radial and graded worlds at 50 images, in every crop. It is mixed in the affine world (a) and the rotation world (e), where HATC is better in 3 to 5 of 7 crops. These are descriptive counts only.

**How to re-run:**
```bash
python features/38_exact_test.py --replicates results/20_align_baselines/replicate_results.csv --output results/38_exact_test
```
