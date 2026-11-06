# Module 04b: pixel-level verification of the closest target-source pairs

**Compute:** CPU, local. Needs the images (read only) for the 200 candidate pairs.

**Question.** Module 04 ranks target-source pairs by encoder similarity (see `04_near_duplicates.md`). Is any of the closest pairs a real copy? The check does not depend on the encoder.

**Method.** The top 200 candidate pairs from `results/04_near_duplicates/nearest.csv`; both images at 64 x 64 grey, standardised; correlation of the target with the source under the 8 flips and 90-degree rotations; a pair is a copy if the correlation is at least 0.9. Side-by-side sheets can be written to a folder outside the repository, because they contain dataset images.

**Result** (`results/04b_verify/verify_summary.txt`): 200 candidates, 0 pixel copies, highest pixel correlation 0.8041.

**Limit.** Only the closest 200 pairs are checked; copies that the encoder ranks lower are not.

**How to re-run:**
```bash
python features/04b_near_duplicate_verify.py --nearest results/04_near_duplicates/nearest.csv --data-root <data-root> --output results/04b_verify --top 200 --sheet-dir <folder-outside-the-repository>
```
