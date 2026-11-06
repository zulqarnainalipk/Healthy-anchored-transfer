# Fig augmentation check (audit item D3)

**Compute:** CPU. Reads the fig image folder (read only). Runs locally in about 25 minutes.

**Question:** are the 8,148 augmented fig images copies of the 2,037 Original images? If so, they add no new information, and evaluation must use only the Originals.

**Method.**
- Each image is converted to 48 × 48 grayscale and normalised to zero mean and unit variance, which removes brightness and contrast change.
- Each Augmented image is compared with the Original of the same class and number (file names are `Class_NNNN_Transform.png`).
- The Original is rotated in 5-degree steps and flipped. The highest correlation is kept. A match is a correlation of at least 0.9.

**Result** (`results/fig_augmentation_check/`):

| Class | Augmented | Matched | Unmatched |
|---|---|---|---|
| Anthracnose | 1,792 | 1,786 | 6 |
| Cercospora leaf spot | 2,100 | 2,089 | 11 |
| Healthy | 2,052 | 2,044 | 8 |
| Mosaic | 2,204 | 2,185 | 19 |
| **Total** | **8,148** | **8,104** | **44** |

- Matches by transform: 2,037 with a flip, 6,067 without. The flip count equals the number of Originals.
- The 44 unmatched images score 0.78 to 0.87, just below the 0.9 threshold. They are most likely rotated copies whose corners were cut off by the rotation. This is not yet confirmed.

**Conclusion.** The augmented fig images are derived from the Originals, so evaluation uses only the 2,037 Original images. The split file (module 02b) already excludes the augmented copies.

**How to re-run:**
```bash
python data_audit_code/fig_augmentation_check.py --input "<data-root>/fig_sartak_sfld" --output results/fig_augmentation_check
```
