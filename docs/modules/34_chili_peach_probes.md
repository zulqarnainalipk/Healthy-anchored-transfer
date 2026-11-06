# Module 34: chili binary target and peach damage probe, with the background control

**Compute:** CPU. Runs locally in about 5 seconds.

**Question.** How well do the DINOv2 features separate the chili classes (healthy and yellow-leaf) and the peach classes (healthy and damage), and how much of that is a background shortcut? (Proposal sections 4 and 7.5.) The audit found that the chili corner brightness alone predicts the class with 98% accuracy (module 03).

**Method.**
- **Chili:** logistic regression on the 1,536 DINOv2 features, trained on the train split and tested on the test split (module 02b). A brightness-only baseline uses a single corner-brightness threshold.
- **Background-matched control:** healthy and yellow-leaf images are paired by nearest corner brightness (within 5 brightness units), separately in the train and test splits, so brightness no longer separates the classes.
- **Peach:** 5-fold cross-validation (peach has no splits, it is a probe only), with the brightness-only baseline.

## Results (`results/34_chili_peach_probes/probe_results.csv`)

| Dataset | Setting | Test images | Brightness only | Feature probe | Probe AUROC | Chance |
|---|---|---|---|---|---|---|
| Chili | all test images | 657 | **0.983** | **1.000** | 1.000 | 0.50 |
| Chili | background-matched | **24** | 0.542 | 1.000 | 1.000 | 0.50 |
| Peach | 5-fold cross-validation | 182 | 0.490 | 0.901 | 0.950 | 0.59 (majority) |

## Reading, with the caveats that matter

- **Chili is not a valid measure of disease recognition as it stands.** The probe is perfect (1.000) on all test images, and the background shortcut alone gives 0.983. A perfect score on a field dataset is a warning sign.
- **The matched control is too small to settle it.** Matching within 5 brightness units leaves only **24 test images**, because the healthy and yellow-leaf backgrounds barely overlap in brightness. On this subset the brightness baseline falls to chance (0.54), and the probe is still 1.000. That suggests the features carry a signal beyond corner brightness, but 24 images is far too few to conclude, and matching on corner brightness removes only one background cue (colour and texture may remain).
- **Frame leakage is likely.** The chili dataset was built from field video (Zenodo record title). Consecutive frames are almost identical, and the split is random by image, so near-identical frames can sit in both training and test. This can give a perfect score without any generalisation. Plant or video identifiers are not available to group the frames (module 02b). The chili score is therefore likely inflated, and the proposal's statement that chili supports a binary metric is weak.
- **Peach is a genuine signal, with its own caveat.** The probe scores 0.90 against a majority rate of 0.59 and a brightness baseline of 0.49, so the black background is not the source of the signal. The cross-validation is by crop, not by source photograph, so crops from the same photograph may appear in both training and test (182 crops from 122 photographs). A grouped split by photograph would be cleaner, but the photograph identifier is not in the file names.

## Consequence

- Chili results are reported only with the shortcut accuracy (0.98) beside them and the frame-leakage caveat. The chili binary metric is not used as evidence of generalisation.
- A grouped split by video or photograph, and a larger background-matched test set (for example matching by colour as well as brightness), are needed before chili can support any claim.

**How to re-run:**
```bash
python features/34_chili_peach_probes.py --features results/05_features_dinov2/remote/features_dinov2 --background results/03_background_check/background_per_image.csv --output results/34_chili_peach_probes
```
