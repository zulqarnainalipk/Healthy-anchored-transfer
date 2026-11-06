# Rerun of all synthetic-benchmark modules with harmonised classes

**Scripts.** `analysis/rerun_all_harmonised.ps1` (PowerShell driver) and `analysis/merge_groups.py` (joins the four parallel world groups of modules 26 and 30 into one table each).

**Compute:** CPU, a few hours on one machine (probes and certificate baselines run as four parallel groups). Run from the repository root.

**What it does.**
1. Copies the first-pass result folders to `results/_first_pass_unharmonised/` (skipped for folders already archived), so nothing is lost.
2. Sets the environment variable `HARMONISE=1` (also the default inside the benchmark scripts; `HARMONISE=0` reproduces the first-pass names), which makes the benchmark modules merge the same disease named differently by PlantVillage and PlantSeg inside a crop (module 03b; 90 crop-disease groups become 78).
3. Reruns, in order: 17, 18, 20; 36, 37, 38; 26 and 30 in four groups each (g1 to g4) and merges them; 31; the ablations 43 and the label-noise runs 35; 41, 42; 50 (two seeds), 12, 46b, 46d; and finally the tables and figures (46, 47).
4. Writes per-group logs next to the results as `results/_rerun_*.log` and `.err` (not committed; see `.gitignore`).

**Prerequisites.** The cached features (`results/05_features_dinov2/remote/features_dinov2/`, see `docs/README.md`), the labelled table, and the projections of modules 08 and 08b.

**Merging.** `python analysis/merge_groups.py` has no arguments and rewrites `results/26_linear_probes/probe_curve.csv` and `results/30_certificate_baselines/auroc.csv` from the `g1` to `g4` subfolders. It is deterministic: running it again gives identical files.

**How to re-run:**
```bash
powershell -File analysis/rerun_all_harmonised.ps1
```
The final numbers of this rerun are in `docs/harmonised_results.md`.
