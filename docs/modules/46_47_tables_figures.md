# Modules 46 and 47: result tables and figures

**Compute:** CPU. Both run locally in under 15 seconds. They read the stored results of earlier modules and recompute nothing.

## Module 46: tables (`analysis/46_tables.py`, outputs in `results/46_tables/`, csv and markdown)

| Table | Content | Source module |
|---|---|---|
| T1 | dataset sizes and augmented-path images | local audit |
| T2 | accuracy of NA, MC, STD, CORAL, HATC and the best method, per world, 200 healthy images | 20 |
| T3 | counts of equivalence outcomes (HATC against best simple method) | 37 |
| T4 | labels per class a probe needs to match HATC | 31 |
| T5 | device components D1 to D4 headline numbers | 39 to 42 |
| T6 | AUROC of the agreement bound, ensemble disagreement and low confidence | 30 |
| T7 | accuracy against healthy n, world a | 43 |
| T8 | HATC-S against HATC on an untouched world seed | 50 |

## Module 47: figures (`analysis/47_figures.py`, outputs in `figures/`, png and pdf)

R1 main accuracy per world; R2 healthy-sample sweep; R3 HATC minus best simple method with 90% intervals (grey band is the ±2 point equivalence margin); R4 labelled probe curve against the HATC level; R5 agreement AUROC; R6 HATC-S minus HATC. Okabe-Ito palette (Wong 2011) and Arial, with hatching and line styles kept so the figures also read in greyscale. (The first version was black and white.) The tables and figures were regenerated from the harmonised results; see `docs/harmonised_results.md`.

## What the tables and figures show

(Numbers below are the harmonised ones, as in `results/46_tables/` and `docs/harmonised_results.md`; an earlier version of this paragraph carried first-pass numbers and was corrected during the repository audit.)

- T3: at 200 healthy images, HATC is worse than the best simple method in 1 cell and inconclusive in 9; at 50 images it is worse in 5 and inconclusive in 5. No cell shows equivalence or a HATC win. All outcomes are exploratory (7 crop clusters, below the registered minimum of 8).
- T2: at 200 healthy images HATC is the best method in world e only.
- Worlds a2, d and f give identical numbers for MC, STD, CORAL and HATC. This is expected, not a bug: they are pure translations, and these four methods are translation-equivariant, so only no adaptation differs.
- T5: D4 refuses in 86 to 94 percent of cases depending on the world; D3 gated (0.641) is close to always-refit (0.649) and above never-refit (0.626).

## Limits

Tables repeat the stored numbers of the modules and carry their limits: synthetic worlds, one projection, exploratory intervals.

**How to re-run:**
```bash
python analysis/46_tables.py --results results --output results/46_tables
python analysis/47_figures.py --results results --output figures
```
