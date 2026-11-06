# Module 03b: label harmonisation inside a crop (found by the failure-case module)

**Compute:** CPU. Switch: environment variable `HARMONISE=1` (read by modules 12, 20, 26, 43, 46b, 46d, 50 through `16_world_generator.harmonise_samples`).

## What was wrong

The failure-case module (46d) listed, as the most frequent confusion of HATC in world a, `potato Early_blight → potato early blight` (147 cases). These are **the same disease under two names**: PlantVillage writes `Early_blight`, PlantSeg writes `potato early blight`. Pooling both datasets as sources split one disease into two classes inside a crop. The same happens for apple (scab, rust), grape (black rot), potato (early and late blight), and tomato (bacterial spot, early blight, late blight, leaf mold, septoria leaf spot, yellow leaf curl virus). Every benchmark module (17 to 20, 26, 30, 31, 35, 41 to 44, 46b, 50) therefore scored a method as wrong when it picked the right disease under the other name. The checklist's label-harmonisation step (original module 03) was replaced earlier by the background check and never done.

## Rule

`canonical_class(crop, class)`: lower case, underscores to spaces, crop prefix removed, plus two aliases (`cedar apple rust` = `rust`, `bacterial leaf spot` = `bacterial spot`). Groups of the same crop with the same harmonised name are merged before the disjoint split. 90 crop-disease groups become 78. Names that are not clearly the same disease are not merged (for example grape `Leaf_blight_Isariopsis_Leaf_Spot` and `grape leaf spot`; `Esca_Black_Measles` and `grape black rot`).

## Sensitivity results (harmonised; `results/03b_harmonised/`)

Main benchmark, healthy n = 200 (module 20; before → after, mean accuracy):

| World | HATC before | HATC after | Best simple after | Best method after |
|---|---|---|---|---|
| a | 0.610 | 0.642 | STD 0.644 | STD |
| b2 | 0.581 | 0.654 | NA 0.638 | HATC |
| c50 | 0.552 | 0.599 | STD 0.601 | STD |
| e | 0.339 | 0.382 | NA 0.326 | CORAL 0.393 |

- Accuracy of every method rises by about 3 to 9 points; HATC rises more than the simple methods and now ties the best simple method in most worlds.
- **Equivalence outcomes** (module 37, 20 cells, exploratory): before, HATC worse in 12 and inconclusive in 8; after, **worse in 5, inconclusive in 15**, none equivalent, none better.
- **Metrics engine** (46b, healthy n = 200, world a): HATC 0.688, S50 0.705, POOL5 (5 labels per class) 0.673, POOL25 0.743. **5 labelled images per class no longer beat HATC in world a**, so the labelled-saving conclusion of modules 26 and 31 (5 labels per class in all 20 cells) was partly an artefact of split classes. The harmonised probe run (module 26) is in progress; until it finishes the labelled-saving number should not be quoted.

## Consequence

The harmonised numbers are the correct ones: a disease that appears under two names is one class. The published results of earlier modules, the proposal's status note and the checklist were written from the unharmonised runs. They are kept as the first pass, and the harmonised rerun is being propagated module by module (done: 20, 36, 37, 46b; running: 26; not yet rerun: 17, 18, 30, 31, 35, 41 to 44, 50). Until then, claims that depend on class counts (HATC loses to simple methods, labelled saving, chili and peach are not affected) must carry the caveat.

## Limits

The merge rule is name-based; two aliases were added by hand; a few same-disease pairs may remain split (grape leaf spot), and a few merged pairs might not be identical in the two datasets' definitions (for example `Early_blight` and `early blight` on different leaf appearances).
