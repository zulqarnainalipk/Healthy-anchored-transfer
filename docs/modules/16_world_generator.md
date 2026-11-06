# Module 16: synthetic worlds with known truth (secondary benchmark; the primary analysis is the real-pair test, `docs/real_pair_results.md`)

**Compute:** CPU. Runs locally in about 20 seconds.

**Question.** When the relation between a source crop and a target crop is known exactly, does HATC recover it, and where does it fail? This was the primary benchmark in the original plan (proposal section 7.4) and is the secondary analysis in the paper.

**Method.**
- Source samples are real projected features (16 dimensions, module 08) from the training split of PlantVillage and PlantSeg, for every crop-disease group with at least 30 training images.
- Target samples are generated from source samples through each world's known operator, so the truth is exact.
- Per replicate: healthy samples of 50 or 200 images, and disease samples of 10 or 50 images per class. 200 replicates per cell (proposal 7.4).

**Worlds** (`results/16_worlds/`):

| World | Operator (target from source) | Expected outcome |
|---|---|---|
| a | affine map M x + t | affine map holds; a tie with the best simple baseline is expected |
| a2 | additive shift x + s | not an affine transport; operator transfer expected to win |
| b | radial: m + r (x − m), r = 1.3 | hypothesis N holds exactly; best case only |
| b2 | a different random affine per source crop | maps do not compose; the agreement check should fire |
| c25, c50, c100 | affine plus λ tanh(x), λ = 0.25, 0.5, 1 | graded departure from N |
| d | common wrong shift on all sources; the target truth is the identity | common-mode failure; the check must fail here, and the miss rate is reported |
| e | rotation and shear, not a transport map | general affine registration expected to win |
| f | real features with a known additive shift | known operator; a check of the pipeline |

**Output.** One archive per world (`world_<name>.npz`) with the true operator (as JSON) and the first replicate's samples. The full replicate set is generated on demand by the same code with the fixed seed.

**Operator strength.** Every operator is scaled by a strength `s`. The first draft (s = 1) was too weak: no adaptation already scored 78% in world a. The first calibration gave 3, but that was found with a train-test leak; the strength used in the benchmark is **2**, chosen on the no-adaptation score alone (smallest strength with no-adaptation accuracy between about 0.40 and 0.65). See `docs/modules/17_benchmark_hatc.md`; the benchmark commands in the other module documents use `--scale 2`.

**Caveats.**
- The operators above are a first draft, chosen to cover the world types in proposal 7.4. Their exact parameters are random draws from a fixed seed, not tuned. The team must review the definitions before any benchmark result is read.
- Sample sizes are drawn from the real healthy and disease counts (healthy 50 and 200; disease 10 and 50 per class), as the proposal requires.
- The minimum detectable difference and power per world are computed in module 36b, after the replicate outputs exist.

**How to re-run:**
```bash
python features/16_world_generator.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --output results/16_worlds
```
