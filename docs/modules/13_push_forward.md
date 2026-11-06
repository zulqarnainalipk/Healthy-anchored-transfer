# Module 13: push-forward of source disease models

**Compute:** CPU. Runs locally in about 11 seconds.

**Question.** Given the affine host maps (module 11), what does each source disease look like on a target crop? This is step 3 of the method (proposal section 5).

**Method.**
- Each source disease (from the **training** split of PlantVillage and PlantSeg) is modelled as a two-component Gaussian mixture with shrunk covariances in the 16-dimensional projection (module 08). Diseases with fewer than 60 training images get one component; diseases with fewer than **30** training images are skipped.
- Only source crops with a host map (module 11) are used. Under option 2, PlantSeg crops count as sources when they have a PlantVillage healthy reference; the others have no map and are not HATC sources.
- Each mixture component is pushed through the affine map x → m_B + T (x − m_A). Its mean becomes m_B + T(μ − m_A) and its covariance becomes T Σ T′.

**Results** (`results/13_push_forward/`):
- 34 source disease models fitted, 68 pushed models (34 for rice, 34 for fig).
- Source crops with models: tomato 16, soybean 6, apple 6, blueberry 5, corn 3, cherry 1, pepper 1.

**Note on the minimum of 30 training images.** This value is my choice and is not in the registration. It affects the verified fig anthracnose pair:
- **Blueberry anthracnose has 29 training images, one short of 30, so it is skipped.** It is the only blueberry disease with enough images to be interesting here, so this matters.
- **Peach anthracnose** is not a source here at all: peach has no host map, so it is excluded.
- The team must decide whether the minimum is 30 (as run), or whether it is lowered. The decision is recorded in the registration before the classification results are read.

**Consequence for the verified fig pairs.** Under the current run, fig anthracnose can only be pushed from blueberry, and blueberry anthracnose is skipped. The fig-anthracnose evidence therefore has no pushed model. This must be resolved before module 14 uses it.

**How to re-run:**
```bash
python features/13_push_forward.py --features results/05_features_dinov2/remote/features_dinov2 --labelled results/01_crop_disease_table/labelled_images.csv --projection results/08_projection/projection_k16.npz --maps results/11_host_maps_affine --output results/13_push_forward
```
