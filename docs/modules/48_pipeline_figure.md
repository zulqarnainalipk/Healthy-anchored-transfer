# Module 48: pipeline figure (Figure 1 of the paper)

**Compute:** CPU, about 30 seconds in total.

**Files.** `analysis/48_pipeline_panels.py` and `analysis/48b_pipeline_panels_v2.py` build the data panels from the stored DINOv2 features; `figures/fig4_pipeline/build_pipeline.py` assembles the figure (HTML with inline SVG) and renders `fig4_pipeline.png` (3600 px wide) and `fig4_pipeline.pdf` with headless Chrome. Source: `pipeline.html`.

**Content.** (a) real photographs of tomato (source) and fig (target); (b) patch tokens cut from a real leaf, stacked transformer blocks, the 1,536 real feature values of one image, the supervised projection; (c) framed 2-D views of the real 16-d projected features with 2σ Gaussian contours, with the tomato disease models pushed through the affine map into the fig panel and real fig anthracnose test images; (d) the real 16 × 16 map T and the two healthy clouds; (e) class probabilities of one real fig leaf under the three pushed tomato models, and the real agreement check between two source crops (early blight of tomato and of potato, both pushed to fig; discriminator accuracy 0.975 on 792 held-out images, Wilson bound TV ≥ 0.93, outcome: refuse).

**Style.** Okabe-Ito colour-blind-safe palette (Wong, Nature Methods 8(6), 441, 2011): bluish green for the source crop, orange for the target crop, blue for transport and decision, vermillion for new leaves. Arial text, thin outlines, small right-angle arrows. Stock icons: Lucide (ISC licence), used for the projection, tag, check and stop symbols.

**Caveats.** The tomato-to-fig pair is for display only; no pushed model reaches the fig classes (module 13), so the class bars are labelled "Class 1 to 3" and the probabilities (0.24, 0.36, 0.41) are diffuse. The agreement check here compares two source crops, as registered; on this real pair the two sources disagree strongly, so the check refuses.

**How to re-run:**
```bash
python analysis/48_pipeline_panels.py --results results --assets figures/fig4_pipeline/assets
python analysis/48b_pipeline_panels_v2.py --results results --assets figures/fig4_pipeline/assets
python figures/fig4_pipeline/build_pipeline.py
```
