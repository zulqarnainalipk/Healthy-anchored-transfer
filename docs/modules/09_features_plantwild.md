# Modules 09 and 10: frozen features of PlantWild (DINOv2 and BioCLIP)

**Code:** `features/09_features_plantwild.py` (`--encoder dinov2` or `bioclip`). **Run time:** about 1 to 2 hours for 17,328 images (the DINOv2 run took 6,874 s and the BioCLIP run 4,040 s).

**Recipe.** The images of the 89 class folders, minus the 1,214 images of the duplicate rule (`results/plantwild_audit/dedupe_exclude.csv`, module 08b), are resized to 224 × 224 (full frame, bicubic). DINOv2: model `facebook/dinov2-base` pinned to revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`, ImageNet normalisation, features = [CLS token | mean patch token] (1,536 values, float16), float32 on the CPU. BioCLIP: open_clip `hf-hub:imageomics/bioclip` (latest commit at run time), CLIP normalisation, 512-value image embedding. Both runs use the same index (`features_index.csv`: row, path, class, role, crop, disease, split mode of the authors' file).

**Verification.** 17,328 rows each, none unreadable, none non-finite, identical index; 40 random DINOv2 vectors recomputed locally with the pinned model match the stored ones (minimum cosine similarity 1.0000, largest relative difference 0.00022; module 57). Outputs are stored in `results/09_features_plantwild/remote/` and `results/10_features_plantwild_bioclip/remote/` (not in git; checksums in the manifests).

**How to re-run:**
```bash
python features/09_features_plantwild.py --data-root <data>/plantwild --exclude results/plantwild_audit/dedupe_exclude.csv --output results/09_features_plantwild/remote/features_plantwild
python features/09_features_plantwild.py --data-root <data>/plantwild --exclude results/plantwild_audit/dedupe_exclude.csv --output results/10_features_plantwild_bioclip/remote/features_plantwild_bioclip --encoder bioclip
```
