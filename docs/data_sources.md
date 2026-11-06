# Data sources, licences and redistribution

**Policy.** The repository holds no dataset image and no copy of any dataset file (exceptions: open points 2 and 7). All seven datasets (six crop datasets and PlantWild v1) are public; the user downloads them from the records below. What the repository does hold, derived from the datasets, is: per-image tables (file paths relative to the dataset folder, labels, split, a content hash used for grouping), cached model features (published as an archive, not in the git history; see `docs/README.md`), and summary results. Two exceptions need the owner's attention and are listed under "Open points".

How to obtain the data: download each record, unpack it into one folder per dataset under a data root of your choice, and keep the folder names below (the stored tables use them as the first path component).

| Folder name | Dataset | Record | Licence stated on the record | Images used |
|---|---|---|---|---|
| `plantvillage_color` | PlantVillage, colour images (Hughes and Salathé 2015; Mohanty et al. 2016) | Zenodo record 1204914 (DOI 10.5281/zenodo.1204914) | Record page: CC BY 4.0. The dataset's own `LICENSE` file and the record text: CC BY-SA 3.0 US, "algorithms trained on the data fall under the same license" | 54,306 |
| `plantseg_v6` | PlantSeg, version 6 (Wei et al.) | Zenodo record 16665770 (DOI 10.5281/zenodo.16665770) | Record page: CC BY 4.0. Per-image licence column in `Metadata.csv`: CC0 for 10,325 images, CC-BY for 1,133 | 11,458 (segmentation masks excluded) |
| `fig_sartak_sfld` | Sartak Fig Leaf Dataset (SFLD) | Mendeley Data, DOI 10.17632/sby6kvrvvv.1 | CC BY 4.0 (record and `LICENSE.txt`) | 2,037 originals (8,148 augmented copies excluded from every evaluation) |
| `rice_leaf_disease` | Rice leaf-disease dataset | Mendeley Data, DOI 10.17632/n8jpkpfz5p.1 | CC BY 4.0 | 6,712 |
| `chili_yellow_leaf` | Chili yellow-leaf dataset | Zenodo record 23322122 (DOI 10.5281/zenodo.23322122) | CC BY 4.0 | 6,566 |
| `peach_leaf_damage` | Peach orchard leaf-damage dataset (Jumilla, Murcia, Spain) | Zenodo record 23475740 (DOI 10.5281/zenodo.23475740) | CC BY 4.0 | 182 leaf crops |
| `plantwild` | PlantWild v1 (Wei et al., ACM Multimedia 2024): web images of 30 healthy crop classes and 59 disease classes. **Basis of the primary real-pair test** (modules 08b, 09, 53 to 59) | Hugging Face dataset `uqtwei2/PlantWild`, file `plantwild.zip` (the data folder holds the 89 class folders, `classes.txt`, `url_record.json`) | CC BY-NC-ND 4.0 (research use; non-commercial, no derivatives). Individual images are web images with their own rights | 18,542 readable; 17,328 after the duplicate rule of module 08b (`results/plantwild_audit/dedupe_exclude.csv`) |

The first six rows are the datasets of the synthetic benchmark and the label audit. PlantWild is used only for the real-pair test; none of its images is stored in the repository.

The licences above were read from the record pages (kept in the project's reference folder, outside this repository) and from the licence files shipped with the data. Re-check them on the day of release.

## What is derived from the data and where it is stored

| Item | Location | Contains | Redistribution note |
|---|---|---|---|
| Inventory of every file | `results/code_audit_result/inventory.csv`, `results/local_audit/inventory.csv` | relative path, size, dimensions, colour mode | file facts only |
| Labelled table, splits | `results/01_crop_disease_table/`, `results/02b_splits/` | relative path, crop, disease, split, MD5-based group id | labels are derived from folder or file names of the datasets |
| Cached features | `results/05_features_dinov2/remote/`, `results/06_features_bioclip/remote/`, `results/09_features_plantwild/remote/`, `results/10_features_plantwild_bioclip/remote/` | DINOv2 (1,536 values) and BioCLIP (512 values) per image, float16 | not git-tracked (size); published as an archive, with SHA-256 in `manifest.json`. PlantWild features: see open point 6 |
| PlantWild audit and duplicate list | `results/plantwild_audit/` | per-class counts, duplicate pairs (relative paths), exclusion list | file facts and relative paths only |
| Summary results, figures | `results/`, `figures/` | tables and plots | authors' own work |
| Figure 1 photographs | `figures/fig4_pipeline/assets/src_*.jpg`, `tgt_*.jpg` | 10 leaf photographs (tomato from PlantVillage, fig from SFLD), reduced size | third-party images, see open points |

## Open points for the owner (decisions needed before release)

1. **PlantVillage licence conflict.** The Zenodo record page says CC BY 4.0, but the licence file inside the data and the record text say CC BY-SA 3.0 US with the clause that algorithms trained on the data fall under the same licence. Decide which one the paper cites; the conservative reading is CC BY-SA 3.0 for anything derived directly from these images (the cached PlantVillage features, and the PlantVillage photographs in Figure 1). Whether features or fitted models count as "algorithms trained on the data" is a legal question this audit cannot settle.
2. **Photographs in `figures/fig4_pipeline/assets/`.** Six tomato photographs (PlantVillage, CC BY-SA 3.0) and four fig photographs (SFLD, CC BY 4.0) are in the repository, so the statement "third-party data are not redistributed" is not literally true. Either keep them with attribution (and mark the tomato panels CC BY-SA 3.0), or replace them by placeholders and ship only the built figure.
3. **`reference/plantseg_v6/Metadata.csv`** (2.4 MB) is a copy of the PlantSeg metadata, used by `data_prep/02b_splits.py` (`--plantseg-metadata`). It is excluded by `.gitignore`; download it from the PlantSeg record instead. It includes source URLs of third-party web pages.
4. **Attribution text.** Add the dataset citations (reference list of the paper) to the README release notes; CC BY requires attribution.
5. **Dataset DOIs and dates.** The record pages carry publication dates in autumn 2026 for the chili, peach, fig and rice records; confirm that these are the final versions used (versions are listed in the table above).
6. **PlantWild licence (CC BY-NC-ND 4.0).** Decide whether the PlantWild feature arrays (53 MB DINOv2, 17 MB BioCLIP) may be released in the archive: features are derived data, and "no derivatives" plus "non-commercial" is a legal reading this audit cannot settle. Options: (a) release the features with a non-commercial notice, (b) release only the extraction script and the checksums, so users rebuild them (1 to 2 hours). `url_record.json` (source page of each image) is not in the repository; `results/plantwild_audit/source_urls.csv` holds only per-class domain counts.
7. **PlantWild sample photographs in `results/plantwild_audit/samples/`.** Twelve web photographs (about 0.5 MB) are stored there for the audit report. They are third-party images under CC BY-NC-ND 4.0 with unknown per-image rights. `.gitignore` now excludes this folder; delete it from the working tree if you do not want it in the release archive either.
