# Module 08b: audit of PlantWild v1 for real source-target pairs

**Compute:** CPU, about 10 minutes (`data_audit_code/audit_plantwild.py`). **Data:** PlantWild v1 (Wei et al., ACM MM 2024; Hugging Face `uqtwei2/PlantWild`, `plantwild.zip`; CC BY-NC-ND 4.0, research use, not redistributed). The data folder is outside the repository.

**Question.** Does PlantWild provide usable real crop pairs (two crops with the same disease, healthy images of the target crop)?

**Results** (`results/plantwild_audit/`)
- 89 class folders matching `classes.txt`; 18,542 readable images; none unreadable; median shorter image side 498 px, smallest 36 px.
- **30 healthy crop classes** (7,840 images, smallest 71 images: tobacco) and 59 disease classes (smallest 44 images). The paper states 33 healthy classes; the folders contain 30 classes named "<crop> leaf".
- **Pairs.** Tier 1 (same pathogen species): tomato and potato early blight (227 and 346 images) and late blight (240 and 295): 4 ordered (pair, disease) units. Tier 2 (same disease type, different species; rust 6 crops, mosaic virus 5, downy mildew 4, powdery mildew 3, early blight 3 with celery, black rot, anthracnose, sheath blight, alternaria leaf spot 2 each): 80 ordered pairs, 22 target crops, all passing the count rule (target healthy at least 50, source and target disease at least 30). Tier 3 (generic leaf spot and leaf blight): 92 pairs, weak.
- **Duplicates.** 902 duplicate pairs (367 exact, 535 near with difference-hash distance at most 3); 1,452 images (7.8%) are in at least one pair; 593 pairs cross classes (65 between a healthy and a disease class; 22 between potato and tomato early blight); 309 pairs inside a class. Rule applied before any analysis (`results/plantwild_audit/dedupe_exclude.csv`): remove all members of any duplicate group that spans more than one class, and all but the first path of a group inside one class. This removes 1,214 images and leaves 17,328; no class falls below 40 images.
- **Sources.** The file `url_record.json` lists the page of each image; no class gets more than a third of its images from one domain (largest share 0.32).
- **Limits.** Web images; no plant or photograph identifiers, so near-duplicate frames may remain in different splits; label noise is not estimated; tier 2 pairs have different pathogens.

**How to re-run:**
```bash
python data_audit_code/audit_plantwild.py --data-root <data-root>/plantwild --output results/plantwild_audit
```
