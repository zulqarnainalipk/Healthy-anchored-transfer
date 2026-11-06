# Module 02: pair gate (proposal section 7.1)

**Compute:** CPU. Reads the module 01 table only (metadata). Runs locally in under a minute.

**Question:** how many disease pairs qualify for the real-pair comparison, under the gate in proposal 7.1? If fewer than 25 qualify, the real-pair comparison is reported as a case study (G1).

**Rule.** A target-disease (rice or fig, from the rice and fig datasets only) qualifies if:
1. the same disease is labelled on at least 2 other source crops, and
2. the target has enough healthy images: 50 for the affine map, 200 for the Brenier map.

Chili (binary) and peach (probe) are excluded, as the proposal states.

Two counting units are reported, because proposal 7.1 allows both:
- **U1, target-disease:** one row per disease of a target crop. Qualifies under the rule above.
- **U2, triples:** (target crop, disease, source crop). A target-disease that qualifies contributes one triple per source crop that carries the disease.

**Name matching.** Disease names are matched after lower-casing and replacing underscores with spaces. Two modes are run:
- `exact`: names must match as written. This is the rule as written.
- `strip_crop` (sensitivity): a leading crop name is removed before matching, so "peach anthracnose" and "anthracnose" are the same disease. This is a sensitivity rule, not a decision. Whether two labels are one disease is a biological judgement for label harmonisation (module 03), and must be made by the team.

**Results** (`results/02_pair_gate/`):

| Mode | Target-diseases | Qualifying U1 | Qualifying U2 triples | Gate (25) met |
|---|---|---|---|---|
| exact | 12 (rice 9, fig 3) | 0 | 0 | no |
| strip_crop | 12 | 2 (fig anthracnose, rice brown spot) | 7 | no |

Under strip_crop, the qualifying target-diseases are:
- fig anthracnose: 5 other crops (banana, blueberry, celery, peach, strawberry).
- rice brown spot: 2 other crops (soybean, tobacco).

**Caution.** The same name on different crops can be a different pathogen. Brown spot on rice and on soybean, and anthracnose on fig and on banana, are not necessarily the same disease. Each strip_crop match needs a biological check before it is counted.

**Visual review (team judgement from side-by-side photos, not expert verification):**

| Target | Compared with | Judgement | Note |
|---|---|---|---|
| fig anthracnose | banana anthracnose | same | pathogen not yet checked |
| fig anthracnose | blueberry anthracnose | same | pathogen not yet checked |
| rice brown spot | soybean brown spot | same | pathogen not yet checked |
| rice brown spot | tobacco brown spot | not yet judged | |

**Causal-agent check (literature, from search results; full papers still to be read before citing):**

| Disease | Causal fungus | Sources |
|---|---|---|
| Rice brown spot | *Bipolaris oryzae* (teleomorph *Cochliobolus miyabeanus*) | [PhytopathDB](https://phytopathdb.org/content/bipolaris-oryzae); [2025 review, Plant Pathology](https://hal.inrae.fr/hal-05180127) |
| Soybean brown spot | *Septoria glycines* | [OSU Extension](https://u.osu.edu/osusoybeandisease/foliar-diseases/brown-spot/extended-version); [2025 molecular study](https://revistas.unasam.edu.pe/index.php/Aporte_Santiaguino/article/view/1288) |
| Tobacco brown spot | *Alternaria alternata* | [Wang et al. 2015 (HERO)](https://hero.epa.gov/hero/index.cfm/reference/details/reference_id/4635452); [CORESTA](https://www.coresta.org/abstracts/chemical-control-brown-spot-disesase-tobacco-caused-alternaria-alternata-4849.html) |
| Fig anthracnose | *Colletotrichum* spp.: *C. siamense*, *C. fructicola*, *C. truncatum*, *C. gloeosporioides* | [Malaysia 2023](https://aminer.org/pub/64a660c6d68f896efa0e3d97/colletotrichum-species-associated-with-fig-ficus-carica-l-in-malaysia) |
| Banana anthracnose | *C. musae* (main); also *C. gloeosporioides*, *C. siamense*, *C. plurivorum* | [Vietnam](https://jst.tnu.edu.vn/jst/article/download/9088/pdf); [Sri Lanka](https://jnsfsl.sljol.info/articles/11217) |
| Blueberry anthracnose | *C. fructicola*, *C. siamense*, *C. gloeosporioides* complex, *C. acutatum* complex | [Sichuan 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7559709/) |
| Peach anthracnose | *C. siamense*, *C. fructicola*, *C. fioriniae* (Korea); *C. siamense*, *C. fructicola* (China); *C. nymphaeae*, *C. fioriniae* (US) | Korea: [Lee et al. 2020, Mycobiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC10635213) (read in full). China and US: see search snippets only |
| Strawberry anthracnose | *C. acutatum*, *C. fragariae*, *C. gloeosporioides*; *C. fructicola* (Korea); *C. karstii* (Brazil) | [Strawberry review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2801091) |
| Celery anthracnose | *C. fioriniae* (main), *C. nymphaeae*, *C. theobromicola*; older reports *C. orbiculare* | [Cornell](https://vegetables.cornell.edu/pest-management/disease-factsheets/celery-anthracnose-leaf-curl-disease); [Ontario](https://ontario.ca/page/celery-leaf-curl-anthracnose) |

**Re-check with the fig abstract (fig's species now confirmed on the fig side):**
- Peach (Korea, full text): shares *C. siamense* and *C. fructicola* with fig. **Counts.**
- Blueberry (Sichuan, full text): shares *C. fructicola* and *C. siamense* with fig. **Counts.**
- Strawberry (Xie 2010, full text): *C. acutatum*, *C. gloeosporioides*, *C. fragariae*. None is on fig's confirmed list. **Does not count.** The *C. gloeosporioides* link to fig came from older snippets and is dropped.
- Banana (Sri Lanka, full text): *C. siamense* was present but **did not cause symptoms**. So it does not show a causal link. **Does not count.**
- Celery: no shared species. **Does not count.**

**Scope caveat.** The fig paper's *C. siamense* result is for **fruit rot and stem canker**, and its leaf result is for **leaf blight**. The dataset's fig "Anthracnose" class is a leaf-image label. The evidence therefore supports a shared fungus in fig and peach or blueberry, but it does not show that the leaf anthracnose in the dataset is the same disease. This is a limit the paper must state.

**Count under option 2:** one target-disease (fig anthracnose), with two supporting crops (peach, blueberry), giving two target-source triples. The gate (25) is not met.

**Healthy-source decision (option 2, recorded by the team):** PlantSeg stays a source for disease images. The 12 PlantSeg crops with a PlantVillage healthy reference are source crops. The other 22 are not HATC sources. A lab-to-field control is registered for the 12 affected crops. The earlier pair counts above assume PlantSeg sources and are unchanged by this decision.

**Decision (option 2, recorded by the team):** count only matches supported by the causal-agent check. Rice brown spot matches are rejected (different fungi). Fig anthracnose is counted as one verified case, with peach. Banana, blueberry and strawberry are not counted until their papers are read in full. The gate (25) is not met, so the real-pair comparison is a case study.

**Full-text status (what has actually been read):**

| Source | Status | What the full text (or abstract) confirms |
|---|---|---|
| Lee et al. 2020, peach, Korea (Mycobiology, DOI 10.1080/12298093.2020.1763116; PMC10635213) | **read in full** | Three species from 20 isolates: *C. fructicola*, *C. siamense*, *C. fioriniae*. Identified by morphology and multilocus phylogeny (ITS, TUB2, GAPDH, CHS-1, CAL, ApMat). All three caused lesions on wounded detached peach fruit (100% incidence) |
| Kurera et al. 2023, banana, Sri Lanka (J. Natl. Sci. Found. Sri Lanka, DOI 10.4038/jnsfsr.v51i2.11217) | **read in full (PDF text)** | *C. musae* (8 isolates), *C. plurivorum* (1), *C. siamense* (1). The *C. siamense* isolate **failed to cause anthracnose symptoms** on inoculated banana. So *C. siamense* was present but not pathogenic in this study |
| Liu et al. 2020, blueberry, Sichuan (Pathogens 9:718) | **read in full** (user-supplied PDF) | 74 isolates from 85 samples. *C. fructicola* 66.2%, *C. siamense* 17.6%, plus *C. kahawae*, *C. karstii*, *C. nymphaeae*, *C. sichuanense*. All species caused anthracnose symptoms on blueberry leaves and stems |
| Xie et al. 2010, strawberry, Zhejiang and Shanghai (J. Zhejiang Univ. Sci. B 11(1):61-70) | **read in full** (user-supplied PDF) | 31 isolates: *C. acutatum* 11, *C. gloeosporioides* 10, *C. fragariae* 10. Pathogenicity tests were inconclusive (controls also produced symptoms). This is a primary study, not the strawberry review in the earlier search |
| Fig, Malaysia 2023 (Crop Protection 169:106256, DOI 10.1016/j.cropro.2023.106256) | **abstract and introduction read** (user-supplied text; methods and results not seen) | 39 isolates: *C. siamense* 30, *C. truncatum* 8, *C. fructicola* 1. Pathogenicity: all three cause **leaf blight**; **stem canker and anthracnose fruit rot** are caused by *C. siamense* |
| Banana, Vietnam (*C. musae*) | **not read** | PDF could not be parsed |
| Blueberry, Sichuan 2020 (PMC7559709) | **not read** | blocked by a bot-check page |
| Strawberry review (PMC2801091) | **not read** | blocked by a bot-check page |
| Peach, China 2022 and US | **not read** | search snippet only |

**What this changes:**
- **Fig ↔ peach:** supported by full text on the peach side. The fig species list still rests on a snippet.
- **Fig ↔ banana:** weaker than the search snippet suggested. The only shared species confirmed in full text (*C. siamense*) did not cause disease in that study.
- **Fig ↔ blueberry, strawberry:** not yet confirmed, because those papers could not be read.

**Reading the check:**
- **Rice vs soybean vs tobacco brown spot: different fungi.** Rice is *Bipolaris*, soybean is *Septoria*, tobacco is *Alternaria*. These are different genera, so the brown-spot matches do not count, whatever they look like.
- **Fig anthracnose vs the other anthracnose crops: shared species at species level.** Fig's reported species are *C. siamense*, *C. fructicola*, *C. truncatum* and *C. gloeosporioides*. Each is also reported on:
  - banana (*C. gloeosporioides*, *C. siamense*): supported
  - blueberry (*C. fructicola*, *C. siamense*, *C. gloeosporioides*): supported
  - peach (*C. siamense*, *C. fructicola*): supported
  - strawberry (*C. gloeosporioides*, *C. fructicola*): supported
  - celery (*C. fioriniae*, *C. nymphaeae*, *C. theobromicola*): no shared species reported, so not supported
- **Limits.** Shared species show that the same fungus can cause both diseases. They do not show it is the same disease in every case, and the sources come from specific regions.

Visual similarity is not proof of the same disease. Anthracnose and brown spot are each caused by different fungi on different crops, so each match needs a check of the causal organism before it is counted.

**Consequence.** Under either mode the gate is not met. The proposal then reports the real-pair comparison as a case study on these six datasets. This is the outcome the proposal anticipated (section 9, "Few qualifying pairs"). **Later development:** the real-pair question was answered with PlantWild v1 (module 08b audit, module 53 test), which is now the primary analysis (`docs/real_pair_results.md`); the synthetic benchmark is secondary.

**How to re-run:**
```bash
python data_prep/02_pair_gate.py --table results/01_crop_disease_table/crop_disease_table.csv --output results/02_pair_gate/exact
python data_prep/02_pair_gate.py --table results/01_crop_disease_table/crop_disease_table.csv --output results/02_pair_gate/strip_crop --strip-crop
```
