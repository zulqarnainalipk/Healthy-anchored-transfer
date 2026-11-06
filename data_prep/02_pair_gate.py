"""Module 02: pair gate (proposal section 7.1).

Reads the module 01 crop and disease table and counts, for each disease target crop (rice, fig):
  - the diseases it carries,
  - for each disease, the number of other source crops that also carry it (name matched after
    lower-casing and replacing underscores with spaces),
  - the target's healthy image count against the 50 (affine) and 200 (Brenier) thresholds.

Two counts are written, because the proposal's wording allows two units:
  U1  target-disease: a (target crop, disease) qualifies if the disease is on >= 2 other crops
      and the target has enough healthy images.
  U2  target-disease-source: a (target crop, disease, source crop) triple qualifies when the
      source crop carries the disease, and the same target-disease qualifies by U1.

Chili (binary) and peach (probe) are excluded, as the proposal states.

CPU module, metadata only. Runs locally.

Usage:
    python 02_pair_gate.py --table path/to/crop_disease_table.csv --output path/to/out
"""

import argparse
import collections
import csv
import json
import time
from pathlib import Path

TARGET_CROPS = {"rice", "fig"}
TARGET_DATASETS = {"rice_leaf_disease", "fig_sartak_sfld"}
EXCLUDED_DATASETS = {"chili_yellow_leaf", "peach_leaf_damage"}
MIN_OTHER_CROPS = 2
HEALTHY_AFFINE = 50
HEALTHY_BRENIER = 200
GATE_PAIRS = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


STRIP_CROP = False
KNOWN_CROPS = set()


def norm(name):
    """Lower-case, underscores to spaces. With --strip-crop, a leading crop name is removed too
    (sensitivity rule: "peach anthracnose" and "anthracnose" are treated as one disease)."""
    d = name.lower().replace("_", " ").strip()
    if STRIP_CROP:
        for c in sorted(KNOWN_CROPS, key=len, reverse=True):
            if d.startswith(c + " "):
                return d[len(c) + 1:]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strip-crop", action="store_true",
                    help="sensitivity rule: remove the crop name from disease labels before matching")
    args = ap.parse_args()
    global STRIP_CROP
    STRIP_CROP = args.strip_crop
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log(f"step 1/3: reading crop and disease table (strip-crop={STRIP_CROP})")
    with open(args.table, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    log(f"  {len(rows)} rows")
    KNOWN_CROPS.update(r["crop"].strip().lower() for r in rows if r["crop"].strip())

    log("step 2/3: counting source crops per disease and target healthy images")
    # disease name -> set of source crops carrying it (sources only, excluding chili and peach)
    carriers = collections.defaultdict(set)
    healthy_by_crop = collections.Counter()
    images_by_crop = collections.Counter()
    for r in rows:
        crop = r["crop"].strip().lower()
        if r["dataset"] in EXCLUDED_DATASETS:
            continue
        n = int(r["images"])
        if int(r["is_healthy"]):
            healthy_by_crop[crop] += n
            continue
        if crop in TARGET_CROPS:
            continue
        carriers[norm(r["disease"])].add(crop)

    # Targets come only from the target datasets. PlantSeg also contains rice and fig-like
    # crops; those are not the target dataset and are not counted as targets.
    targets = [r for r in rows if r["dataset"] in TARGET_DATASETS and not int(r["is_healthy"])]

    u1 = []
    u2 = []
    for t in sorted(targets, key=lambda r: (r["crop"], r["disease"])):
        crop = t["crop"].strip().lower()
        disease = norm(t["disease"])
        others = sorted(c for c in carriers.get(disease, set()) if c != crop)
        healthy = healthy_by_crop[crop]
        enough = healthy >= HEALTHY_AFFINE
        qualifies = len(others) >= MIN_OTHER_CROPS and enough
        u1.append({
            "target_crop": crop,
            "disease": disease,
            "target_images": t["images"],
            "other_source_crops": len(others),
            "source_crops": "; ".join(others),
            "target_healthy": healthy,
            "affine_ok": int(healthy >= HEALTHY_AFFINE),
            "brenier_ok": int(healthy >= HEALTHY_BRENIER),
            "qualifies_U1": int(qualifies),
        })
        for s in others:
            u2.append({"target_crop": crop, "disease": disease, "source_crop": s,
                       "qualifies_U1": int(qualifies)})

    log("step 3/3: writing pair_gate tables")
    with open(out / "pair_gate_U1_target_disease.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(u1[0].keys()) if u1 else ["target_crop"])
        w.writeheader()
        w.writerows(u1)
    with open(out / "pair_gate_U2_triples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target_crop", "disease", "source_crop", "qualifies_U1"])
        w.writeheader()
        w.writerows(u2)

    n_u1 = sum(r["qualifies_U1"] for r in u1)
    n_u2 = sum(r["qualifies_U1"] for r in u2)
    summary = {
        "module": "02_pair_gate",
        "targets": sorted(TARGET_CROPS),
        "target_disease_rows": len(u1),
        "qualifying_U1": n_u1,
        "qualifying_U2_triples": n_u2,
        "gate_threshold_pairs": GATE_PAIRS,
        "gate_met_under_U1": n_u1 >= GATE_PAIRS,
        "gate_met_under_U2": n_u2 >= GATE_PAIRS,
        "healthy_by_target": {c: healthy_by_crop[c] for c in sorted(TARGET_CROPS)},
    }
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
