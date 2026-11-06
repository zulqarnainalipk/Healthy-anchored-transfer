"""Module 01: crop and disease table with corrected labels.

Reads the audit inventory (paths and metadata only, not the images) and writes:
  - labelled_images.csv     one row per image, with label, crop, role and eligibility
  - crop_disease_table.csv  counts per dataset, crop and disease, split by healthy/disease
  - label_rules.md          the rules applied, with counts per rule

Rules (from the audit findings, results/code_audit_result/audit_findings.md):
  P1  PlantSeg masks (annotations/, PNG) are excluded. Only images/ JPG photos are used.
  P2  Labels are read from file names where folders do not carry them:
        chili_yellow_leaf  : Health_Leaf -> healthy, Yellow_Leaf -> disease (binary target)
        plantseg_v6        : disease from the file-name stem, source tag removed
  P3  fig_sartak_sfld: only SFLD/Original is eligible for evaluation (augmented copies are not).
  Other datasets keep their folder labels.

CPU module. Reads metadata only; no image is opened.

Usage:
    python 01_crop_disease_table.py --inventory path/to/inventory.csv --output path/to/out
"""

import argparse
import collections
import csv
import json
import os
import re
import time
from pathlib import Path

SOURCE_TAGS = re.compile(r"_(google|Bing|Baidu|Yahoo|Flickr|web|Web)$", re.IGNORECASE)
PLANTSEG_META = {}  # image file name -> Metadata.csv row, filled in main()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parts_after_dataset(path):
    """Return the path parts below the dataset folder name."""
    p = path.split("/")
    return p[1:]  # the first part is the dataset folder name


def plantseg_label(stem):
    base = SOURCE_TAGS.sub("", stem)
    base = re.sub(r"[_\-\s]*\d+$", "", base)  # trailing photo numbers
    return base


def classify(row):
    """Return (role, crop, disease, is_healthy, eligible, rule) for one inventory row."""
    ds = row["dataset"]
    path = row["path"]
    parts = parts_after_dataset(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    ext = row["ext"].lower()

    if ds == "plantseg_v6":
        if "annotations" in parts or ext == ".png":
            return "mask", "", "", "", 0, "P1 mask excluded"
        meta = PLANTSEG_META.get(os.path.basename(path))
        if meta is None:
            return "image", "", plantseg_label(stem), 0, 1, "P2 PlantSeg not in Metadata.csv (file-name fallback)"
        crop = meta["Plant"].strip().lower()
        disease = meta["Disease"].strip().lower()
        healthy = int("healthy" in disease)
        return "image", crop, disease, healthy, 1, "P2b PlantSeg label from Metadata.csv"

    if ds == "chili_yellow_leaf":
        healthy = int(stem.startswith("Health"))
        label = "healthy" if healthy else "yellow_leaf"
        return "image", "chili", label, healthy, 1, "P2 chili label from file name (binary)"

    if ds == "fig_sartak_sfld":
        is_original = "Original" in parts
        disease = parts[-2].lower() if len(parts) >= 2 else ""
        healthy = int("healthy" in disease)
        return "image", "fig", disease, healthy, int(is_original), "P3 fig: Original only" if is_original else "P3 fig augmented, not eligible"

    if ds == "rice_leaf_disease":
        disease = parts[-2] if len(parts) >= 2 else ""
        healthy = int(disease.lower() == "healthy")
        return "image", "rice", disease, healthy, 1, "folder label"

    if ds == "peach_leaf_damage":
        disease = parts[-2] if len(parts) >= 2 else ""
        healthy = int(disease == "healthy")
        return "image", "peach", disease, healthy, 1, "folder label (damage probe)"

    if ds == "plantvillage_color":
        folder = parts[-2] if len(parts) >= 2 else ""
        if "___" in folder:
            crop, disease = folder.split("___", 1)
        else:
            crop, disease = folder, folder
        healthy = int("healthy" in disease.lower())
        return "image", crop, disease, healthy, 1, "folder label (crop___disease)"

    return "other", "", "", "", 0, "dataset not in rules"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plantseg-metadata", default="reference/plantseg_v6/Metadata.csv")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 0/4: reading PlantSeg Metadata.csv")
    if Path(args.plantseg_metadata).exists():
        with open(args.plantseg_metadata, encoding="utf-8-sig", newline="") as f:
            for m in csv.DictReader(f):
                PLANTSEG_META[m["Name"]] = m
        log(f"  {len(PLANTSEG_META)} PlantSeg rows loaded")
    else:
        log("  Metadata.csv not found; PlantSeg uses file-name fallback")

    log("step 1/4: reading inventory")
    with open(args.inventory, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["is_image"] == "1" or r["ext"] == ".png"]
    log(f"  {len(rows)} image and mask rows")

    log("step 2/4: applying label rules")
    labelled = []
    rule_counts = collections.Counter()
    for i, r in enumerate(rows, 1):
        if i % 20000 == 0:
            log(f"  processed {i}/{len(rows)}")
        role, crop, disease, healthy, eligible, rule = classify(r)
        rule_counts[(r["dataset"], rule)] += 1
        labelled.append({
            "dataset": r["dataset"],
            "path": r["path"],
            "role": role,
            "crop": crop,
            "disease": disease,
            "is_healthy": healthy,
            "eligible_for_evaluation": eligible,
            "rule": rule,
        })

    log("step 3/4: writing labelled_images.csv")
    fields = ["dataset", "path", "role", "crop", "disease", "is_healthy", "eligible_for_evaluation", "rule"]
    with open(out / "labelled_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(labelled)

    log("step 4/4: writing crop_disease_table.csv and rules")
    table = collections.Counter()
    for r in labelled:
        if r["role"] != "image" or not r["eligible_for_evaluation"]:
            continue
        table[(r["dataset"], r["crop"], r["disease"], r["is_healthy"])] += 1
    with open(out / "crop_disease_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "crop", "disease", "is_healthy", "images"])
        for (ds, crop, dis, h), c in sorted(table.items()):
            w.writerow([ds, crop, dis, h, c])

    rules_lines = ["# Label rules applied (module 01)", "", "| dataset | rule | rows |", "|---|---|---|"]
    rules_lines += [f"| {ds} | {rule} | {c} |" for (ds, rule), c in sorted(rule_counts.items())]
    (out / "label_rules.md").write_text("\n".join(rules_lines) + "\n", encoding="utf-8")

    manifest = {
        "module": "01_crop_disease_table",
        "inventory": args.inventory,
        "rows_in": len(rows),
        "rows_out": len(labelled),
        "eligible_images": sum(1 for r in labelled if r["role"] == "image" and r["eligible_for_evaluation"]),
        "masks_excluded": sum(1 for r in labelled if r["role"] == "mask"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {manifest}")


if __name__ == "__main__":
    main()
