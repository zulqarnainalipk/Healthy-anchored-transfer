"""Module 02b: split files grouped by source photo, plant or exact duplicate.

Inputs:
  - results/local_audit/inventory.csv        (paths and md5 hashes, from the local audit)
  - results/01_crop_disease_table/labelled_images.csv   (role and eligibility, from module 01)
  - reference/plantseg_v6/Metadata.csv        (PlantSeg official split and photo names)

Rules (proposal section 4, "Grouping"):
  - PlantSeg: use the official Split column of Metadata.csv (Training, Validation, Test).
    Each PlantSeg photo is its own group, so no photo spans two splits.
  - PlantVillage and rice: group = exact duplicate (md5), so copies of one file never span
    two splits. Groups are assigned to train, validation and test at 80/10/10 with a fixed seed.
    No plant IDs exist, so no plant-level claim is made.
  - Fig: Original folder only (eligible). Group = mosaic number (one source photo and its
    augmented copies share the number). Assigned 70/15/15 by group.
  - Chili: group = exact duplicate. 80/10/10. No plant IDs.
  - Peach: a probe, not trained on. Group = file stem before the annotation suffix. All rows
    are marked split = probe.
  - Masks are excluded.

Outputs in results/02b_splits/:
  - splits.csv              one row per eligible image: path, dataset, group, split
  - split_summary.csv       counts per dataset and split
  - manifest.json           SHA-256 of splits.csv, seed, rule version, leakage check result

CPU module, metadata and hashes only. Runs locally.

Usage:
    python 02b_splits.py --inventory results/local_audit/inventory.csv \
        --labelled results/01_crop_disease_table/labelled_images.csv \
        --plantseg-metadata reference/plantseg_v6/Metadata.csv --output results/02b_splits
"""

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path

SEED = 2026
RULE_VERSION = "02b-1"
FIG_MOSAIC = re.compile(r"Mosaic_(\d+)", re.IGNORECASE)
PEACH_ANN = re.compile(r"_ann\d+$", re.IGNORECASE)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assign_groups(groups, fractions, rng):
    """Assign each group id to a split by the given fractions (dict split -> fraction)."""
    ids = sorted(groups)
    rng.shuffle(ids)
    n = len(ids)
    out = {}
    cut = 0
    names = list(fractions)
    for i, name in enumerate(names):
        take = round(fractions[name] * n) if i < len(names) - 1 else n - cut
        for gid in ids[cut:cut + take]:
            out[gid] = name
        cut += take
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--plantseg-metadata", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    log("step 1/5: reading inventory (paths, md5) and labelled role table")
    md5 = {}
    with open(args.inventory, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            md5[r["path"]] = r["md5"]
    with open(args.labelled, encoding="utf-8") as f:
        labelled = [r for r in csv.DictReader(f) if r["role"] == "image"]
    log(f"  {len(labelled)} image rows, {len(md5)} hashed files")

    log("step 2/5: reading PlantSeg Metadata.csv for the official split")
    meta = {}
    with open(args.plantseg_metadata, encoding="utf-8-sig", newline="") as f:
        for m in csv.DictReader(f):
            meta[m["Name"]] = m
    split_map = {"Training": "train", "Validation": "val", "Test": "test"}
    log(f"  {len(meta)} PlantSeg rows")

    log("step 3/5: assigning groups and splits")
    rows = []
    # Groups that need random assignment, per dataset.
    random_groups = collections.defaultdict(set)
    for r in labelled:
        if r["eligible_for_evaluation"] != "1" and r["dataset"] != "plantseg_v6":
            continue
        ds, path = r["dataset"], r["path"]
        if ds == "plantseg_v6":
            m = meta.get(Path(path).name)
            if m is None:
                continue
            gid = "photo:" + Path(path).name
            rows.append({"path": path, "dataset": ds, "group": gid,
                         "split": split_map[m["Split"]]})
        elif ds == "fig_sartak_sfld":
            m = FIG_MOSAIC.search(Path(path).name)
            gid = "mosaic:" + (m.group(1) if m else Path(path).stem)
            rows.append({"path": path, "dataset": ds, "group": gid, "split": None})
            random_groups[ds].add(gid)
        elif ds == "peach_leaf_damage":
            gid = "peach:" + PEACH_ANN.sub("", Path(path).stem)
            rows.append({"path": path, "dataset": ds, "group": gid, "split": "probe"})
        else:
            gid = "md5:" + md5.get(path, Path(path).name)
            rows.append({"path": path, "dataset": ds, "group": gid, "split": None})
            random_groups[ds].add(gid)

    log("step 4/5: assigning random splits to groups")
    assigned = {}
    fractions = {
        "plantvillage_color": {"train": 0.8, "val": 0.1, "test": 0.1},
        "rice_leaf_disease": {"train": 0.8, "val": 0.1, "test": 0.1},
        "chili_yellow_leaf": {"train": 0.8, "val": 0.1, "test": 0.1},
        "fig_sartak_sfld": {"train": 0.7, "val": 0.15, "test": 0.15},
    }
    for ds, groups in random_groups.items():
        log(f"  {ds}: {len(groups)} groups")
        assigned[ds] = assign_groups(groups, fractions[ds], rng)
    for r in rows:
        if r["split"] is None:
            r["split"] = assigned[r["dataset"]][r["group"]]

    log("step 5/5: leakage check and writing outputs")
    seen = collections.defaultdict(set)
    for r in rows:
        seen[(r["dataset"], r["group"])].add(r["split"])
    leaks = [k for k, v in seen.items() if len(v) > 1 and "probe" not in v]
    if leaks:
        raise SystemExit(f"leakage: {len(leaks)} groups span splits, e.g. {leaks[:3]}")

    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "dataset", "group", "split"])
        w.writeheader()
        w.writerows(rows)

    counts = collections.Counter((r["dataset"], r["split"]) for r in rows)
    with open(out / "split_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "images"])
        for (ds, sp), c in sorted(counts.items()):
            w.writerow([ds, sp, c])

    manifest = {
        "module": "02b_splits",
        "rule_version": RULE_VERSION,
        "seed": SEED,
        "rows": len(rows),
        "splits_csv_sha256": sha256_of(out / "splits.csv"),
        "leakage_check": "passed: no group spans two splits (probe rows excluded)",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {manifest}")


if __name__ == "__main__":
    main()
