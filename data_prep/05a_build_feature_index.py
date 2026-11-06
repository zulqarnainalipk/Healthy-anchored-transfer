"""Module 05a: build the feature index that modules 05 and 06 read (one row per eligible image).

The index is the join of two stored tables, so it can be rebuilt without the images:
  - results/02b_splits/splits.csv                    dataset, path, group, split
  - results/01_crop_disease_table/labelled_images.csv  dataset, path, role, disease, is_healthy,
                                                      eligible_for_evaluation

Rows come out in the order of splits.csv. Only rows that are in both tables and are eligible
(eligible_for_evaluation = 1) are kept, which drops PlantSeg masks and the augmented fig copies.

Output: <output>/feature_index.csv with the columns dataset, path, split, class, is_healthy, and manifest.json
(row counts per dataset, SHA-256 of the file).

CPU module, runs locally in a few seconds. Logs each step with a time stamp.

Usage:
    python data_prep/05a_build_feature_index.py --splits results/02b_splits/splits.csv --labelled results/01_crop_disease_table/labelled_images.csv --output results/05_features_dinov2
"""

import argparse
import collections
import csv
import hashlib
import json
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/4: reading the labelled-image table")
    with open(args.labelled, encoding="utf-8-sig") as f:
        labelled = {(r["dataset"], r["path"]): r for r in csv.DictReader(f)}
    log(f"  {len(labelled)} rows")

    log("step 2/4: reading the split table")
    with open(args.splits, encoding="utf-8-sig") as f:
        splits = list(csv.DictReader(f))
    log(f"  {len(splits)} rows")

    log("step 3/4: joining, keeping eligible images only")
    rows, missing = [], 0
    for i, s in enumerate(splits):
        lab = labelled.get((s["dataset"], s["path"]))
        if lab is None:
            missing += 1
            continue
        if lab["eligible_for_evaluation"] != "1":
            continue
        rows.append([s["dataset"], s["path"], s["split"], lab["disease"], lab["is_healthy"]])
        if (i + 1) % 20000 == 0:
            log(f"  {i + 1}/{len(splits)} split rows read")
    log(f"  kept {len(rows)} rows; {missing} split rows had no labelled row")

    log("step 4/4: writing feature_index.csv and manifest.json")
    path = out / "feature_index.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "path", "split", "class", "is_healthy"])
        w.writerows(rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    per_ds = collections.Counter(r[0] for r in rows)
    (out / "feature_index_manifest.json").write_text(json.dumps({
        "module": "05a_build_feature_index", "rows": len(rows), "rows_per_dataset": dict(per_ds),
        "unmatched_split_rows": missing, "sha256": digest, "seconds": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    log(f"done: {len(rows)} rows, sha256 {digest[:12]}")


if __name__ == "__main__":
    main()
