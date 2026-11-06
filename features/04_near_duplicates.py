"""Module 04: near-duplicate check between target and source images (proposal 7.9).

Uses the cached DINOv2 features from module 05. For each target image (rice, fig Original, chili,
peach) it finds the most similar source image (PlantSeg and PlantVillage). Similarity is the cosine
of the L2-normalised class-token vector (the first 768 values).

Outputs:
  - nearest.csv           one row per target image: best source path and similarity
  - sensitivity.csv       count of target images with a source at or above each threshold,
                          per target dataset (threshold chosen later by hand-checked validation)
  - summary.txt           totals

The threshold is NOT chosen here. Proposal 7.9 requires a hand-checked sample of pairs before any
threshold is registered. This module supplies the candidates and the sensitivity table.

CPU module. Uses the feature files and the index only. Runs locally.

Usage:
    python 04_near_duplicates.py --features results/05_features_dinov2/remote/features_dinov2 --output results/04_near_duplicates
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

TARGETS = ["rice_leaf_disease", "fig_sartak_sfld", "chili_yellow_leaf", "peach_leaf_damage"]
SOURCES = ["plantseg_v6", "plantvillage_color"]
THRESHOLDS = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99]
CHUNK = 2048


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(features_dir, ds):
    a = np.load(features_dir / f"features_{ds}.npy")
    cls = a[:, :768].astype(np.float32)
    cls /= np.linalg.norm(cls, axis=1, keepdims=True) + 1e-9
    return cls


def paths_of(index_rows, ds):
    return [r["path"] for r in index_rows if r["dataset"] == ds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading index and features")
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    src_vecs, src_paths = [], []
    for ds in SOURCES:
        v = load(feats, ds)
        src_vecs.append(v)
        src_paths += paths_of(index, ds)
    src = np.vstack(src_vecs)
    log(f"  source images: {src.shape[0]}")

    log("step 2/3: nearest source for each target image")
    nearest_rows = []
    sens = {}
    for ds in TARGETS:
        t = load(feats, ds)
        tp = paths_of(index, ds)
        best = np.empty(t.shape[0], dtype=np.float32)
        arg = np.empty(t.shape[0], dtype=np.int64)
        for s in range(0, t.shape[0], CHUNK):
            sim = t[s:s + CHUNK] @ src.T
            arg[s:s + CHUNK] = sim.argmax(axis=1)
            best[s:s + CHUNK] = sim.max(axis=1)
            if (s // CHUNK) % 2 == 0:
                log(f"    {ds}: {min(s + CHUNK, t.shape[0])}/{t.shape[0]}")
        for k in range(t.shape[0]):
            nearest_rows.append({"target_dataset": ds, "target_path": tp[k],
                                 "source_path": src_paths[arg[k]], "similarity": round(float(best[k]), 4)})
        sens[ds] = {th: int(np.sum(best >= th)) for th in THRESHOLDS}
        log(f"  {ds}: max similarity {best.max():.4f}, median {np.median(best):.4f}")

    log("step 3/3: writing outputs")
    with open(out / "nearest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(nearest_rows[0].keys()))
        w.writeheader()
        w.writerows(nearest_rows)
    with open(out / "sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_dataset", "images"] + [f"at_or_above_{th}" for th in THRESHOLDS])
        for ds in TARGETS:
            n = sum(1 for r in nearest_rows if r["target_dataset"] == ds)
            w.writerow([ds, n] + [sens[ds][th] for th in THRESHOLDS])
    total = sum(sens[ds][0.95] for ds in TARGETS)
    (out / "summary.txt").write_text(
        f"targets: {len(nearest_rows)}; sources: {src.shape[0]}; at or above 0.95 (total): {total}; "
        f"seconds: {round(time.time() - t0, 1)}\n", encoding="utf-8")
    log(f"done: at or above 0.95 total {total}")


if __name__ == "__main__":
    main()
