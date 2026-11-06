"""Module 36: cluster-bootstrap intervals for paired accuracy differences (proposal 7.7).

Input: results/20_align_baselines/replicate_results.csv (HATC and baselines on the synthetic worlds).
Each replicate is one crop (the cluster). Replicates in the same crop share source samples, so the
bootstrap resamples CROPS (clusters) with replacement, not replicates.

For each world and healthy-image count:
  - the paired difference HATC - baseline, per replicate, for baselines NA, MC, STD, CORAL and
    "best simple" (the best of NA, MC and STD by mean accuracy in that cell)
  - the mean difference, and percentile intervals from RESAMPLES cluster-bootstrap resamples
    (90 percent, for the equivalence test in module 37, and 95 percent)
  - the number of clusters (crops) in the cell; below MIN_CLUSTERS the interval is marked exploratory

The proposal's two-way bootstrap (target crop and source crop) reduces to one cluster dimension in the
synthetic worlds, because each replicate uses one crop as both. This is stated as a limitation.

Outputs (results/36_cluster_bootstrap/): intervals.csv, manifest.json

CPU module, runs locally.

Usage:
    python 36_cluster_bootstrap.py --replicates results/20_align_baselines/replicate_results.csv
        --output results/36_cluster_bootstrap
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

RESAMPLES = 10000
MIN_CLUSTERS = 8
SEED = 2026
BASELINES = ["NA", "MC", "STD", "CORAL"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    log("step 1/3: reading replicate accuracies")
    data = {}  # (world, hn) -> {method: {replicate: (crop, accuracy)}}
    with open(args.replicates, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["world"], int(r["healthy_n"]))
            data.setdefault(key, {}).setdefault(r["method"], {})[int(r["replicate"])] = (r["crop"], float(r["accuracy"]))
    log(f"  {len(data)} cells")

    log("step 2/3: cluster bootstrap per cell and baseline")
    rows = []
    for (world, hn), methods in sorted(data.items()):
        reps = sorted(methods["HATC"])
        crops = np.array([methods["HATC"][r][0] for r in reps])
        acc = {m: np.array([methods[m][r][1] for r in reps]) for m in methods}
        simple = ["NA", "MC", "STD"]
        best_simple = max(simple, key=lambda m: acc[m].mean())
        diffs = {m: acc["HATC"] - acc[m] for m in BASELINES}
        diffs["best_simple"] = acc["HATC"] - acc[best_simple]
        unique = np.unique(crops)
        n_clusters = len(unique)
        idx_by = {c: np.where(crops == c)[0] for c in unique}
        for name, d in diffs.items():
            boots = np.empty(RESAMPLES)
            for b in range(RESAMPLES):
                pick = rng.choice(unique, size=n_clusters, replace=True)
                boots[b] = np.mean(np.concatenate([d[idx_by[c]] for c in pick]))
            rows.append({
                "world": world, "healthy_n": hn, "contrast": f"HATC - {name}",
                "best_simple": best_simple if name == "best_simple" else "",
                "mean_difference": round(float(d.mean()), 4),
                "ci90_low": round(float(np.percentile(boots, 5)), 4),
                "ci90_high": round(float(np.percentile(boots, 95)), 4),
                "ci95_low": round(float(np.percentile(boots, 2.5)), 4),
                "ci95_high": round(float(np.percentile(boots, 97.5)), 4),
                "clusters": n_clusters, "replicates": len(d),
                "status": "ok" if n_clusters >= MIN_CLUSTERS else "exploratory (fewer than 8 clusters)",
            })
        log(f"  {world} n={hn}: {n_clusters} clusters, best simple {best_simple}")

    log("step 3/3: writing intervals")
    with open(out / "intervals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    manifest = {"module": "36_cluster_bootstrap", "resamples": RESAMPLES, "min_clusters": MIN_CLUSTERS,
                "seed": SEED, "cells": len(data), "rows": len(rows),
                "limitation": "one cluster dimension (crop) in the synthetic worlds",
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
