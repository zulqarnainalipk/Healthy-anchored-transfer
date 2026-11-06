"""Module 38: exact sign test, descriptive only (proposal 7.7).

For each world and healthy-image cell, the mean accuracy difference HATC - best simple baseline is computed
per crop (cluster). An exact two-sided sign test is run across crops (ties dropped). The proposal reports
the exact test as DESCRIPTIVE ONLY, because it assumes independent pairs, which crops sharing source
samples are not. The smallest achievable two-sided p-value at the number of crops is also reported, so the
reader sees what the test could ever show at this sample size.

Input: results/20_align_baselines/replicate_results.csv.
Output (results/38_exact_test/): sign_tests.csv, manifest.json

CPU module, runs locally.

Usage:
    python 38_exact_test.py --replicates results/20_align_baselines/replicate_results.csv
        --output results/38_exact_test
"""

import argparse
import csv
import json
import time
from math import comb
from pathlib import Path

import numpy as np


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sign_test_p(k, n):
    """Exact two-sided sign test p-value for k positives out of n non-zero differences."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/3: reading the replicate results")
    data = {}
    with open(args.replicates, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["world"], int(r["healthy_n"]))
            data.setdefault(key, {}).setdefault(r["method"], {})[int(r["replicate"])] = (r["crop"], float(r["accuracy"]))
    log(f"  {len(data)} cells (world x healthy n)")
    log("step 2/3: exact sign test per cell, over crops")
    rows = []
    for (world, hn), methods in sorted(data.items()):
        reps = sorted(methods["HATC"])
        crops = np.array([methods["HATC"][r][0] for r in reps])
        acc = {m: np.array([methods[m][r][1] for r in reps]) for m in methods}
        best = max(["NA", "MC", "STD"], key=lambda m: acc[m].mean())
        d = acc["HATC"] - acc[best]
        per_crop = [float(d[crops == c].mean()) for c in np.unique(crops)]
        pos = sum(x > 0 for x in per_crop)
        neg = sum(x < 0 for x in per_crop)
        n = pos + neg
        rows.append({"world": world, "healthy_n": hn, "best_simple": best, "crops": len(per_crop),
                     "crops_hatc_better": pos, "crops_hatc_worse": neg,
                     "exact_p_two_sided": round(sign_test_p(pos, n), 4),
                     "smallest_achievable_p": round(2 / 2 ** len(per_crop), 4)})
    log("step 3/3: writing sign_tests.csv and manifest.json")
    with open(out / "sign_tests.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps({
        "module": "38_exact_test", "cells": len(rows), "descriptive_only": True,
        "note": "crops share source samples, so independence does not hold; no inference is drawn"}, indent=2),
        encoding="utf-8")
    log(f"done: {len(rows)} cells; smallest achievable p at {rows[0]['crops']} crops = {rows[0]['smallest_achievable_p']}")


if __name__ == "__main__":
    main()
