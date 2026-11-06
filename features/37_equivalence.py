"""Module 37: equivalence outcomes for the primary contrast (proposal 7.6).

Input: results/36_cluster_bootstrap/intervals.csv.
Primary contrast: HATC minus the best simple baseline, in each world and healthy-image cell.

Three exclusive outcomes, with the equivalence margin of 0.02 accuracy (registered; 2 points) and alpha 0.05,
read from the 90 percent cluster-bootstrap interval (two one-sided tests at 0.05):
  equivalent    the 90% interval lies inside (-0.02, +0.02)
  different     the 90% interval lies entirely above +0.02 (HATC better) or below -0.02 (HATC worse)
  inconclusive  anything else. An inconclusive result is never reported as a tie.

If the cell has fewer than 8 clusters, the outcome is marked exploratory (registered rule).

All other contrasts (against NA, MC, STD, CORAL) are labelled exploratory in this table; the proposal
allows Holm correction or exploratory labelling for them (7.7). Exploratory labelling is used.

Outputs (results/37_equivalence/): primary_outcomes.csv, exploratory_outcomes.csv, summary.txt, manifest.json

CPU module, runs locally.

Usage:
    python 37_equivalence.py --intervals results/36_cluster_bootstrap/intervals.csv --output results/37_equivalence
"""

import argparse
import collections
import csv
import json
import time
from pathlib import Path

MARGIN = 0.02


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def outcome(low, high):
    if low > -MARGIN and high < MARGIN:
        return "equivalent"
    if low > MARGIN:
        return "different (HATC better)"
    if high < -MARGIN:
        return "different (HATC worse)"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/3: reading the cluster-bootstrap intervals")
    with open(args.intervals, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    log(f"  {len(rows)} interval rows")
    log("step 2/3: classifying each interval (equivalent, different, inconclusive)")
    primary, explo = [], []
    for r in rows:
        low, high = float(r["ci90_low"]), float(r["ci90_high"])
        res = {"world": r["world"], "healthy_n": r["healthy_n"], "contrast": r["contrast"],
               "best_simple": r["best_simple"], "mean_difference": r["mean_difference"],
               "ci90_low": r["ci90_low"], "ci90_high": r["ci90_high"], "clusters": r["clusters"],
               "outcome": outcome(low, high),
               "status": "exploratory (fewer than 8 clusters)" if r["status"].startswith("exploratory") else "ok"}
        (primary if r["contrast"] == "HATC - best_simple" else explo).append(res)

    log("step 3/3: writing outcome tables, summary and manifest")
    for name, data in (("primary_outcomes.csv", primary), ("exploratory_outcomes.csv", explo)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    counts = collections.Counter(r["outcome"] for r in primary)
    lines = ["Primary contrast: HATC - best simple baseline", f"cells: {len(primary)}"]
    lines += [f"  {k}: {v}" for k, v in sorted(counts.items())]
    exploratory_all = all(r["status"].startswith("exploratory") for r in primary)
    lines.append(f"all cells exploratory (fewer than 8 clusters): {exploratory_all}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"module": "37_equivalence", "margin": MARGIN,
                                                    "primary_cells": len(primary),
                                                    "outcome_counts": counts}, indent=2), encoding="utf-8")
    for ln in lines:
        log(ln)
    log("done")


if __name__ == "__main__":
    main()
