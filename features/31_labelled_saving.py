"""Module 31: labelled-data saving on the synthetic worlds (proposal section 5, intended effect).

HATC uses no labelled diseased images of the target crop. The primary baseline is a few-shot linear probe
(module 26) trained on the target's labelled diseased images at 0, 5, 10, 25, 50, 100 and 200 labels per
class. The baseline "reaches" HATC at the first grid point where its mean accuracy is within the registered
tolerance (0.02) of HATC's mean accuracy in the same world and healthy-image cell. That grid point is the
number of labelled images per class that HATC saves.

If the probe never reaches HATC by 200 labels, the cell is censored: the saving is at least 200 labels per
class (a lower bound, not interpolated beyond the grid).

Inputs:
  - results/20_align_baselines/replicate_results.csv   HATC accuracy per world, healthy n, replicate
  - results/26_linear_probes/probe_curve.csv           probe mean accuracy per world and label count
Outputs (results/31_labelled_saving/):
  - labelled_saving.csv   one row per (world, healthy n): HATC accuracy, probe curve, first reaching grid
                          point, censored flag, saving in labels per class
  - manifest.json

Pairing note: the probe test samples equal HATC's test samples only for healthy n = 50 (the random stream
aligns for that cell). For healthy n = 200 the probe and HATC use different draws from the same
distribution, so the comparison is between means over replicates, not paired replicate by replicate.

CPU module, runs locally.

Usage:
    python 31_labelled_saving.py --hatc results/20_align_baselines/replicate_results.csv
        --probes results/26_linear_probes/probe_curve.csv --output results/31_labelled_saving
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

TOLERANCE = 0.02
GRID = [5, 10, 25, 50, 100, 200]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hatc", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/3: reading HATC replicate accuracies")
    hatc = {}
    with open(args.hatc, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["method"] == "HATC":
                hatc.setdefault((r["world"], int(r["healthy_n"])), []).append(float(r["accuracy"]))
    log(f"  {len(hatc)} (world, healthy n) cells")

    log("step 2/3: reading probe curve")
    probe = {}
    with open(args.probes, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            probe[(r["world"], int(r["labels_per_class"]))] = float(r["mean_accuracy"])

    log("step 3/3: finding the first grid point where the probe reaches HATC")
    rows = []
    for (world, hn), accs in sorted(hatc.items()):
        h = float(np.mean(accs))
        reach = next((g for g in GRID if probe.get((world, g), -1) >= h - TOLERANCE), None)
        rows.append({
            "world": world, "healthy_n": hn, "hatc_accuracy": round(h, 4),
            "probe_0": round(probe.get((world, 0), float("nan")), 4),
            **{f"probe_{g}": round(probe.get((world, g), float("nan")), 4) for g in GRID},
            "reaching_labels_per_class": reach if reach is not None else "",
            "censored": int(reach is None),
            "saving_labels_per_class": f">={GRID[-1]}" if reach is None else str(reach),
        })
    with open(out / "labelled_saving.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    manifest = {"module": "31_labelled_saving", "tolerance": TOLERANCE, "grid": GRID,
                "cells": len(rows), "censored": sum(r["censored"] for r in rows)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()

