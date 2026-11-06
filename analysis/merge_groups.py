"""Merges the per-group outputs of modules 26 and 30 (run as four parallel world groups) into single csv files.

Writes results/26_linear_probes/probe_curve.csv (rows of g1..g4/probe_curve.csv) and
results/30_certificate_baselines/auroc.csv (rows of g1..g4/auroc.csv). CPU, instant.
Usage: python analysis/merge_groups.py
"""

import csv
import time
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


for folder, name in (("results/26_linear_probes", "probe_curve.csv"), ("results/30_certificate_baselines", "auroc.csv")):
    rows, header = [], None
    for g in ("g1", "g2", "g3", "g4"):
        p = Path(folder) / g / name
        with open(p, encoding="utf-8-sig", newline="") as f:
            rd = csv.reader(f)
            h = next(rd)
            header = header or h
            rows += list(rd)
    with open(Path(folder) / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log(f"{folder}/{name}: {len(rows)} rows")
