"""Module 62: summary of the post-hoc healthy-only regularisation and control runs of the real-pair test (module 53, folder sens/).

Reads results/53_real_pair_test/pw_pca (registered covariance, lambda 0.1) and results/53_real_pair_test/sens/<tag> for the covariance
estimators (Ledoit-Wolf, OAS, rank-2/4/8 factor model, lambda 0.03 and 0.3), HATC-S (select) and the frame-only features (border).
For each run, tier 2: mean balanced accuracy of NA, MC, CORAL, HATC (and SELECT), and the 90% two-way cluster-bootstrap interval of
HATC minus mean-centring and of HATC minus no adaptation (the best simple method on frame-only features is no adaptation).
Writes results/62_sensitivity_summary/summary.csv.

Usage: python 62_sensitivity_summary.py --results results
"""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m53", HERE.parent / "features" / "53_real_pair_test.py")
m53 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m53)

RUNS = [("registered (lambda 0.1)", "pw_pca"), ("lambda 0.03", "sens/lam0.03"), ("lambda 0.3", "sens/lam0.3"), ("Ledoit-Wolf", "sens/cov_lw"),
        ("OAS", "sens/cov_oas"), ("rank 2 + noise", "sens/cov_rank2"), ("rank 4 + noise", "sens/cov_rank4"), ("rank 8 + noise", "sens/cov_rank8"),
        ("HATC-S (select)", "sens/select"), ("frame-only features", "sens/border")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    R = Path(args.results) / "53_real_pair_test"
    out = Path(args.results) / "62_sensitivity_summary"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in RUNS:
        f = R / path / "units.csv"
        if not f.exists():
            print(f"[{time.strftime('%H:%M:%S')}] missing {path}", flush=True)
            continue
        u = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["tier"] == "2"]
        mean = lambda m: float(np.mean([float(r[f"ba_{m}"]) for r in u]))
        row = {"run": label, "folder": path, "units": len(u)}
        for m in ("NA", "MC", "CORAL", "HATC", "OC", "SELECT", "probe5"):
            row[f"ba_{m}"] = round(mean(m), 3) if f"ba_{m}" in u[0] else ""
        for base in ("MC", "NA"):
            v = [(r["target"], r["source"], float(r["ba_HATC"]) - float(r[f"ba_{base}"])) for r in u]
            m_, lo, hi, _, _ = m53.two_way_bootstrap(v, "HATC", resamples=10000)
            row[f"HATC_minus_{base}"] = round(100 * m_, 1)
            row[f"HATC_minus_{base}_ci90"] = f"{100 * lo:.1f} to {100 * hi:.1f}"
        rows.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {label}: HATC-MC {row['HATC_minus_MC']} ({row['HATC_minus_MC_ci90']}), HATC-NA {row['HATC_minus_NA']}", flush=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[{time.strftime('%H:%M:%S')}] done", flush=True)


if __name__ == "__main__":
    main()
