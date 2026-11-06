"""Module 63: summary of the post-hoc checks in the third round of post-hoc checks (registration section 10).

(a) Learning curve: tier-2 results of module 53 with 10, 25, 50, 100 and 200 target healthy images in the map set
    (folders curve/n10, curve/n25, pw_pca_n50, curve/n100, pw_pca). Mean balanced accuracy of NA, MC, CORAL, HATC, probe5 and
    HATC minus MC with a 90% two-way cluster bootstrap interval.
(b) Decomposition of the gain by disease type for DINOv2 and BioCLIP: MC (mean only), STD (mean + per-coordinate scale),
    CORAL and HATC (full covariance); HATC minus MC and the share of that gap already reached by STD.
(c) Extra baselines (sens/extra_dino, sens/extra_bioclip): EIG (eigenvector-matched rotation, healthy images only) and the
    unlabelled-target reference arms UDA_CORAL and UDA_BW (pooled healthy + disease source against pooled unlabelled target images).
Writes results/63_review4_summary/{learning_curve,by_disease,extra_baselines}.csv.

Usage: python 63_review4_summary.py --results results
"""

import argparse
import csv
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m53", HERE.parent / "features" / "53_real_pair_test.py")
m53 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m53)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(path):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["tier"] == "2"]


def f(r, m):
    return float(r[f"ba_{m}"])


def diff_ci(u, a, b):
    v = [(r["target"], r["source"], f(r, a) - f(r, b)) for r in u]
    m, lo, hi, _, _ = m53.two_way_bootstrap(v, a, resamples=10000)
    return round(100 * m, 1), f"{100 * lo:.1f} to {100 * hi:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    R = Path(args.results) / "53_real_pair_test"
    out = Path(args.results) / "63_review4_summary"
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/3: learning curve")
    rows = []
    for n, p in ((10, "curve/n10"), (25, "curve/n25"), (50, "pw_pca_n50"), (100, "curve/n100"), (200, "pw_pca")):
        u = load(R / p / "units.csv")
        row = {"healthy_images_in_map_set": n, "units": len(u)}
        for m in ("NA", "MC", "CORAL", "HATC", "OC", "probe5"):
            row[f"ba_{m}"] = round(float(np.mean([f(r, m) for r in u])), 3)
        row["HATC_minus_MC"], row["HATC_minus_MC_ci90"] = diff_ci(u, "HATC", "MC")
        row["HATC_minus_CORAL"], row["HATC_minus_CORAL_ci90"] = diff_ci(u, "HATC", "CORAL")
        rows.append(row)
        log(f"  n={n}: HATC {row['ba_HATC']} MC {row['ba_MC']} gap {row['HATC_minus_MC']} ({row['HATC_minus_MC_ci90']})")
    write(out / "learning_curve.csv", rows)

    log("step 2/3: by disease type")
    rows = []
    for enc, p in (("dinov2", "pw_pca"), ("bioclip", "pw_bioclip_pca")):
        u = load(R / p / "units.csv")
        groups = defaultdict(list)
        for r in u:
            groups[r["disease"]].append(r)
        for d, g in sorted(groups.items()):
            row = {"encoder": enc, "disease": d, "units": len(g)}
            for m in ("NA", "MC", "STD", "CORAL", "HATC", "probe5"):
                row[f"ba_{m}"] = round(float(np.mean([f(r, m) for r in g])), 3)
            gap = row["ba_HATC"] - row["ba_MC"]
            row["HATC_minus_MC"] = round(100 * gap, 1)
            row["STD_minus_MC"] = round(100 * (row["ba_STD"] - row["ba_MC"]), 1)
            row["CORAL_minus_MC"] = round(100 * (row["ba_CORAL"] - row["ba_MC"]), 1)
            rows.append(row)
        row = {"encoder": enc, "disease": "ALL", "units": len(u)}
        for m in ("NA", "MC", "STD", "CORAL", "HATC", "probe5"):
            row[f"ba_{m}"] = round(float(np.mean([f(r, m) for r in u])), 3)
        row["HATC_minus_MC"] = round(100 * (row["ba_HATC"] - row["ba_MC"]), 1)
        row["STD_minus_MC"] = round(100 * (row["ba_STD"] - row["ba_MC"]), 1)
        row["CORAL_minus_MC"] = round(100 * (row["ba_CORAL"] - row["ba_MC"]), 1)
        rows.append(row)
        log(f"  {enc}: HATC-MC {row['HATC_minus_MC']}, STD-MC {row['STD_minus_MC']}, CORAL-MC {row['CORAL_minus_MC']}")
    write(out / "by_disease.csv", rows)

    log("step 3/3: extra baselines")
    rows = []
    for enc, p in (("dinov2", "sens/extra_dino"), ("bioclip", "sens/extra_bioclip")):
        u = load(R / p / "units.csv")
        row = {"encoder": enc, "units": len(u)}
        for m in ("NA", "MC", "CORAL", "HATC", "EIG", "UDA_CORAL", "UDA_BW", "probe5"):
            row[f"ba_{m}"] = round(float(np.mean([f(r, m) for r in u])), 3)
        for m in ("EIG", "CORAL", "UDA_CORAL", "UDA_BW", "probe5"):
            row[f"{m}_minus_HATC"], row[f"{m}_minus_HATC_ci90"] = diff_ci(u, m, "HATC")
        row["HATC_minus_MC"], row["HATC_minus_MC_ci90"] = diff_ci(u, "HATC", "MC")
        rows.append(row)
        log(f"  {enc}: " + ", ".join(f"{m} {row['ba_' + m]}" for m in ("MC", "CORAL", "HATC", "EIG", "UDA_CORAL", "UDA_BW", "probe5")))
    write(out / "extra_baselines.csv", rows)
    (out / "manifest.json").write_text(json.dumps({"module": "63_review4_summary"}, indent=2), encoding="utf-8")
    log("done")


def write(path, rows):
    names = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names, restval="")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
