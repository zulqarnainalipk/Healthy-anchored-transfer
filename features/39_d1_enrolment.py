"""Module 39: D1 enrolment gate with refusal (proposal section 6, device workflow).

D1 builds a profile from the healthy captures of a new crop and enrols the crop only if the sample count, the
covariance conditioning and the quality spread pass. This module tests the gate in simulation, on real projected
features, against a mismatch set:
  - clean captures: n healthy images of one crop (should be enrolled)
  - mixed captures: the same n images, but a fraction FRACTION come from a DIFFERENT crop (a different species
    captured as the new crop). These should be refused.

Gate statistics (computed in the 16-dimensional projection):
  - count      at least MIN_COUNT (50) captures
  - condition  condition number of the shrunk covariance, at most the threshold
  - spread     trace of the shrunk covariance, at most the threshold (a mixture of species is more spread out)
Thresholds are set on SOURCE crops only: the 95th percentile of each statistic over clean captures of the
source crops, at each n. No target data is used.

Metrics (proposal section 6): false enrolment rate (mixed captures enrolled) and false refusal rate (clean
captures refused), compared with the baseline of enrolment without the gate (everything enrolled).

Outputs (results/39_d1_enrolment/): thresholds.csv, enrolment_rates.csv, manifest.json

CPU module, runs locally.

Usage:
    python 39_d1_enrolment.py --features ... --labelled ... --projection ... --output results/39_d1_enrolment
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
s16 = importlib.util.spec_from_file_location("w16", HERE / "16_world_generator.py")
w16 = importlib.util.module_from_spec(s16)
s16.loader.exec_module(w16)
s17 = importlib.util.spec_from_file_location("w17", HERE / "17_benchmark_hatc.py")
w17 = importlib.util.module_from_spec(s17)
s17.loader.exec_module(w17)

MIN_COUNT = 50
SIZES = [50, 100, 200]
FRACTIONS = [0.1, 0.25, 0.5]
REPS = 200
SEED = 2026
QUANTILE = 95


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(Z):
    C = w17.shrunk_cov(Z)
    ev = np.linalg.eigvalsh(C)
    return float(ev.max() / ev.min()), float(np.trace(C))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading healthy samples per crop (source crops, training split)")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "plantvillage_color" and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    Z = (np.load(feats / "features_plantvillage_color.npy").astype(np.float64) - mean) @ W
    rows = [i for i, r in enumerate(index) if r["dataset"] == "plantvillage_color"]
    healthy = {}
    for k, i in enumerate(rows):
        r = index[i]
        if r["is_healthy"] == "1" and r["split"] == "train" and r["path"] in crop_of:
            healthy.setdefault(crop_of[r["path"]], []).append(Z[k])
    healthy = {c: np.array(v) for c, v in healthy.items() if len(v) >= 250}
    crops = sorted(healthy)
    log(f"  crops with at least 250 healthy training images: {crops}")

    log("step 2/3: thresholds from clean source captures, then rates")
    rng = np.random.default_rng(SEED)
    thresholds = {}
    for n in SIZES:
        cond, spread = [], []
        for _ in range(REPS):
            c = crops[rng.integers(len(crops))]
            cc, ss = stats(healthy[c][rng.choice(len(healthy[c]), size=n, replace=False)])
            cond.append(cc)
            spread.append(ss)
        thresholds[n] = (float(np.percentile(cond, QUANTILE)), float(np.percentile(spread, QUANTILE)))
    with open(out / "thresholds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "condition_threshold", "spread_threshold", "quantile"])
        for n, (a, b) in thresholds.items():
            w.writerow([n, round(a, 3), round(b, 3), QUANTILE])

    results = []
    for n in SIZES:
        t_cond, t_spread = thresholds[n]
        # false refusal: clean captures refused (a fresh draw, not the threshold draws)
        refused = 0
        trials = 0
        for _ in range(REPS):
            c = crops[rng.integers(len(crops))]
            cc, ss = stats(healthy[c][rng.choice(len(healthy[c]), size=n, replace=False)])
            refused += int(n < MIN_COUNT or cc > t_cond or ss > t_spread)
            trials += 1
        far = refused / trials
        for frac in FRACTIONS:
            enrolled = 0
            trials = 0
            for _ in range(REPS):
                c, other = rng.choice(crops, size=2, replace=False)
                n_other = max(1, int(round(frac * n)))
                a = healthy[c][rng.choice(len(healthy[c]), size=n - n_other, replace=False)]
                b = healthy[other][rng.choice(len(healthy[other]), size=n_other, replace=False)]
                cc, ss = stats(np.vstack([a, b]))
                enrolled += int(not (n < MIN_COUNT or cc > t_cond or ss > t_spread))
                trials += 1
            results.append({"n": n, "mixed_fraction": frac, "false_enrolment_rate_gate": round(enrolled / trials, 4),
                            "false_enrolment_rate_no_gate": 1.0, "false_refusal_rate_gate": round(far, 4),
                            "trials": trials})
            log(f"  n={n} mix={frac}: false enrolment {enrolled / trials:.3f}, false refusal {far:.3f}")

    log("step 3/3: writing results")
    with open(out / "enrolment_rates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "manifest.json").write_text(json.dumps({
        "module": "39_d1_enrolment", "crops": crops, "sizes": SIZES, "fractions": FRACTIONS,
        "replicates": REPS, "quantile": QUANTILE, "seed": SEED, "seconds": round(time.time() - t0, 1),
        "note": "thresholds set on source crops only; mismatch is a mixture with a different crop"}, indent=2),
        encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
