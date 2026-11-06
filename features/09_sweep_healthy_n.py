"""Module 09: source-only sweep of the healthy-image sample size (proposal 7.2).

Question: how many healthy images does a crop need before its affine transport map is stable?
This is the source-only analogue of the 50 and 200 thresholds. It uses source crops only, so no
target result can influence it (accepted timing deviation, proposal section 7).

Data: PlantVillage healthy images of crops with at least MIN_HEALTHY healthy images, from the TRAINING
split (for sampling) and the VALIDATION split of the same crop (as held-out truth). Features are the
projection with 16 dimensions from module 08 (main setting).

Method, for each ordered pair (A, B) of such crops and each n in N_GRID:
  1. sample n healthy training images of A and n of B (fixed seed)
  2. fit a Gaussian to each in the projected space (shrunk covariance)
  3. map A to B with the affine Monge map between the two Gaussians
  4. error = Bures-Wasserstein (Frechet) distance between the mapped Gaussian and the Gaussian of
     B's held-out healthy images (validation split), computed in the projected space
Outputs per (pair, n) and, per n, the median error over pairs. The stability rule (registered
later) compares median errors across n, against the error at the largest n.

CPU module. Reads cached features, the projection file and the index. Runs locally.

Usage:
    python 09_sweep_healthy_n.py --features results/05_features_dinov2/remote/features_dinov2
        --labelled results/01_crop_disease_table/labelled_images.csv
        --projection results/08_projection/projection_k16.npz --output results/09_sweep_healthy_n
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

DATASET = "plantvillage_color"
MIN_HEALTHY = 400
N_GRID = [25, 50, 100, 200, 400]
REPS = 5
SEED = 2026
SHRINK = 0.1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrunk_cov(Z):
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    return (1 - SHRINK) * C + SHRINK * np.trace(C) / d * np.eye(d)


def frechet(m1, C1, m2, C2):
    """Bures-Wasserstein (Frechet) distance between two Gaussians."""
    s1 = sqrtm(C1).real
    cross = sqrtm(s1 @ C2 @ s1).real
    return float(np.sum((m1 - m2) ** 2) + np.trace(C1 + C2 - 2 * cross))


def affine_map(C1, C2):
    """Monge map T = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2} from Gaussian 1 to Gaussian 2."""
    s1 = sqrtm(C1).real
    s1_inv = np.linalg.inv(s1)
    mid = sqrtm(s1 @ C2 @ s1).real
    return s1_inv @ mid @ s1_inv


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

    log("step 1/4: loading projection, features and labels")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = [r for r in csv.DictReader(f) if r["dataset"] == DATASET]
    feat = np.load(feats / f"features_{DATASET}.npy").astype(np.float64)
    # Map each index row to its crop through the labelled table (paths are unique).
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == DATASET and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    Z = (feat - mean) @ W
    healthy = [i for i, r in enumerate(index) if r["is_healthy"] == "1"]
    by_crop = {}
    for i in healthy:
        c = crop_of.get(index[i]["path"])
        if c:
            by_crop.setdefault(c, {"train": [], "val": []})
            if index[i]["split"] in ("train", "val"):
                by_crop[c][index[i]["split"]].append(i)
    crops = sorted(c for c, v in by_crop.items()
                   if len(v["train"]) + len(v["val"]) >= MIN_HEALTHY and len(v["val"]) >= 50)
    log(f"  {len(crops)} crops with at least {MIN_HEALTHY} healthy images and 50 validation: {crops}")

    log("step 2/4: sweeping n over ordered crop pairs")
    rng = np.random.default_rng(SEED)
    rows = []
    pairs = [(a, b) for a in crops for b in crops if a != b]
    for pi, (a, b) in enumerate(pairs, 1):
        if pi % 20 == 0:
            log(f"  pair {pi}/{len(pairs)}")
        A_tr = np.array(by_crop[a]["train"], dtype=int)
        B_tr = np.array(by_crop[b]["train"], dtype=int)
        B_va = np.array(by_crop[b]["val"], dtype=int)
        Bva = Z[B_va]
        mB_true = Bva.mean(axis=0)
        CB_true = shrunk_cov(Bva)
        for n in N_GRID:
            errs = []
            for r in range(REPS):
                ia = rng.choice(A_tr, size=min(n, len(A_tr)), replace=False)
                ib = rng.choice(B_tr, size=min(n, len(B_tr)), replace=False)
                Za, Zb = Z[ia], Z[ib]
                mA, CA = Za.mean(axis=0), shrunk_cov(Za)
                mB, CB = Zb.mean(axis=0), shrunk_cov(Zb)
                T = affine_map(CA, CB)
                # The affine Monge map x -> mB + T (x - mA) sends A's Gaussian to
                # N(mB, T CA T'). Compare it with B's held-out Gaussian.
                C_t = T @ CA @ T.T
                errs.append(frechet(mB, C_t, mB_true, CB_true))
            rows.append({"source_crop": a, "target_crop": b, "n": n,
                         "median_error": round(float(np.median(errs)), 4),
                         "mean_error": round(float(np.mean(errs)), 4)})

    log("step 3/4: summary per n")
    summary = {}
    for n in N_GRID:
        vals = [r["median_error"] for r in rows if r["n"] == n]
        summary[n] = {"pairs": len(vals), "median_over_pairs": round(float(np.median(vals)), 4),
                      "mean_over_pairs": round(float(np.mean(vals)), 4)}
    ref = summary[max(N_GRID)]["median_over_pairs"]
    for n in N_GRID:
        summary[n]["relative_to_largest_n"] = round(summary[n]["median_over_pairs"] / ref, 4) if ref else None

    log("step 4/4: writing outputs")
    with open(out / "sweep_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "crops": crops, "n_grid": N_GRID, "reps": REPS, "seed": SEED, "shrinkage": SHRINK,
        "projection_dimensions": int(W.shape[1]), "per_n": summary,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
