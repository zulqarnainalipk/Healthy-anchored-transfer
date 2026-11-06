"""Module 17: first benchmark run on the synthetic worlds (proposal 7.4), affine branch.

For each world (module 16), each cell (healthy n in {50, 200}, disease n per class = 50) and each of
REPS replicates, the target crop's disease samples are classified among that crop's disease classes.
Three methods:
  NA    no adaptation: source class Gaussians, applied directly to the target
  MC    per-crop mean-centring: source Gaussians shifted so the source healthy mean matches the
        target healthy mean (no map)
  HATC  affine host map fitted from the source healthy sample to the target healthy sample
        (module 11 logic), applied to the source class Gaussians (module 13 logic)

Truth: a target disease sample of class d is correct when it is assigned to class d.

Outputs (results/17_benchmark_hatc/):
  - replicate_results.csv   one row per (world, cell, replicate, method): accuracy
  - summary.csv             mean accuracy per (world, cell, method) and paired difference HATC - NA
  - manifest.json

The worlds are regenerated here with the same operators and seed as module 16 (imported from that
file), so the benchmark uses exactly the operators defined there.

CPU module, runs locally.

Usage:
    python 17_benchmark_hatc.py --features ... --labelled ... --projection ... --output results/17_benchmark_hatc
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("world16", HERE / "16_world_generator.py")
w16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w16)

REPS = 200
HEALTHY_N = [50, 200]
DISEASE_N = 50
SHRINK = float(__import__("os").environ.get("HATC_SHRINK", 0.1))   # covariance shrinkage towards the scaled identity (registered value 0.1; sensitivity runs override it)
WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrunk_cov(Z):
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    return (1 - SHRINK) * C + SHRINK * np.trace(C) / d * np.eye(d)


def gauss_loglik(X, m, C):
    """Log-density of rows of X under N(m, C), up to a constant shared by all classes of one target."""
    sign, logdet = np.linalg.slogdet(C)
    inv = np.linalg.inv(C)
    D = X - m
    return -0.5 * (np.einsum("ij,jk,ik->i", D, inv, D) + logdet)


def classify(X, models):
    """Index of the class with the highest log-likelihood."""
    scores = np.vstack([gauss_loglik(X, m, C) for m, C in models])
    return scores.argmax(axis=0)


def affine_T(CA, CB):
    from scipy.linalg import sqrtm
    s1 = sqrtm(CA).real
    s1_inv = np.linalg.inv(s1)
    mid = sqrtm(s1 @ CB @ s1).real
    return s1_inv @ mid @ s1_inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="operator strength multiplier (module 16)")
    ap.add_argument("--worlds", nargs="+", default=None, help="subset of worlds (default: all)")
    ap.add_argument("--reps", type=int, default=None, help="replicates (default 200)")
    args = ap.parse_args()
    global REPS, WORLDS
    if args.reps:
        REPS = args.reps
    if args.worlds:
        WORLDS = args.worlds
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading source samples (same grouping as module 16)")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in w16.SOURCE_DATASETS and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    samples = {}
    for ds in w16.SOURCE_DATASETS:
        Z = (np.load(feats / f"features_{ds}.npy").astype(np.float64) - mean) @ W
        rows = [i for i, r in enumerate(index) if r["dataset"] == ds]
        for k, i in enumerate(rows):
            r = index[i]
            if r["split"] != "train" or r["path"] not in crop_of:
                continue
            key = (crop_of[r["path"]], "healthy" if r["is_healthy"] == "1" else r["class"])
            samples.setdefault(key, []).append(Z[k])
    samples = {k: np.array(v) for k, v in samples.items() if len(v) >= w16.MIN_TRAIN}
    if __import__("os").environ.get("HARMONISE", "1") == "1":   # merge same-disease names inside a crop (module 03b)
        samples = w16.harmonise_samples(samples)
    src_s, test_s, train_s = w16.split_samples(samples)
    samples = src_s  # source models use the source part only; target draws use test_s / train_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops with healthy samples")

    log("step 2/3: running worlds x cells x replicates")
    rep_rows = []
    for wi, name in enumerate(WORLDS, 1):
        rng = np.random.default_rng(w16.SEED)
        op, _truth = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in HEALTHY_N:
            for rep in range(REPS):
                r2 = np.random.default_rng(w16.SEED + rep)
                crop = crops[rep % len(crops)]
                k = crop_idx[crop]
                src_h = samples[(crop, "healthy")]
                th = test_s[(crop, "healthy")]
                H_tgt = op(th[r2.choice(len(th), size=hn, replace=True)], k)  # target healthy sample
                # Source class models for this crop (true source distribution, training samples)
                classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
                if len(classes) < 2:
                    continue
                cls_data = {d: samples[(crop, d)] for d in classes}
                # Target disease samples: operator applied to source class samples
                Xt, yt = [], []
                for ci, d in enumerate(classes):
                    arr = test_s[(crop, d)]
                    Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                    yt += [ci] * DISEASE_N
                Xt = np.vstack(Xt)
                yt = np.array(yt)
                # Source models: class Gaussians from source training samples
                src_models = [(cls_data[d].mean(axis=0), shrunk_cov(cls_data[d])) for d in classes]
                # NA: no adaptation
                pred_na = classify(Xt, src_models)
                # MC: shift each class mean by (target healthy mean - source healthy mean)
                shift = H_tgt.mean(axis=0) - src_h.mean(axis=0)
                mc_models = [(m + shift, C) for m, C in src_models]
                pred_mc = classify(Xt, mc_models)
                # HATC (affine): map from the source healthy sample to the target healthy sample
                mA, CA = src_h.mean(axis=0), shrunk_cov(src_h)
                mB, CB = H_tgt.mean(axis=0), shrunk_cov(H_tgt)
                T = affine_T(CA, CB)
                hatc_models = [(mB + (m - mA) @ T.T, T @ C @ T.T) for m, C in src_models]
                pred_h = classify(Xt, hatc_models)
                for method, pred in (("NA", pred_na), ("MC", pred_mc), ("HATC", pred_h)):
                    rep_rows.append({"world": name, "healthy_n": hn, "replicate": rep, "crop": crop,
                                     "method": method, "accuracy": round(float(np.mean(pred == yt)), 4)})
        log(f"  world {name} done ({wi}/{len(WORLDS)})")

    log("step 3/3: summary and paired differences")
    with open(out / "replicate_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        w.writeheader()
        w.writerows(rep_rows)
    summary = []
    for name in WORLDS:
        for hn in HEALTHY_N:
            sub = [r for r in rep_rows if r["world"] == name and r["healthy_n"] == hn]
            acc = {m: np.array([r["accuracy"] for r in sub if r["method"] == m]) for m in ("NA", "MC", "HATC")}
            diff = acc["HATC"] - acc["NA"]
            summary.append({"world": name, "healthy_n": hn,
                            "acc_NA": round(float(acc["NA"].mean()), 4),
                            "acc_MC": round(float(acc["MC"].mean()), 4),
                            "acc_HATC": round(float(acc["HATC"].mean()), 4),
                            "paired_diff_HATC_minus_NA": round(float(diff.mean()), 4),
                            "replicates": int(len(diff))})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    manifest = {"module": "17_benchmark_hatc", "worlds": WORLDS, "healthy_n": HEALTHY_N,
                "disease_n": DISEASE_N, "replicates": REPS, "operator_scale": args.scale, "methods": ["NA", "MC", "HATC"],
                "note": "affine branch only; intervals (module 36) and the Brenier branch come later",
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()

