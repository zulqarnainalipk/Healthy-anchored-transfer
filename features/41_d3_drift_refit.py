"""Module 41: D3, cached push-through with drift-gated refit (proposal section 6).

D3 stores the pushed disease models for a new crop and refits the host map only when a drift statistic passes
a threshold fixed on source crops. This module simulates a target crop whose captures drift over time.

Setup (synthetic world a, strength 2, disjoint pools as in modules 17 and 20):
  - the target at time t is the world operator applied to source samples, plus a drift shift
    d(t) = (t / T) * DRIFT * u, where u is a fixed random unit vector and T the number of time steps
  - at each step the device captures HEALTHY_STEP new target healthy images; the healthy window is the most
    recent WINDOW images
  - drift statistic: Mahalanobis distance of the window mean from the profile mean (profile = the window at the
    last refit), plus the relative Frobenius change of the window covariance. Threshold: the 95th percentile of
    the same statistic computed on windows of CLEAN source-crop captures (no drift), so no target data sets it
Warm-up: no policy runs until the healthy window is full (WINDOW images), so no policy starts from a small
first fit. All policies are evaluated only from that step onward.
Policies:
  never      fit once when the window is full and never refit (cached forever)
  gated      refit when the drift statistic exceeds the threshold
  always     refit at every step
Metrics, per policy: number of refits, share of steps served from cache, mean accuracy over time and accuracy at
the last step, on target disease test samples drawn at each step with the same drift.

Outputs (results/41_d3_drift/): drift_results.csv, manifest.json

CPU module, runs locally.

Usage:
    python 41_d3_drift_refit.py --features ... --labelled ... --projection ... --output results/41_d3_drift
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

STEPS = 12
HEALTHY_STEP = 25
WINDOW = 100
DISEASE_N = 40
REPS = 60
DRIFTS = [0.0, 1.0, 2.0, 4.0]
SEED = 2026
QUANTILE = 95


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def drift_stat(win, prof_mean, prof_cov, prof_cov_norm):
    D = win.mean(axis=0) - prof_mean
    m = float(np.sqrt(D @ np.linalg.inv(prof_cov) @ D))
    c = float(np.linalg.norm(w17.shrunk_cov(win) - prof_cov) / prof_cov_norm)
    return m + c


def fit_models(src_h, cls_data, classes, window):
    mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
    mB, CB = window.mean(axis=0), w17.shrunk_cov(window)
    T = w17.affine_T(CA, CB)
    return [(mB + (cls_data[d].mean(axis=0) - mA) @ T.T,
             T @ w17.shrunk_cov(cls_data[d]) @ T.T) for d in classes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading source samples")
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
    src_s, test_s, _ = w16.split_samples(samples)
    samples = src_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    rng0 = np.random.default_rng(w16.SEED)
    op, _ = w16.operators("a", rng0, mu, len(crops), s=args.scale)
    u = np.random.default_rng(SEED).standard_normal(16)
    u /= np.linalg.norm(u)

    log("step 2/3: drift threshold from clean source-crop captures")
    stat_null = []
    rngn = np.random.default_rng(SEED)
    for _ in range(300):
        c = crops[rngn.integers(len(crops))]
        H = samples[(c, "healthy")]
        prof = H[rngn.choice(len(H), size=WINDOW, replace=True)]
        win = H[rngn.choice(len(H), size=WINDOW, replace=True)]
        pc = w17.shrunk_cov(prof)
        stat_null.append(drift_stat(win, prof.mean(axis=0), pc, np.linalg.norm(pc)))
    threshold = float(np.percentile(stat_null, QUANTILE))
    log(f"  drift threshold ({QUANTILE}th percentile of clean captures): {threshold:.3f}")

    log("step 3/3: simulating drift over time")
    results = []
    for drift in DRIFTS:
        agg = {p: {"refits": [], "acc_mean": [], "acc_last": []} for p in ("never", "gated", "always")}
        for rep in range(REPS):
            r2 = np.random.default_rng(SEED + rep)
            crop = crops[rep % len(crops)]
            k = crops.index(crop)
            classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
            if len(classes) < 2:
                continue
            src_h = samples[(crop, "healthy")]
            cls_data = {d: samples[(crop, d)] for d in classes}
            th = test_s[(crop, "healthy")]
            pool = []
            state = {}
            for p in agg:
                state[p] = {"models": None, "prof": None, "refits": 0, "accs": []}
            for step in range(STEPS):
                shift = (step / (STEPS - 1)) * drift * u
                new = op(th[r2.choice(len(th), size=HEALTHY_STEP, replace=True)], k) + shift
                pool = (pool + list(new))[-WINDOW:]
                win = np.array(pool)
                Xt, yt = [], []
                for ci, d in enumerate(classes):
                    arr = test_s[(crop, d)]
                    Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k) + shift)
                    yt += [ci] * DISEASE_N
                Xt, yt = np.vstack(Xt), np.array(yt)
                for p in agg:
                    s = state[p]
                    do_fit = s["models"] is None
                    if not do_fit and p == "always":
                        do_fit = True
                    if not do_fit and p == "gated" and len(pool) >= WINDOW:
                        pm, pc = s["prof"]
                        do_fit = drift_stat(win, pm, pc, np.linalg.norm(pc)) > threshold
                    if len(pool) < WINDOW:
                        continue  # warm-up: no policy runs until the healthy window is full
                    if do_fit:
                        s["models"] = fit_models(src_h, cls_data, classes, win)
                        s["prof"] = (win.mean(axis=0), w17.shrunk_cov(win))
                        s["refits"] += 1
                    s["accs"].append(float(np.mean(w17.classify(Xt, s["models"]) == yt)))
            for p in agg:
                agg[p]["refits"].append(state[p]["refits"])
                agg[p]["acc_mean"].append(float(np.mean(state[p]["accs"])))
                agg[p]["acc_last"].append(state[p]["accs"][-1])
        for p in agg:
            results.append({"drift": drift, "policy": p,
                            "refits_per_run": round(float(np.mean(agg[p]["refits"])), 2),
                            "share_served_from_cache": round(1 - float(np.mean(agg[p]["refits"])) / (STEPS - WINDOW // HEALTHY_STEP + 1), 3),
                            "mean_accuracy": round(float(np.mean(agg[p]["acc_mean"])), 4),
                            "final_accuracy": round(float(np.mean(agg[p]["acc_last"])), 4),
                            "runs": len(agg[p]["refits"])})
        log(f"  drift {drift} done")
    with open(out / "drift_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "manifest.json").write_text(json.dumps({
        "module": "41_d3_drift_refit", "steps": STEPS, "window": WINDOW, "drifts": DRIFTS,
        "threshold": round(threshold, 4), "threshold_quantile": QUANTILE, "reps": REPS,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()

