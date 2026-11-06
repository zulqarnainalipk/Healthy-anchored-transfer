"""Module 42: D4, sequential agreement check with a refusal state (proposal section 6).

After each batch of healthy captures, the device recomputes the agreement bound between two sources' pushed
predictions (module 18 setup). The check PASSES when the bound is at or below a threshold tau; it stops at the
first pass. If no pass happens within the capture budget, the device enters a REFUSAL state (a distinct state,
not a low-confidence answer).

Threshold tau: set on a NULL case with no target data: the two sources are the same crop's class samples with no
different look (V = identity), so any disagreement is sampling noise. tau is the 95th percentile of the bound in
that null case.

Test worlds (strength 2, disjoint pools): a (affine; sources differ by a mild look V), b2 (non-composing),
e (rotation and shear, not a transport map), f (additive shift). Per world: the pass rate within the budget,
the mean number of captures to a decision, and the refusal rate.

Budget: STEPS batches of HEALTHY_STEP captures (the window is the most recent WINDOW images).

Outputs (results/42_d4_sequential/): sequential_results.csv, manifest.json

CPU module, runs locally.

Usage:
    python 42_d4_sequential.py --features ... --labelled ... --projection ... --output results/42_d4_sequential
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
s18 = importlib.util.spec_from_file_location("w18", HERE / "18_agreement_benchmark.py")
w18 = importlib.util.module_from_spec(s18)
s18.loader.exec_module(w18)

HEALTHY_STEP = 25
STEPS = 8
WINDOW = 200
MODEL_N = 100
NULL_REPS = 150
REPS = 60
WORLDS = ["a", "b2", "e", "f"]
SEED = 2026
QUANTILE = 95


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def agreement_bound(src_h1, src_h2, cls1, cls2, classes, H, r2):
    """Mean TV lower bound between the two sources' pushed samples, given the current target healthy window H."""
    mB, CB = H.mean(axis=0), w17.shrunk_cov(H)
    pushed = [[], []]
    for si, (sh, cls) in enumerate(((src_h1, cls1), (src_h2, cls2))):
        mA, CA = sh.mean(axis=0), w17.shrunk_cov(sh)
        T = w17.affine_T(CA, CB)
        for d in classes:
            arr = cls[d]
            pushed[si].append(mB + (arr[r2.choice(len(arr), size=MODEL_N, replace=True)] - mA) @ T.T)
    return float(np.mean([w18.tv_lower_bound(a, b, r2) for a, b in zip(pushed[0], pushed[1])]))


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
    V = w16.random_affine(np.random.default_rng(SEED), 0.15 * args.scale * 0.5)

    def classes_of(crop):
        return sorted(d for (c, d) in samples if c == crop and d != "healthy")

    log("step 2/3: threshold tau from the null case (identical sources, sampling noise only)")
    rng = np.random.default_rng(SEED)
    null = []
    for rep in range(NULL_REPS):
        crop = crops[rep % len(crops)]
        classes = classes_of(crop)
        if len(classes) < 2:
            continue
        H_src = samples[(crop, "healthy")]
        cls = {d: samples[(crop, d)] for d in classes}
        H = H_src[rng.choice(len(H_src), size=WINDOW, replace=True)]
        null.append(agreement_bound(H_src, H_src, cls, cls, classes, H, rng))
    tau = float(np.percentile(null, QUANTILE))
    log(f"  tau (95th percentile of the null bound) = {tau:.3f}")

    log("step 3/3: sequential decisions in the test worlds")
    results = []
    for name in WORLDS:
        orng = np.random.default_rng(w16.SEED)
        op, _ = w16.operators(name, orng, mu, len(crops), s=args.scale)
        passes, captures = [], []
        for rep in range(REPS):
            r2 = np.random.default_rng(SEED + rep)
            crop = crops[rep % len(crops)]
            k = crops.index(crop)
            classes = classes_of(crop)
            if len(classes) < 2:
                continue
            src1_h = samples[(crop, "healthy")]
            src2_h = src1_h @ V.T
            cls1 = {d: samples[(crop, d)] for d in classes}
            cls2 = {d: samples[(crop, d)] @ V.T for d in classes}
            th = test_s[(crop, "healthy")]
            pool = []
            decided = None
            for step in range(1, STEPS + 1):
                new = op(th[r2.choice(len(th), size=HEALTHY_STEP, replace=True)], k)
                pool = (pool + list(new))[-WINDOW:]
                if len(pool) < 50:
                    continue  # D1 gate: at least 50 captures
                b = agreement_bound(src1_h, src2_h, cls1, cls2, classes, np.array(pool), r2)
                if b <= tau:
                    decided = step
                    break
            passes.append(int(decided is not None))
            captures.append(decided if decided is not None else STEPS)
        results.append({"world": name, "runs": len(passes),
                        "pass_rate": round(float(np.mean(passes)), 4),
                        "refusal_rate": round(1 - float(np.mean(passes)), 4),
                        "mean_captures_to_decision": round(float(np.mean(captures)) * HEALTHY_STEP, 1)})
        log(f"  world {name}: pass rate {results[-1]['pass_rate']}")
    with open(out / "sequential_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "manifest.json").write_text(json.dumps({
        "module": "42_d4_sequential", "tau": round(tau, 4), "tau_quantile": QUANTILE, "null_reps": len(null),
        "steps": STEPS, "healthy_step": HEALTHY_STEP, "window": WINDOW, "reps": REPS,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
