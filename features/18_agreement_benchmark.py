"""Module 18: first test of the agreement check with two sources, on the synthetic worlds.

The agreement check compares what two source crops predict for the same disease on the target. A large
disagreement means at least one source is wrong (proposal section 5, step 5). This module tests whether the
disagreement measure predicts transfer failure in the synthetic worlds.

Two-source setup (a simplification, documented): source 1 is a real source crop's class samples; source 2
is the same classes seen through a fixed mild random affine look V (a different crop with the same diseases).
The target is the world operator applied to source 1's distribution. Each source is pushed to the target
through its own affine map, fitted from its own healthy sample to the target healthy sample.

For each replicate and each class:
  - pushed samples from source 1 and source 2 (MODEL_N each) are compared by a logistic-regression
    discriminator, evaluated on held-out samples (50 percent). The discriminator accuracy a gives a lower
    confidence bound on the total variation distance: TV >= 2a - 1 (Wilson lower bound on a, 95 percent).
  - the replicate's disagreement is the mean bound over classes.
Failure label: the replicate fails when the mean HATC accuracy of the two sources is below the
no-adaptation accuracy minus MARGIN (0.02).
Reported per world and healthy n: AUROC of the disagreement against the failure label, mean bound, and
the failure rate.

CPU module, runs locally.

Usage:
    python 18_agreement_benchmark.py --features ... --labelled ... --projection ... --output results/18_agreement
        --scale 3
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec16 = importlib.util.spec_from_file_location("w16", HERE / "16_world_generator.py")
w16 = importlib.util.module_from_spec(spec16)
spec16.loader.exec_module(w16)
spec17 = importlib.util.spec_from_file_location("w17", HERE / "17_benchmark_hatc.py")
w17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(w17)

REPS = 200
HEALTHY_N = [50, 200]
DISEASE_N = 50
MODEL_N = 100
MARGIN = 0.02
WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / (1 + z * z / n))


def tv_lower_bound(P1, P2, rng):
    """Lower confidence bound on TV between two sample sets, from a held-out discriminator."""
    n = min(len(P1), len(P2))
    idx1 = rng.permutation(len(P1))[:n]
    idx2 = rng.permutation(len(P2))[:n]
    h = n // 2
    Xtr = np.vstack([P1[idx1[:h]], P2[idx2[:h]]])
    ytr = np.array([0] * h + [1] * h)
    Xte = np.vstack([P1[idx1[h:]], P2[idx2[h:]]])
    yte = np.array([0] * (n - h) + [1] * (n - h))
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    correct = int((clf.predict(Xte) == yte).sum())
    a_lo = wilson_lower(correct, len(yte))
    return max(0.0, 2 * a_lo - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    global REPS
    if args.reps:
        REPS = args.reps
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
    src_s, test_s, train_s = w16.split_samples(samples)
    samples = src_s  # source models use the source part only; target draws use test_s / train_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops")

    log("step 2/3: running worlds x cells x replicates (two sources)")
    rng0 = np.random.default_rng(SEED)
    V = w16.random_affine(rng0, 0.15 * args.scale * 0.5)  # source 2's look
    v_inv = np.linalg.inv(V)
    result_rows = []
    for wi, name in enumerate(WORLDS, 1):
        rng = np.random.default_rng(w16.SEED)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in HEALTHY_N:
            bounds, fails = [], []
            for rep in range(REPS):
                r2 = np.random.default_rng(SEED + rep)
                crop = crops[rep % len(crops)]
                k = crop_idx[crop]
                classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
                if len(classes) < 2:
                    continue
                src_h = samples[(crop, "healthy")]
                th = test_s[(crop, "healthy")]
                H_tgt = op(th[r2.choice(len(th), size=hn, replace=True)], k)
                src1_h = src_h
                src2_h = src_h @ V.T
                Xt, yt = [], []
                models1, models2, pushed1, pushed2 = [], [], [], []
                for ci, d in enumerate(classes):
                    arr = test_s[(crop, d)]
                    Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                    yt += [ci] * DISEASE_N
                Xt = np.vstack(Xt)
                yt = np.array(yt)
                mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
                for src_h_i, own, store_models, store_push in ((src1_h, False, models1, pushed1),
                                                              (src2_h, True, models2, pushed2)):
                    mA, CA = src_h_i.mean(axis=0), w17.shrunk_cov(src_h_i)
                    T = w17.affine_T(CA, CB)
                    for d in classes:
                        arr = samples[(crop, d)]
                        cls = arr @ V.T if own else arr
                        m, C = cls.mean(axis=0), w17.shrunk_cov(cls)
                        store_models.append((mB + (m - mA) @ T.T, T @ C @ T.T))
                        store_push.append(mB + (cls[r2.choice(len(cls), size=MODEL_N, replace=True)] - mA) @ T.T)
                acc1 = float(np.mean(w17.classify(Xt, models1) == yt))
                acc2 = float(np.mean(w17.classify(Xt, models2) == yt))
                src_models_na = [(samples[(crop, d)].mean(axis=0), w17.shrunk_cov(samples[(crop, d)])) for d in classes]
                acc_na = float(np.mean(w17.classify(Xt, src_models_na) == yt))
                bound = float(np.mean([tv_lower_bound(a, b, r2) for a, b in zip(pushed1, pushed2)]))
                fail = int(((acc1 + acc2) / 2) < acc_na - MARGIN)
                bounds.append(bound)
                fails.append(fail)
            auroc = roc_auc_score(fails, bounds) if len(set(fails)) == 2 else float("nan")
            result_rows.append({"world": name, "healthy_n": hn, "replicates": len(bounds),
                                "failure_rate": round(float(np.mean(fails)), 4),
                                "mean_disagreement_bound": round(float(np.mean(bounds)), 4),
                                "auroc_bound_vs_failure": round(float(auroc), 4) if auroc == auroc else ""})
        log(f"  world {name} done ({wi}/{len(WORLDS)})")

    log("step 3/3: writing results")
    with open(out / "agreement_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        w.writeheader()
        w.writerows(result_rows)
    manifest = {"module": "18_agreement_benchmark", "worlds": WORLDS, "healthy_n": HEALTHY_N,
                "replicates": REPS, "model_n": MODEL_N, "margin": MARGIN, "operator_scale": args.scale,
                "setup": "two sources: real crop samples, and the same samples through a mild affine look V",
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
