"""Module 30: label-free reliability scores against the agreement bound (proposal section 8, certificate baselines).

Same two-source setup, worlds, disjoint pools and failure label as module 18. For each replicate the
following label-free scores are computed (all use only target images without labels, plus the source models):
  bound        mean discriminator lower bound on TV between the two sources' pushed samples (module 18)
  ens_disagree fraction of target samples on which the two sources' classifiers assign different classes
               (ensemble disagreement across sources, Jiang et al.)
  low_conf     1 - mean maximum posterior of the source-1 classifier on the target samples
               (max-softmax confidence; low confidence predicts failure)
  healthy_n    the healthy sample size of the target crop (a constant per cell, so its AUROC is 0.5 by
               construction within a cell; it is reported only as a reference)
Failure label: mean HATC accuracy of the two sources is below the no-adaptation accuracy by more than 0.02.

Output (results/30_certificate_baselines/): replicate_scores.csv, auroc.csv, manifest.json

CPU module, runs locally. Can be run on a subset of worlds (--worlds) in parallel jobs.

Usage:
    python 30_certificate_baselines.py --features ... --labelled ... --projection ... --output <folder> --scale 2
        --worlds a a2 b
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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

REPS = 200
HEALTHY_N = [50, 200]
DISEASE_N = 50
MODEL_N = 100
MARGIN = 0.02
SEED = 2026
WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]
SCORES = ["bound", "ens_disagree", "low_conf"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def posteriors(X, models):
    """Posterior class probabilities for rows of X under equal-prior Gaussian classes."""
    scores = np.vstack([w17.gauss_loglik(X, m, C) for m, C in models])
    scores -= scores.max(axis=0)
    p = np.exp(scores)
    return p / p.sum(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--worlds", nargs="+", default=None)
    args = ap.parse_args()
    global WORLDS
    if args.worlds:
        WORLDS = args.worlds
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
    samples = src_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops")

    log("step 2/3: running worlds x cells x replicates")
    rng0 = np.random.default_rng(SEED)
    V = w16.random_affine(rng0, 0.15 * args.scale * 0.5)
    rep_rows = []
    for wi, name in enumerate(WORLDS, 1):
        rng = np.random.default_rng(w16.SEED)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in HEALTHY_N:
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
                src2_h = src_h @ V.T
                Xt, yt = [], []
                for ci, d in enumerate(classes):
                    arr = test_s[(crop, d)]
                    Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                    yt += [ci] * DISEASE_N
                Xt, yt = np.vstack(Xt), np.array(yt)
                mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
                models, pushed = [[], []], [[], []]
                for si, (sh, own) in enumerate(((src_h, False), (src2_h, True))):
                    mA, CA = sh.mean(axis=0), w17.shrunk_cov(sh)
                    T = w17.affine_T(CA, CB)
                    for d in classes:
                        arr = samples[(crop, d)]
                        cls = arr @ V.T if own else arr
                        m, C = cls.mean(axis=0), w17.shrunk_cov(cls)
                        models[si].append((mB + (m - mA) @ T.T, T @ C @ T.T))
                        pushed[si].append(mB + (cls[r2.choice(len(cls), size=MODEL_N, replace=True)] - mA) @ T.T)
                pred1 = w17.classify(Xt, models[0])
                pred2 = w17.classify(Xt, models[1])
                acc1, acc2 = float(np.mean(pred1 == yt)), float(np.mean(pred2 == yt))
                na_models = [(samples[(crop, d)].mean(axis=0), w17.shrunk_cov(samples[(crop, d)])) for d in classes]
                acc_na = float(np.mean(w17.classify(Xt, na_models) == yt))
                bound = float(np.mean([w18.tv_lower_bound(a, b, r2) for a, b in zip(pushed[0], pushed[1])]))
                ens = float(np.mean(pred1 != pred2))
                low_conf = float(1.0 - np.mean(posteriors(Xt, models[0]).max(axis=0)))
                fail = int(((acc1 + acc2) / 2) < acc_na - MARGIN)
                rep_rows.append({"world": name, "healthy_n": hn, "replicate": rep, "crop": crop,
                                 "bound": round(bound, 4), "ens_disagree": round(ens, 4),
                                 "low_conf": round(low_conf, 4), "fail": fail})
        log(f"  world {name} done ({wi}/{len(WORLDS)})")

    log("step 3/3: AUROC per cell and score")
    with open(out / "replicate_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        w.writeheader()
        w.writerows(rep_rows)
    au = []
    for name in WORLDS:
        for hn in HEALTHY_N:
            sub = [r for r in rep_rows if r["world"] == name and r["healthy_n"] == hn]
            y = [r["fail"] for r in sub]
            row = {"world": name, "healthy_n": hn, "replicates": len(sub),
                   "failure_rate": round(float(np.mean(y)), 4)}
            for s in SCORES:
                row[f"auroc_{s}"] = (round(float(roc_auc_score(y, [r[s] for r in sub])), 4)
                                     if len(set(y)) == 2 else "")
            au.append(row)
    with open(out / "auroc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(au[0].keys()))
        w.writeheader()
        w.writerows(au)
    manifest = {"module": "30_certificate_baselines", "worlds": WORLDS, "scores": SCORES,
                "operator_scale": args.scale, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
