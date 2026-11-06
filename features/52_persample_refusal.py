"""Module 52: per-sample refusal criteria (design check; exploratory).

The agreement check of modules 18 and 30 decides per replicate (per target crop and sample of images). A deployed system
would refuse per image. This module compares per-image scores for detecting the classifier's own errors:
  conf        maximum posterior probability of the pushed models (selective prediction by confidence)
  density     maximum log-likelihood of the image under the pushed class models (out-of-model score)
  disagree    the second source's classifier (same classes seen through a fixed mild random affine look, pushed through its
              own healthy-anchored map) predicts a different class than the first source's classifier
For each score: AUROC for detecting errors (pooled over images of all replicates) and accuracy when the 20% of images with
the least trustworthy score are refused (coverage 0.8). Methods: MC (mean-centring) and HATC.

Outputs (results/52_persample_refusal/): refusal.csv, manifest.json. CPU module, runs locally (a few minutes).

Usage:
    python 52_persample_refusal.py --features ... --labelled ... --projection ... --output results/52_persample_refusal
        --worlds a c50 e f --healthy-n 200 --reps 100
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w16 = _load("w16", "16_world_generator.py")
w17 = _load("w17", "17_benchmark_hatc.py")
w50 = _load("w50", "50_hatc_select.py")
m46 = _load("m46", "46b_metrics_cost.py")

DISEASE_N = 50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--worlds", nargs="+", default=["a", "c50", "e", "f"])
    ap.add_argument("--healthy-n", type=int, default=200)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--world-seed", type=int, default=2026)
    args = ap.parse_args()
    feats, out = Path(args.features), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading source samples and projection")
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
    samples = w16.harmonise_samples(samples)
    src_s, test_s, _ = w16.split_samples(samples)
    samples = src_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops")

    log("step 2/3: per-image scores")
    rows_out = []
    hn = args.healthy_n
    for wi, name in enumerate(args.worlds, 1):
        rng = np.random.default_rng(args.world_seed)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        V = w16.random_affine(np.random.default_rng(args.world_seed + 5), 0.15 * args.scale * 0.5, W.shape[1])
        crop_idx = {c: i for i, c in enumerate(crops)}
        store = {m: {"err": [], "conf": [], "dens": [], "dis": []} for m in ("MC", "HATC")}
        for rep in range(args.reps):
            r2 = np.random.default_rng(args.world_seed + 100 + rep)
            crop = crops[rep % len(crops)]
            k = crop_idx[crop]
            classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
            if len(classes) < 2:
                continue
            th = test_s[(crop, "healthy")]
            H_tgt = op(th[r2.choice(len(th), size=hn, replace=True)], k)
            cls_data = {d: samples[(crop, d)] for d in classes}
            Xt, yt = [], []
            for ci, d in enumerate(classes):
                arr = test_s[(crop, d)]
                Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                yt += [ci] * DISEASE_N
            Xt, yt = np.vstack(Xt), np.array(yt)
            src_h = samples[(crop, "healthy")]
            mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
            mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
            src_h2 = src_h @ V.T
            mA2, CA2 = src_h2.mean(axis=0), w17.shrunk_cov(src_h2)
            cls_data2 = {d: v @ V.T for d, v in cls_data.items()}
            for m in ("MC", "HATC"):
                models = m46.fit_models(w50.make_point_map(m, mA, CA, mB, CB), cls_data, classes)
                models2 = m46.fit_models(w50.make_point_map(m, mA2, CA2, mB, CB), cls_data2, classes)
                Pm = m46.scores(Xt, models)
                pred = Pm.argmax(axis=1)
                pred2 = m46.scores(Xt, models2).argmax(axis=1)
                dens = np.max(np.stack([w17.gauss_loglik(Xt, mm, CC) for mm, CC in models], axis=1), axis=1)
                store[m]["err"].append((pred != yt).astype(int))
                store[m]["conf"].append(Pm.max(axis=1))
                store[m]["dens"].append(dens)
                store[m]["dis"].append((pred != pred2).astype(float))
        for m, d in store.items():
            err = np.concatenate(d["err"])
            row_base = {"world": name, "healthy_n": hn, "method": m, "images": len(err), "accuracy": round(1 - err.mean(), 4)}
            for score, arr, sign in (("conf", np.concatenate(d["conf"]), -1), ("density", np.concatenate(d["dens"]), -1),
                                     ("disagree", np.concatenate(d["dis"]), 1)):
                badness = sign * arr
                auroc = float(roc_auc_score(err, badness)) if 0 < err.sum() < len(err) else float("nan")
                keep = np.argsort(badness)[: int(0.8 * len(err))]          # refuse the 20% with the worst score
                rows_out.append({**row_base, "score": score, "auroc_error_detection": round(auroc, 4),
                                 "accuracy_at_80pct_coverage": round(float(1 - err[keep].mean()), 4)})
        log(f"  world {name} done ({wi}/{len(args.worlds)})")

    log("step 3/3: writing outputs")
    with open(out / "refusal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (out / "manifest.json").write_text(json.dumps({"module": "52_persample_refusal", "worlds": args.worlds,
                                                    "healthy_n": hn, "reps": args.reps, "seconds": round(time.time() - t0, 1)},
                                                   indent=2), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
