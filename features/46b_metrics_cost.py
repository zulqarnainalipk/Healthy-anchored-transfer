"""Modules 23, 27, 46b and 46c in one engine: extra baselines, per-class metrics, calibration and cost.

Same synthetic worlds, pools and replicate structure as modules 17, 20 and 43 (disjoint source, test and probe pools).
Methods:
  healthy-only arm:   NA, MC, STD, CORAL, HATC, S50 (module 50 maps)
  unlabelled-field arm (module 23, "-style" reimplementations on frozen features; they use 150 UNLABELLED target
                      disease images per replicate, which the healthy-only arm does not):
      ADABN   standardise the unlabelled target disease images with their own mean and std (AdaBN-style), then
              use the NA class models
      SHOT    start from the MC class models, pseudo-label the unlabelled target disease images, move each class
              mean towards the mean of its pseudo-labelled images (3 rounds, centroid pseudo-labelling)
  labelled-target arm (module 27, pooled source): source class samples (unaligned) plus K labelled target images per
              class (K = 5 and 25), one shrunk Gaussian per class (POOL5, POOL25)
Metrics per method (module 46b): accuracy, macro recall, worst-class recall, Brier score, expected calibration error
(10 bins, top class) using softmax of the class log-likelihoods with equal priors.
Cost (module 46c): mean fit-plus-classify seconds per replicate for each method (one process, one machine).

Outputs (results/46b_metrics_cost/<tag>/): metrics.csv (world, healthy n, method, metrics, seconds), manifest.json
CPU module, runs locally.

Usage:
    python 46b_metrics_cost.py --features ... --labelled ... --projection ... --output results/46b_metrics_cost
        --tag n200 --healthy-n 200 --worlds a c50 e f --reps 100
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w16 = _load("w16", "16_world_generator.py")
w17 = _load("w17", "17_benchmark_hatc.py")
w50 = _load("w50", "50_hatc_select.py")

DISEASE_N = 50
UNLAB_N = 150
METHODS = ["NA", "MC", "STD", "CORAL", "HATC", "S50", "ADABN", "SHOT", "POOL5", "POOL25"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scores(X, models):
    """Class probabilities from Gaussian log-likelihoods, equal priors (softmax)."""
    L = np.stack([w17.gauss_loglik(X, m, C) for m, C in models], axis=1)
    L = L - L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=1, keepdims=True)


def metrics(P, y, k):
    pred = P.argmax(axis=1)
    acc = float(np.mean(pred == y))
    rec = [float(np.mean(pred[y == c] == c)) for c in range(k)]
    brier = float(np.mean(np.sum((P - np.eye(k)[y]) ** 2, axis=1)))
    conf = P.max(axis=1)
    ece, edges = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(np.mean(pred[m] == y[m]) - conf[m].mean())
    return acc, float(np.mean(rec)), float(np.min(rec)), brier, float(ece)


def fit_models(g, cls_data, classes):
    out = []
    for d in classes:
        Y = g(cls_data[d])
        out.append((Y.mean(axis=0), w17.shrunk_cov(Y)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--healthy-n", type=int, default=200)
    ap.add_argument("--worlds", nargs="+", default=["a", "c50", "e", "f"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--world-seed", type=int, default=2026)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output) / args.tag
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
    if __import__("os").environ.get("HARMONISE", "1") == "1":   # merge same-disease names inside a crop (module 03b)
        samples = w16.harmonise_samples(samples)
    src_s, test_s, probe_s = w16.split_samples(samples)
    samples = src_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops")

    log("step 2/3: worlds x replicates x methods")
    rows_out = []
    hn = args.healthy_n
    for wi, name in enumerate(args.worlds, 1):
        rng = np.random.default_rng(args.world_seed)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        acc = {m: [] for m in METHODS}
        secs = {m: [] for m in METHODS}
        for rep in range(args.reps):
            r2 = np.random.default_rng(args.world_seed + 100 + rep)
            crop = crops[rep % len(crops)]
            k = crop_idx[crop]
            classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
            if len(classes) < 2:
                continue
            kc = len(classes)
            th = test_s[(crop, "healthy")]
            H_tgt = op(th[r2.choice(len(th), size=hn, replace=True)], k)
            cls_data = {d: samples[(crop, d)] for d in classes}
            Xt, yt = [], []
            for ci, d in enumerate(classes):
                arr = test_s[(crop, d)]
                Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                yt += [ci] * DISEASE_N
            Xt, yt = np.vstack(Xt), np.array(yt)
            # unlabelled target disease images and labelled target images come from the probe pool
            U = np.vstack([op(probe_s[(crop, d)][r2.choice(len(probe_s[(crop, d)]), size=UNLAB_N // kc + 1, replace=True)], k)
                           for d in classes])
            lab = {d: op(probe_s[(crop, d)][r2.choice(len(probe_s[(crop, d)]), size=25, replace=True)], k) for d in classes}
            src_h = samples[(crop, "healthy")]
            mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
            mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
            for m in METHODS:
                t1 = time.perf_counter()
                Xe = Xt
                if m in ("NA", "MC", "STD", "CORAL", "HATC", "S50"):
                    models = fit_models(w50.make_point_map(m, mA, CA, mB, CB), cls_data, classes)
                elif m == "ADABN":
                    models = fit_models(lambda X: X, cls_data, classes)
                    sU = U.std(axis=0) + 1e-6
                    sS = np.vstack(list(cls_data.values())).std(axis=0) + 1e-6
                    Xe = (Xt - U.mean(axis=0)) / sU * sS + np.vstack(list(cls_data.values())).mean(axis=0)
                elif m == "SHOT":
                    models = fit_models(w50.make_point_map("MC", mA, CA, mB, CB), cls_data, classes)
                    for _ in range(3):
                        lab_u = scores(U, models).argmax(axis=1)
                        models = [((models[c][0] + U[lab_u == c].mean(axis=0)) / 2 if (lab_u == c).sum() >= 3 else models[c][0],
                                   models[c][1]) for c in range(kc)]
                else:
                    kk = 5 if m == "POOL5" else 25
                    models = []
                    for d in classes:
                        Y = np.vstack([cls_data[d], lab[d][:kk]])
                        models.append((Y.mean(axis=0), w17.shrunk_cov(Y)))
                Pm = scores(Xe, models)
                secs[m].append(time.perf_counter() - t1)
                acc[m].append(metrics(Pm, yt, kc))
        for m in METHODS:
            a = np.array(acc[m])
            rows_out.append({"world": name, "healthy_n": hn, "method": m, "replicates": len(a),
                             "accuracy": round(float(a[:, 0].mean()), 4), "macro_recall": round(float(a[:, 1].mean()), 4),
                             "worst_class_recall": round(float(a[:, 2].mean()), 4), "brier": round(float(a[:, 3].mean()), 4),
                             "ece": round(float(a[:, 4].mean()), 4), "ms_per_replicate": round(1000 * float(np.mean(secs[m])), 2)})
        log(f"  world {name} done ({wi}/{len(args.worlds)})")

    log("step 3/3: writing metrics")
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (out / "manifest.json").write_text(json.dumps({
        "module": "46b_metrics_cost", "healthy_n": hn, "worlds": args.worlds, "reps": args.reps, "operator_scale": args.scale,
        "unlabelled_target_images": UNLAB_N, "world_seed": args.world_seed, "seconds": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    log(f"done in {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()
