"""Modules 12 and 44b: nonparametric (Brenier-type) host map, affine extension, and diversified predictions.

Same synthetic worlds, pools and replicates as modules 17, 20, 43 and 46b (disjoint source, test and probe pools).
Methods (accuracy):
  NA, MC, CORAL, HATC, S50              as in earlier modules
  BRENIER      empirical optimal transport between the source healthy sample and the target healthy sample (equal sizes
               n, squared Euclidean cost, exact assignment by the Hungarian algorithm), extended to new points by
               barycentric interpolation over the K = 10 nearest source healthy points
  BRENIER_AFF  the affine map (HATC) plus the interpolated residual of the empirical map (the "affine extension" of
               the proposal: the nonparametric correction on top of the affine part)
  HATC_BOOT    diversified predictions (module 44b): HATC fitted on 10 bootstrap resamples of the target healthy
               images, class probabilities averaged
  ENS          diversified predictions: average of the class probabilities of HATC and BRENIER_AFF
The empirical map needs at least 200 healthy images (proposal rule); for smaller n, BRENIER and BRENIER_AFF are not run.

Outputs (results/12_host_maps_brenier/<tag>/): summary.csv, manifest.json
CPU module, runs locally (the 400-image assignment takes well under a second).

Usage:
    python 12_host_maps_brenier.py --features ... --labelled ... --projection ... --output results/12_host_maps_brenier
        --tag k16 --healthy-n 200 400 --worlds a c50 e f --reps 100
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

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
K_NN = 10
BOOT = 10
METHODS = ["NA", "MC", "CORAL", "HATC", "S50", "BRENIER", "BRENIER_AFF", "HATC_BOOT", "ENS"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def brenier_maps(src, tgt, mA, CA, mB, CB):
    """Return (BRENIER, BRENIER_AFF) point maps from an exact empirical assignment of src to tgt."""
    n = min(len(src), len(tgt))
    S, T = src[:n], tgt[:n]
    r, c = linear_sum_assignment(cdist(S, T, "sqeuclidean"))
    matched = T[c]                                    # matched[i] is the image of S[r[i]]
    Sm = S[r]
    aff = w50.make_point_map("HATC", mA, CA, mB, CB)
    resid = matched - aff(Sm)

    def interp(X, kind):
        d = cdist(X, Sm, "sqeuclidean")
        nn = np.argsort(d, axis=1)[:, :K_NN]
        w = 1.0 / (np.take_along_axis(d, nn, axis=1) + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        if kind == "B":
            return np.einsum("nk,nkd->nd", w, matched[nn])
        return aff(X) + np.einsum("nk,nkd->nd", w, resid[nn])

    return (lambda X: interp(X, "B")), (lambda X: interp(X, "A"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--healthy-n", nargs="+", type=int, default=[200, 400])
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
    src_s, test_s, _ = w16.split_samples(samples)
    samples = src_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops; source healthy sizes "
        f"{sorted(len(samples[(c, 'healthy')]) for c in crops)}")

    log("step 2/3: worlds x healthy sizes x replicates")
    rows_out = []
    for wi, name in enumerate(args.worlds, 1):
        rng = np.random.default_rng(args.world_seed)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in args.healthy_n:
            acc = {m: [] for m in METHODS}
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
                mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
                mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
                probs = {}
                for m in ("NA", "MC", "CORAL", "HATC", "S50"):
                    probs[m] = m46.scores(Xt, m46.fit_models(w50.make_point_map(m, mA, CA, mB, CB), cls_data, classes))
                if hn >= 200 and len(src_h) >= 50:
                    gb, ga = brenier_maps(src_h[r2.permutation(len(src_h))], H_tgt, mA, CA, mB, CB)
                    probs["BRENIER"] = m46.scores(Xt, m46.fit_models(gb, cls_data, classes))
                    probs["BRENIER_AFF"] = m46.scores(Xt, m46.fit_models(ga, cls_data, classes))
                    probs["ENS"] = (probs["HATC"] + probs["BRENIER_AFF"]) / 2
                bp = []
                for _ in range(BOOT):
                    Hb = H_tgt[r2.integers(0, len(H_tgt), len(H_tgt))]
                    g = w50.make_point_map("HATC", mA, CA, Hb.mean(axis=0), w17.shrunk_cov(Hb))
                    bp.append(m46.scores(Xt, m46.fit_models(g, cls_data, classes)))
                probs["HATC_BOOT"] = np.mean(bp, axis=0)
                for m, Pm in probs.items():
                    acc[m].append(float(np.mean(Pm.argmax(axis=1) == yt)))
            row = {"world": name, "healthy_n": hn, "replicates": len(acc["NA"])}
            for m in METHODS:
                row[f"acc_{m}"] = round(float(np.mean(acc[m])), 4) if acc[m] else ""
            rows_out.append(row)
            log(f"  world {name}, n = {hn}: HATC {row['acc_HATC']}, BRENIER {row['acc_BRENIER']}, "
                f"BRENIER_AFF {row['acc_BRENIER_AFF']}, ENS {row['acc_ENS']}")
        log(f"  world {name} done ({wi}/{len(args.worlds)})")

    log("step 3/3: writing summary")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (out / "manifest.json").write_text(json.dumps({
        "module": "12_host_maps_brenier", "healthy_n": args.healthy_n, "worlds": args.worlds, "reps": args.reps,
        "knn": K_NN, "bootstrap_maps": BOOT, "operator_scale": args.scale, "world_seed": args.world_seed,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()
