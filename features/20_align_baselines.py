"""Module 20: alignment baselines on the synthetic worlds (proposal section 8, healthy-only arm).

Same worlds, replicates and metric as module 17 (single source), with more methods that use only the
target's healthy images:
  NA     no adaptation
  MC     mean-centring
  STD    per-dimension standardisation (AdaBN-like): scale each dimension by the ratio of target and
         source healthy standard deviations, and shift the means
  CORAL  second-order alignment: A = C_B^{1/2} C_A^{-1/2}, applied with a mean shift
  WHITEN whiten the target with its own healthy covariance and re-colour it with the source healthy
         covariance (the inverse alignment, applied to the target samples)
  HATC   the affine Monge map (module 11 logic)

About world e (rotation and shear). The proposal expects a general affine registration to win there. With
healthy distributions alone (no paired points), the rotation part of an affine map is not identifiable, so no
healthy-only method can recover it. CORAL and WHITEN are the general (non-symmetric) alignments that use only
distribution moments; they are reported here as the healthy-only affine alternatives.

Outputs (results/20_align_baselines/): replicate_results.csv, summary.csv, manifest.json.

CPU module, runs locally. Uses modules 16 and 17 for worlds and helpers.

Usage:
    python 20_align_baselines.py --features ... --labelled ... --projection ... --output results/20_align_baselines
        --scale 3
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

HERE = Path(__file__).resolve().parent
s16 = importlib.util.spec_from_file_location("w16", HERE / "16_world_generator.py")
w16 = importlib.util.module_from_spec(s16)
s16.loader.exec_module(w16)
s17 = importlib.util.spec_from_file_location("w17", HERE / "17_benchmark_hatc.py")
w17 = importlib.util.module_from_spec(s17)
s17.loader.exec_module(w17)

REPS = 200
HEALTHY_N = [50, 200]
DISEASE_N = 50
WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]
METHODS = ["NA", "MC", "STD", "CORAL", "WHITEN", "HATC"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inv_sqrt(C):
    return np.linalg.inv(sqrtm(C).real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=3.0)
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
    src_s, test_s, train_s = w16.split_samples(samples)
    samples = src_s  # source models use the source part only; target draws use test_s / train_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops")

    log("step 2/3: running worlds x cells x replicates")
    rep_rows = []
    for wi, name in enumerate(WORLDS, 1):
        rng = np.random.default_rng(w16.SEED)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in HEALTHY_N:
            for rep in range(REPS):
                r2 = np.random.default_rng(w16.SEED + rep)
                crop = crops[rep % len(crops)]
                k = crop_idx[crop]
                classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
                if len(classes) < 2:
                    continue
                src_h = samples[(crop, "healthy")]
                th = test_s[(crop, "healthy")]
                H_tgt = op(th[r2.choice(len(th), size=hn, replace=True)], k)
                cls_data = {d: samples[(crop, d)] for d in classes}
                Xt, yt = [], []
                for ci, d in enumerate(classes):
                    arr = test_s[(crop, d)]
                    Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                    yt += [ci] * DISEASE_N
                Xt, yt = np.vstack(Xt), np.array(yt)
                mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
                mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
                src_models = [(cls_data[d].mean(axis=0), w17.shrunk_cov(cls_data[d])) for d in classes]
                # per-method model lists (source class Gaussians moved to the target), or moved target data
                preds = {}
                preds["NA"] = w17.classify(Xt, src_models)
                preds["MC"] = w17.classify(Xt, [(m + (mB - mA), C) for m, C in src_models])
                dstd = np.sqrt(np.diag(CB) / np.diag(CA))
                D = np.diag(dstd)
                preds["STD"] = w17.classify(Xt, [(mB + D @ (m - mA), D @ C @ D) for m, C in src_models])
                A = sqrtm(CB).real @ inv_sqrt(CA)
                preds["CORAL"] = w17.classify(Xt, [(mB + A @ (m - mA), A @ C @ A.T) for m, C in src_models])
                B = sqrtm(CA).real @ inv_sqrt(CB)
                Xw = mA + (Xt - mB) @ B.T
                preds["WHITEN"] = w17.classify(Xw, src_models)
                T = w17.affine_T(CA, CB)
                preds["HATC"] = w17.classify(Xt, [(mB + (m - mA) @ T.T, T @ C @ T.T) for m, C in src_models])
                for m in METHODS:
                    rep_rows.append({"world": name, "healthy_n": hn, "replicate": rep, "crop": crop,
                                     "method": m, "accuracy": round(float(np.mean(preds[m] == yt)), 4)})
        log(f"  world {name} done ({wi}/{len(WORLDS)})")

    log("step 3/3: summary")
    with open(out / "replicate_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        w.writeheader()
        w.writerows(rep_rows)
    summary = []
    for name in WORLDS:
        for hn in HEALTHY_N:
            sub = [r for r in rep_rows if r["world"] == name and r["healthy_n"] == hn]
            row = {"world": name, "healthy_n": hn}
            for m in METHODS:
                row[f"acc_{m}"] = round(float(np.mean([r["accuracy"] for r in sub if r["method"] == m])), 4)
            best = max(METHODS, key=lambda m: row[f"acc_{m}"])
            row["best_method"] = best
            summary.append(row)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    manifest = {"module": "20_align_baselines", "methods": METHODS, "worlds": WORLDS,
                "healthy_n": HEALTHY_N, "replicates": REPS, "operator_scale": args.scale,
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
