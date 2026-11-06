"""Module 50: HATC-S, healthy-validated alignment selection (post-hoc extension; exploratory).

Motivation (modules 17, 20, 37): no single fixed alignment wins in every world. Mean-centring wins for shifts,
standardisation for scaling, and the affine map only when the relation is close to affine and healthy images
are plentiful. A device always has the target's healthy images, which need no labels. HATC-S uses them to
VALIDATE the alignment:

  1. Candidates: NA (identity), MC (mean shift), STD (per-dimension), CORAL, HATC (affine Monge map), and a
     shrunk map S50 = (1 - 0.5) I + 0.5 T with the mean shift.
  2. For each candidate, fit it on the source healthy sample and half of the target healthy sample, transport
     the source healthy sample, and measure the Frechet (Bures-Wasserstein) distance to the OTHER half of the
     target healthy sample. Swap halves and average (two-fold cross-fitting).
  3. Select the candidate with the lowest healthy transport error. Refit it on all target healthy images and
     use it to classify target disease samples (source class Gaussians moved by the chosen alignment).
  4. The lowest healthy transport error is also a label-free reliability score (higher = less reliable).

No disease labels and no target disease images are used in the selection.

This is a design change made after the registered benchmark (module 20) was read. The registered method,
HATC, stays the primary method; HATC-S is reported as an exploratory extension, and it is tested on freshly
generated worlds (different operator seeds) as well as the original worlds.

Outputs (results/50_hatc_select/<tag>/): replicate_results.csv, summary.csv, selection_counts.csv,
certificate_auroc.csv, manifest.json

CPU module, runs locally. Uses modules 16 and 17.

Usage:
    python 50_hatc_select.py --features ... --labelled ... --projection ... --output <folder> --scale 2
        --world-seed 2026 --tag original
"""

import argparse
import collections
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import roc_auc_score

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
MARGIN = 0.02
WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]
CANDIDATES = ["NA", "MC", "SCALE", "STD", "S25", "S50", "S75", "CORAL", "HATC"]
SIMPLEST_FIRST = CANDIDATES  # order of simplicity, used by the parsimony rule
TOL = 0.05                  # parsimony: take the simplest candidate within (1 + TOL) of the minimum error


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def inv_sqrt(C):
    return np.linalg.inv(sqrtm(C).real)


def make_point_map(name, mA, CA, mB, CB):
    """Return g(x) mapping source points to the target, for a candidate alignment."""
    if name == "NA":
        return lambda X: X
    if name == "MC":
        return lambda X: X + (mB - mA)
    if name == "SCALE":
        s = np.sqrt(np.trace(CB) / np.trace(CA))
        return lambda X: mB + (X - mA) * s
    if name == "STD":
        s = np.sqrt(np.diag(CB) / np.diag(CA))
        return lambda X: mB + (X - mA) * s
    if name == "CORAL":
        A = sqrtm(CB).real @ inv_sqrt(CA)
        return lambda X: mB + (X - mA) @ A.T
    T = w17.affine_T(CA, CB)
    if name == "HATC":
        return lambda X: mB + (X - mA) @ T.T
    if name in ("S25", "S50", "S75"):
        lam = {"S25": 0.25, "S50": 0.5, "S75": 0.75}[name]
        M = (1 - lam) * np.eye(len(mA)) + lam * T
        return lambda X: mB + (X - mA) @ M.T
    raise ValueError(name)


def frechet(m1, C1, m2, C2):
    s1 = sqrtm(C1).real
    cross = sqrtm(s1 @ C2 @ s1).real
    return float(np.sum((m1 - m2) ** 2) + np.trace(C1 + C2 - 2 * cross))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--world-seed", type=int, default=2026)
    ap.add_argument("--tag", default="original")
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
    samples = src_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops; world seed {args.world_seed}")

    log("step 2/3: running worlds x cells x replicates")
    rep_rows, sel_counts, cert = [], collections.Counter(), []
    for wi, name in enumerate(WORLDS, 1):
        rng = np.random.default_rng(args.world_seed)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in HEALTHY_N:
            for rep in range(REPS):
                r2 = np.random.default_rng(args.world_seed + 100 + rep)
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
                src_models = [(cls_data[d].mean(axis=0), w17.shrunk_cov(cls_data[d])) for d in classes]
                mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
                # healthy cross-fitting
                perm = r2.permutation(len(H_tgt))
                halves = (H_tgt[perm[: len(perm) // 2]], H_tgt[perm[len(perm) // 2:]])
                err = {}
                for cand in CANDIDATES:
                    e = []
                    for fit, held in ((halves[0], halves[1]), (halves[1], halves[0])):
                        g = make_point_map(cand, mA, CA, fit.mean(axis=0), w17.shrunk_cov(fit))
                        mapped = g(src_h)
                        e.append(frechet(mapped.mean(axis=0), w17.shrunk_cov(mapped),
                                         held.mean(axis=0), w17.shrunk_cov(held)))
                    err[cand] = float(np.mean(e))
                lowest = min(err.values())
                chosen = next(c for c in SIMPLEST_FIRST if err[c] <= (1 + TOL) * lowest)
                sel_counts[(name, hn, chosen)] += 1
                # accuracy of each candidate fitted on all target healthy images
                mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
                acc = {}
                for cand in CANDIDATES:
                    g = make_point_map(cand, mA, CA, mB, CB)
                    # class Gaussians moved by sampling the class samples through the point map
                    models = []
                    for d in classes:
                        Y = g(cls_data[d])
                        models.append((Y.mean(axis=0), w17.shrunk_cov(Y)))
                    acc[cand] = float(np.mean(w17.classify(Xt, models) == yt))
                acc["SELECT"] = acc[chosen]
                oracle = max(acc[c] for c in CANDIDATES)
                for m in CANDIDATES + ["SELECT"]:
                    rep_rows.append({"world": name, "healthy_n": hn, "replicate": rep, "crop": crop,
                                     "method": m, "accuracy": round(acc[m], 4)})
                cert.append({"world": name, "healthy_n": hn, "min_error": err[chosen],
                             "fail": int(acc["SELECT"] < acc["NA"] - MARGIN)})
        log(f"  world {name} done ({wi}/{len(WORLDS)})")

    log("step 3/3: summaries")
    with open(out / "replicate_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        w.writeheader()
        w.writerows(rep_rows)
    methods = CANDIDATES + ["SELECT"]
    summary = []
    for name in WORLDS:
        for hn in HEALTHY_N:
            sub = [r for r in rep_rows if r["world"] == name and r["healthy_n"] == hn]
            row = {"world": name, "healthy_n": hn}
            for m in methods:
                row[f"acc_{m}"] = round(float(np.mean([r["accuracy"] for r in sub if r["method"] == m])), 4)
            best_fixed = max(CANDIDATES, key=lambda m: row[f"acc_{m}"])
            row["best_fixed"] = best_fixed
            row["select_minus_best_fixed"] = round(row["acc_SELECT"] - row[f"acc_{best_fixed}"], 4)
            row["select_minus_HATC"] = round(row["acc_SELECT"] - row["acc_HATC"], 4)
            summary.append(row)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(out / "selection_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["world", "healthy_n", "chosen", "count"])
        for (name, hn, ch), c in sorted(sel_counts.items()):
            w.writerow([name, hn, ch, c])
    au = []
    for name in WORLDS:
        for hn in HEALTHY_N:
            sub = [c for c in cert if c["world"] == name and c["healthy_n"] == hn]
            y = [c["fail"] for c in sub]
            au.append({"world": name, "healthy_n": hn, "failure_rate": round(float(np.mean(y)), 4),
                       "auroc_min_healthy_error": (round(float(roc_auc_score(y, [c["min_error"] for c in sub])), 4)
                                                  if len(set(y)) == 2 else "")})
    with open(out / "certificate_auroc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(au[0].keys()))
        w.writeheader()
        w.writerows(au)
    (out / "manifest.json").write_text(json.dumps({
        "module": "50_hatc_select", "tag": args.tag, "world_seed": args.world_seed, "operator_scale": args.scale,
        "candidates": CANDIDATES, "replicates": REPS, "seconds": round(time.time() - t0, 1),
        "parsimony_tolerance": TOL, "status": "post-hoc, exploratory extension; registered method is HATC; iteration 2"}, indent=2), encoding="utf-8")
    log(f"done in {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()

