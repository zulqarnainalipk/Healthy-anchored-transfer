"""Module 43: ablation engine for the synthetic worlds (proposal section 8, ablations).

Runs the fixed alignments of modules 20 and 50 (NA, MC, SCALE, STD, S25, S50, S75, CORAL, HATC) over a chosen
projection, set of healthy-image counts and set of worlds, with the same disjoint pools and replicate structure as
modules 17 and 20. Used for:
  - the healthy-sample sweep (how many healthy images does each alignment need?)
  - the shrinkage sweep (S25, S50, S75 against the full map)
  - the projection ablations of module 44 (dimension, PCA against the supervised projection)

Outputs (results/43_ablations/<tag>/): summary.csv (mean accuracy per world, healthy n, method), manifest.json

CPU module, runs locally. Uses modules 16, 17 and 50.

Usage:
    python 43_ablation_engine.py --features ... --labelled ... --projection <npz> --output <folder> --tag <name>
        --healthy-n 10 25 50 100 200 400 --worlds a c50 e --reps 100 --scale 2
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
CANDIDATES = ["NA", "MC", "SCALE", "STD", "S25", "S50", "S75", "CORAL", "HATC"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--healthy-n", nargs="+", type=int, default=[50, 200])
    ap.add_argument("--worlds", nargs="+", default=["a", "c50", "e"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--world-seed", type=int, default=2026)
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="fraction of source class labels flipped to another class of the same crop (module 35)")
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log(f"step 1/3: loading source samples, projection {args.projection}")
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
    dim = int(W.shape[1])
    log(f"  {len(samples)} groups, {len(crops)} crops, {dim} dimensions")

    log("step 2/3: running worlds x healthy sizes x replicates")
    rows_out = []
    for wi, name in enumerate(args.worlds, 1):
        rng = np.random.default_rng(args.world_seed)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for hn in args.healthy_n:
            accs = {m: [] for m in CANDIDATES}
            for rep in range(args.reps):
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
                if args.label_noise > 0:
                    # symmetric label noise on the SOURCE labels: a sample keeps its place but takes another class
                    allx = np.vstack([cls_data[d] for d in classes])
                    ally = np.concatenate([[ci] * len(cls_data[d]) for ci, d in enumerate(classes)])
                    flip = r2.random(len(ally)) < args.label_noise
                    ally = ally.copy()
                    for i in np.where(flip)[0]:
                        ally[i] = r2.choice([c for c in range(len(classes)) if c != ally[i]])
                    cls_data = {d: allx[ally == ci] for ci, d in enumerate(classes)}
                    if any(len(v) < 5 for v in cls_data.values()):
                        continue
                Xt, yt = [], []
                for ci, d in enumerate(classes):
                    arr = test_s[(crop, d)]
                    Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                    yt += [ci] * DISEASE_N
                Xt, yt = np.vstack(Xt), np.array(yt)
                mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
                mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
                for cand in CANDIDATES:
                    g = w50.make_point_map(cand, mA, CA, mB, CB)
                    models = []
                    for d in classes:
                        Y = g(cls_data[d])
                        models.append((Y.mean(axis=0), w17.shrunk_cov(Y)))
                    accs[cand].append(float(np.mean(w17.classify(Xt, models) == yt)))
            row = {"world": name, "healthy_n": hn, "dimensions": dim, "replicates": len(accs["NA"])}
            for m in CANDIDATES:
                row[f"acc_{m}"] = round(float(np.mean(accs[m])), 4)
            best_simple = max(row["acc_NA"], row["acc_MC"], row["acc_STD"], row["acc_SCALE"])
            row["HATC_minus_best_simple"] = round(row["acc_HATC"] - best_simple, 4)
            row["S50_minus_best_simple"] = round(row["acc_S50"] - best_simple, 4)
            rows_out.append(row)
        log(f"  world {name} done ({wi}/{len(args.worlds)})")

    log("step 3/3: writing summary")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (out / "manifest.json").write_text(json.dumps({
        "module": "43_ablation_engine", "tag": args.tag, "projection": args.projection,
        "dimensions": dim, "healthy_n": args.healthy_n, "worlds": args.worlds, "reps": args.reps,
        "operator_scale": args.scale, "world_seed": args.world_seed, "label_noise": args.label_noise,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {round(time.time() - t0, 1)} s")


if __name__ == "__main__":
    main()

