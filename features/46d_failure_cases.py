"""Module 46d: failure cases of HATC and the shrunk map on the synthetic worlds (proposal 8, failure-case section).

For the chosen worlds and healthy n, pools the target test predictions over all replicates and reports, per method:
  - per (crop, disease) recall, sorted, with the worst ten
  - the ten most frequent confusions (true class -> predicted class, same crop)
  - the most confidently wrong predictions: share of wrong predictions with probability above 0.9, and the ten
    (crop, true class) cells with the highest mean confidence when wrong
No images are shown: the synthetic worlds act on cached features, so there are no per-sample images to display.

Outputs (results/46d_failure_cases/<tag>/): per_class_recall.csv, top_confusions.csv, confident_errors.csv, manifest.json
CPU module, runs locally in about 30 seconds.

Usage:
    python 46d_failure_cases.py --features ... --labelled ... --projection ... --output results/46d_failure_cases
        --tag n200 --healthy-n 200 --worlds a e --reps 58
"""

import argparse
import csv
import importlib.util
import json
import time
from collections import defaultdict
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
m46 = _load("m46", "46b_metrics_cost.py")

DISEASE_N = 50
METHODS = ["HATC", "S50"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--healthy-n", type=int, default=200)
    ap.add_argument("--worlds", nargs="+", default=["a", "e"])
    ap.add_argument("--reps", type=int, default=58)
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

    log("step 2/3: pooling predictions over replicates")
    rec_rows, conf_rows, err_rows = [], [], []
    hn = args.healthy_n
    for name in args.worlds:
        rng = np.random.default_rng(args.world_seed)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        hits = {m: defaultdict(lambda: [0, 0]) for m in METHODS}          # (crop, class) -> [correct, total]
        confus = {m: defaultdict(int) for m in METHODS}                   # (crop, true, pred) -> count
        wrong_conf = {m: defaultdict(list) for m in METHODS}              # (crop, true) -> confidences when wrong
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
            for m in METHODS:
                Pm = m46.scores(Xt, m46.fit_models(w50.make_point_map(m, mA, CA, mB, CB), cls_data, classes))
                pred, conf = Pm.argmax(axis=1), Pm.max(axis=1)
                for y, p, c in zip(yt, pred, conf):
                    key = (crop, classes[y])
                    hits[m][key][1] += 1
                    if y == p:
                        hits[m][key][0] += 1
                    else:
                        confus[m][(crop, classes[y], classes[p])] += 1
                        wrong_conf[m][key].append(float(c))
        for m in METHODS:
            for (crop, d), (ok, tot) in hits[m].items():
                rec_rows.append({"world": name, "method": m, "crop": crop, "disease": d, "recall": round(ok / tot, 3), "n": tot})
            for (crop, t, p), c in sorted(confus[m].items(), key=lambda kv: -kv[1])[:10]:
                conf_rows.append({"world": name, "method": m, "crop": crop, "true": t, "predicted": p, "count": c})
            allw = [c for v in wrong_conf[m].values() for c in v]
            err_rows.append({"world": name, "method": m, "wrong_predictions": len(allw),
                             "share_wrong_above_0.9": round(float(np.mean(np.array(allw) > 0.9)), 3) if allw else 0})
        log(f"  world {name} done")

    log("step 3/3: writing outputs")
    rec_rows.sort(key=lambda r: (r["world"], r["method"], r["recall"]))
    write(out / "per_class_recall.csv", rec_rows)
    write(out / "top_confusions.csv", conf_rows)
    write(out / "confident_errors.csv", err_rows)
    (out / "manifest.json").write_text(json.dumps({
        "module": "46d_failure_cases", "healthy_n": hn, "worlds": args.worlds, "reps": args.reps,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
