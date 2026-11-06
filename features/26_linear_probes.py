"""Module 26: few-shot linear probes on the label grid, synthetic worlds (proposal section 5, labelled saving).

For each world and replicate (same replicate and test samples as modules 17 and 20), a logistic-regression
classifier is trained on labelled TARGET disease samples at 0, 5, 10, 25, 50, 100 and 200 labels per class,
then tested on the same held-out target samples. At 0 labels the score is chance (1 / number of classes).

The probe sees the same number of healthy images as HATC only through the dimension of its features; it
does not use healthy images (the probe classifies among disease classes). Probe accuracy is compared with
HATC accuracy from module 20 in module 31.

Outputs (results/26_linear_probes/):
  - probe_replicates.csv    one row per (world, replicate, labels): probe accuracy
  - probe_curve.csv         mean accuracy per (world, labels)
  - manifest.json

CPU module, runs locally. Uses modules 16 and 17 for worlds and helpers.

Usage:
    python 26_linear_probes.py --features ... --labelled ... --projection ... --output results/26_linear_probes
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

HERE = Path(__file__).resolve().parent
s16 = importlib.util.spec_from_file_location("w16", HERE / "16_world_generator.py")
w16 = importlib.util.module_from_spec(s16)
s16.loader.exec_module(w16)

REPS = 200
DISEASE_N = 50
GRID = [0, 5, 10, 25, 50, 100, 200]
WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--worlds", nargs="+", default=None, help="subset of worlds (default: all)")
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
    samples = src_s  # source models use the source part only; target draws use test_s / train_s
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    mu = np.vstack(list(samples.values())).mean(axis=0)
    log(f"  {len(samples)} groups, {len(crops)} crops")

    log("step 2/3: probes per world x replicate x label count")
    rows_out = []
    for wi, name in enumerate(WORLDS, 1):
        rng = np.random.default_rng(w16.SEED)
        op, _ = w16.operators(name, rng, mu, len(crops), s=args.scale)
        crop_idx = {c: i for i, c in enumerate(crops)}
        for rep in range(REPS):
            r2 = np.random.default_rng(w16.SEED + rep)
            crop = crops[rep % len(crops)]
            k = crop_idx[crop]
            classes = sorted(d for (c, d) in samples if c == crop and d != "healthy")
            if len(classes) < 2:
                continue
            # same draw order as modules 17 and 20 up to the test samples (healthy draw is first)
            src_h = samples[(crop, "healthy")]
            _ = r2.choice(len(src_h), size=50, replace=True)  # keeps the stream aligned for hn = 50 draws
            Xt, yt = [], []
            for ci, d in enumerate(classes):
                arr = test_s[(crop, d)]
                Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
                yt += [ci] * DISEASE_N
            Xt, yt = np.vstack(Xt), np.array(yt)
            r3 = np.random.default_rng(SEED_PROBE + rep)
            for n_lab in GRID:
                if n_lab == 0:
                    acc = 1.0 / len(classes)
                else:
                    Xtr, ytr = [], []
                    for ci, d in enumerate(classes):
                        arr = train_s[(crop, d)]
                        Xtr.append(op(arr[r3.choice(len(arr), size=n_lab, replace=True)], k))
                        ytr += [ci] * n_lab
                    clf = LogisticRegression(max_iter=300).fit(np.vstack(Xtr), np.array(ytr))
                    acc = float(np.mean(clf.predict(Xt) == yt))
                rows_out.append({"world": name, "replicate": rep, "crop": crop,
                                 "labels_per_class": n_lab, "accuracy": round(acc, 4)})
        log(f"  world {name} done ({wi}/{len(WORLDS)})")

    log("step 3/3: writing results")
    with open(out / "probe_replicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    curve = []
    for name in WORLDS:
        for g in GRID:
            sub = [r["accuracy"] for r in rows_out if r["world"] == name and r["labels_per_class"] == g]
            curve.append({"world": name, "labels_per_class": g, "mean_accuracy": round(float(np.mean(sub)), 4),
                          "replicates": len(sub)})
    with open(out / "probe_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    manifest = {"module": "26_linear_probes", "grid": GRID, "worlds": WORLDS, "replicates": REPS,
                "operator_scale": args.scale, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


SEED_PROBE = 7000

if __name__ == "__main__":
    main()

