"""Module 03b: background shortcut check (audit items P4 and P5).

Question: can the background alone predict the class? If so, a classifier could learn the
background instead of the leaf. This matters most for peach (mostly black backgrounds) and chili.

Method, for each image of the chosen dataset:
  - mean brightness of the four 16 x 16 corner patches (0 = black, 255 = white)
  - a background label: black (< 30), white (> 225), or other
Then, per class:
  - the number of images and the share of each background label
  - a shortcut test: a single brightness threshold chosen on the training folds predicts the
    class. Its accuracy is reported on held-out folds (5-fold, stratified by class). An accuracy
    well above the majority-class rate means the background carries class information.

CPU module. Reads images for the chosen datasets only (small: peach 182, chili 6,566).
Logs progress every 1,000 images.

Usage:
    python 03_background_check.py --labelled results/01_crop_disease_table/labelled_images.csv
        --data-root "<data-root>" --datasets peach_leaf_damage chili_yellow_leaf
        --output results/03_background_check
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

PATCH = 16
BLACK, WHITE = 30, 225
FOLDS = 5
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def corner_brightness(path):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"), dtype=np.float64)
    h, w = g.shape
    p = PATCH
    corners = [g[:p, :p], g[:p, w - p:], g[h - p:, :p], g[h - p:, w - p:]]
    return float(np.mean([c.mean() for c in corners]))


def background_label(b):
    if b < BLACK:
        return "black"
    if b > WHITE:
        return "white"
    return "other"


def best_threshold(values, labels):
    """Threshold on brightness that maximises accuracy of a 'class A if above' rule, on training data."""
    order = np.argsort(values)
    v = np.asarray(values)[order]
    y = np.asarray(labels)[order]
    best_acc, best_t = -1.0, float(np.median(v))
    for i in range(1, len(v)):
        if v[i] == v[i - 1]:
            continue
        t = (v[i] + v[i - 1]) / 2
        pred = v > t
        acc = max(np.mean(pred == y), np.mean(pred != y))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def shortcut_accuracy(values, labels, seed):
    """Five-fold, stratified by class: threshold learned on training folds, accuracy on the held-out fold."""
    classes = sorted(set(labels))
    if len(classes) != 2:
        return None
    rng = random.Random(seed)
    idx = {c: [i for i, l in enumerate(labels) if l == c] for c in classes}
    fold_of = {}
    for c in classes:
        ids = idx[c][:]
        rng.shuffle(ids)
        for k, i in enumerate(ids):
            fold_of[i] = k % FOLDS
    correct, total = 0, 0
    maj = max(len(idx[c]) for c in classes) / len(labels)
    y = np.array([1 if l == classes[1] else 0 for l in labels])
    v = np.asarray(values)
    for f in range(FOLDS):
        tr = [i for i in range(len(labels)) if fold_of[i] != f]
        te = [i for i in range(len(labels)) if fold_of[i] == f]
        t = best_threshold(v[tr], y[tr])
        pred = (v[te] > t).astype(int)
        # orientation: choose the direction that did better on the training fold
        flip = np.mean((v[tr] > t).astype(int) == y[tr]) < 0.5
        if flip:
            pred = 1 - pred
        correct += int(np.sum(pred == y[te]))
        total += len(te)
    return {"shortcut_accuracy": round(correct / total, 4), "majority_rate": round(maj, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    log("step 1/3: reading labelled table")
    with open(args.labelled, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == "image" and r["dataset"] in args.datasets
                and r["eligible_for_evaluation"] == "1"]
    log(f"  {len(rows)} eligible images across {args.datasets}")

    log("step 2/3: corner brightness for every image")
    results = []
    for i, r in enumerate(rows, 1):
        if i % 1000 == 0:
            log(f"  {i}/{len(rows)}")
        b = corner_brightness(root / r["path"])
        results.append({"dataset": r["dataset"], "class": r["disease"] or r["crop"],
                        "is_healthy": int(r["is_healthy"]), "brightness": round(b, 2),
                        "background": background_label(b), "path": r["path"]})

    log("step 3/3: summaries and shortcut test")
    with open(out / "background_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    summary = {}
    for ds in args.datasets:
        rs = [r for r in results if r["dataset"] == ds]
        classes = sorted({r["class"] for r in rs})
        per_class = {}
        for c in classes:
            cs = [r for r in rs if r["class"] == c]
            per_class[c] = {
                "n": len(cs),
                "black": sum(r["background"] == "black" for r in cs),
                "white": sum(r["background"] == "white" for r in cs),
                "other": sum(r["background"] == "other" for r in cs),
                "mean_brightness": round(float(np.mean([r["brightness"] for r in cs])), 2),
            }
        # Shortcut test on the binary healthy-or-not split (the class the proposal cares about),
        # plus the full class split when only two classes exist.
        values = [r["brightness"] for r in rs]
        healthy_labels = ["healthy" if r["is_healthy"] else "not_healthy" for r in rs]
        summary[ds] = {
            "per_class": per_class,
            "shortcut_healthy_vs_rest": shortcut_accuracy(values, healthy_labels, SEED),
        }
        log(f"  {ds}: {summary[ds]['shortcut_healthy_vs_rest']}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
