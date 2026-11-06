"""Module 34: chili binary target and peach damage probe, with the background control (proposal 4, 7.5).

Chili (healthy against yellow-leaf, two classes) has a background shortcut: the corner brightness alone predicts the
class (module 03). This module reports, for chili:
  - brightness-only baseline: a single corner-brightness threshold, accuracy on the test split
  - feature probe: logistic regression on the 1,536 DINOv2 features, accuracy and AUROC on the test split
  - BACKGROUND-MATCHED control: healthy and yellow-leaf images are paired by nearest corner brightness (within
    MATCH_TOL), so brightness no longer separates the classes; the brightness-only baseline falls to chance, and
    the feature probe is trained and tested again on the matched subset. Accuracy above chance there is leaf signal,
    not background.
Peach (healthy against damage) has near-uniform black backgrounds, so there is no brightness shortcut; the probe is
reported by 5-fold cross-validation (peach has no splits, it is a probe only), with the brightness-only baseline.

Inputs: results/05 features and index, results/03 background_per_image.csv, results/02b splits (for chili).
Outputs (results/34_chili_peach_probes/): probe_results.csv, manifest.json

CPU module, runs locally.

Usage:
    python 34_chili_peach_probes.py --features ... --background results/03_background_check/background_per_image.csv
        --output results/34_chili_peach_probes
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

MATCH_TOL = 5.0
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def threshold_accuracy(train_x, train_y, test_x, test_y):
    """Brightness-only baseline: best single threshold on the training data, accuracy on test."""
    order = np.argsort(train_x)
    x, y = train_x[order], train_y[order]
    best_acc, best_t, best_sign = -1, 0.0, 1
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            continue
        t = (x[i] + x[i - 1]) / 2
        pred = (x > t).astype(int)
        for sign in (1, -1):
            acc = np.mean((pred if sign == 1 else 1 - pred) == y)
            if acc > best_acc:
                best_acc, best_t, best_sign = acc, t, sign
    pred = (test_x > best_t).astype(int)
    pred = pred if best_sign == 1 else 1 - pred
    return float(np.mean(pred == test_y))


def fit_probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500, C=0.1).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return float(np.mean((p > 0.5) == yte)), float(roc_auc_score(yte, p))


def matched_subset(bright, y, rng, tol=MATCH_TOL):
    """Pair each minority-class image with the nearest-brightness unused image of the other class."""
    idx0 = list(np.where(y == 0)[0])
    idx1 = list(np.where(y == 1)[0])
    small, large = (idx0, idx1) if len(idx0) <= len(idx1) else (idx1, idx0)
    large_sorted = sorted(large, key=lambda i: bright[i])
    large_b = np.array([bright[i] for i in large_sorted])
    used = np.zeros(len(large_sorted), dtype=bool)
    keep = []
    order = list(small)
    rng.shuffle(order)
    for i in order:
        pos = np.searchsorted(large_b, bright[i])
        best, best_d = None, tol + 1e-9
        for j in range(max(0, pos - 25), min(len(large_b), pos + 25)):
            if used[j]:
                continue
            d = abs(large_b[j] - bright[i])
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            used[best] = True
            keep += [i, large_sorted[best]]
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    log("step 1/3: loading features, index and background brightness")
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    bright = {}
    with open(args.background, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bright[r["path"]] = float(r["brightness"])
    results = []

    # ---- chili
    log("step 2/3: chili")
    rows = [r for r in index if r["dataset"] == "chili_yellow_leaf"]
    X = np.load(feats / "features_chili_yellow_leaf.npy").astype(np.float32)
    y = np.array([1 if r["is_healthy"] == "1" else 0 for r in rows])   # 1 = healthy
    b = np.array([bright[r["path"]] for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, te = split == "train", split == "test"
    acc_b = threshold_accuracy(b[tr], y[tr], b[te], y[te])
    acc_f, auc_f = fit_probe(X[tr], y[tr], X[te], y[te])
    results.append({"dataset": "chili", "setting": "all test images", "images_test": int(te.sum()),
                    "brightness_only_accuracy": round(acc_b, 4), "feature_probe_accuracy": round(acc_f, 4),
                    "feature_probe_auroc": round(auc_f, 4), "chance": 0.5})
    # background-matched subset, matched within each split so train and test stay separate
    keep_tr = np.where(tr)[0][matched_subset(b[tr], y[tr], rng)]
    keep_te = np.where(te)[0][matched_subset(b[te], y[te], rng)]
    acc_bm = threshold_accuracy(b[keep_tr], y[keep_tr], b[keep_te], y[keep_te])
    acc_fm, auc_fm = fit_probe(X[keep_tr], y[keep_tr], X[keep_te], y[keep_te])
    results.append({"dataset": "chili", "setting": f"background-matched (within {MATCH_TOL} brightness)",
                    "images_test": int(len(keep_te)), "brightness_only_accuracy": round(acc_bm, 4),
                    "feature_probe_accuracy": round(acc_fm, 4), "feature_probe_auroc": round(auc_fm, 4),
                    "chance": 0.5})
    log(f"  chili all: brightness {acc_b:.3f}, probe {acc_f:.3f}; matched ({len(keep_te)} test): "
        f"brightness {acc_bm:.3f}, probe {acc_fm:.3f}")

    # ---- peach
    log("step 3/3: peach")
    rows = [r for r in index if r["dataset"] == "peach_leaf_damage"]
    X = np.load(feats / "features_peach_leaf_damage.npy").astype(np.float32)
    y = np.array([1 if r["is_healthy"] == "1" else 0 for r in rows])
    b = np.array([bright[r["path"]] for r in rows])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    accs_f, accs_b, aucs = [], [], []
    for tr_i, te_i in skf.split(X, y):
        a, u = fit_probe(X[tr_i], y[tr_i], X[te_i], y[te_i])
        accs_f.append(a)
        aucs.append(u)
        accs_b.append(threshold_accuracy(b[tr_i], y[tr_i], b[te_i], y[te_i]))
    maj = max(y.mean(), 1 - y.mean())
    results.append({"dataset": "peach", "setting": "5-fold cross-validation", "images_test": int(len(y)),
                    "brightness_only_accuracy": round(float(np.mean(accs_b)), 4),
                    "feature_probe_accuracy": round(float(np.mean(accs_f)), 4),
                    "feature_probe_auroc": round(float(np.mean(aucs)), 4), "chance": round(float(maj), 4)})
    log(f"  peach: brightness {np.mean(accs_b):.3f}, probe {np.mean(accs_f):.3f}, majority {maj:.3f}")

    with open(out / "probe_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "manifest.json").write_text(json.dumps({
        "module": "34_chili_peach_probes", "match_tolerance": MATCH_TOL, "seed": SEED,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
