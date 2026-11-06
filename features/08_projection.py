"""Module 08: supervised projection fitted on source crops only (proposal section 5, step 1).

Fits a shrinkage linear discriminant on the TRAINING split of the source datasets (PlantSeg and
PlantVillage) and keeps the first 4, 8 and 16 discriminant directions. Target datasets (rice, fig,
chili, peach) are never used in the fit.

Validation: nearest-class-centroid accuracy on the VALIDATION split of the source datasets, in the
projected space, for each dimension. Dimension choice is reported; the registered choice comes
from this table (proposal 7.2 and 7.11).

Outputs:
  - projection_k{4,8,16}.npz   mean and projection matrix (1536 x k)
  - validation.csv             accuracy per dimension
  - manifest.json              sizes, seed, class counts, timing

CPU module. Reads cached features only. Runs locally.

Usage:
    python 08_projection.py --features results/05_features_dinov2/remote/features_dinov2 --output results/08_projection
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SOURCES = ["plantseg_v6", "plantvillage_color"]
DIMS = [4, 8, 16]
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/4: loading source features and index")
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    X_tr, y_tr, X_va, y_va = [], [], [], []
    for ds in SOURCES:
        a = np.load(feats / f"features_{ds}.npy").astype(np.float32)
        rows = [r for r in index if r["dataset"] == ds]
        for r, vec in zip(rows, a):
            if r["split"] == "train":
                X_tr.append(vec)
                y_tr.append(f"{ds}:{r['class']}")
            elif r["split"] == "val":
                X_va.append(vec)
                y_va.append(f"{ds}:{r['class']}")
    X_tr = np.vstack(X_tr)
    X_va = np.vstack(X_va)
    log(f"  train {X_tr.shape}, validation {X_va.shape}, classes {len(set(y_tr))}")

    log("step 2/4: fitting shrinkage LDA on training split (sources only)")
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda.fit(X_tr, y_tr)
    # Full discriminant directions, ordered by the eigen decomposition of the LDA solver.
    W_full = lda.scalings_[:, : min(lda.scalings_.shape[1], max(DIMS))]
    mean = X_tr.mean(axis=0)  # overall training mean, the centring used by the projection
    log(f"  kept {W_full.shape[1]} discriminant directions (limit {max(DIMS)})")

    log("step 3/4: validation accuracy by dimension")
    classes = sorted(set(y_tr))
    cls_index = {c: i for i, c in enumerate(classes)}
    y_tr_i = np.array([cls_index[c] for c in y_tr])
    rows = []
    for k in DIMS:
        W = W_full[:, :k]
        Ztr = (X_tr - mean) @ W
        centroids = np.vstack([Ztr[y_tr_i == i].mean(axis=0) for i in range(len(classes))])
        Zva = (X_va - mean) @ W
        seen = [y in cls_index for y in y_va]
        d = ((Zva[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        pred = d.argmin(axis=1)
        keep = np.array(seen)
        y_va_i = np.array([cls_index[c] for c in np.array(y_va)[keep]])
        acc = float(np.mean(pred[keep] == y_va_i))
        rows.append({"dimensions": k, "validation_accuracy": round(acc, 4),
                     "validation_rows": int(keep.sum()), "validation_rows_unseen_class_dropped": int((~keep).sum())})
        np.savez(out / f"projection_k{k}.npz", mean=mean.astype(np.float32), W=W.astype(np.float32))
        log(f"  k={k}: validation accuracy {acc:.4f} on {int(keep.sum())} rows")

    log("step 4/4: writing validation table and manifest")
    with open(out / "validation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    manifest = {
        "module": "08_projection",
        "method": "shrinkage LDA (solver eigen, shrinkage auto), fitted on source training split only",
        "sources": SOURCES,
        "train_rows": int(X_tr.shape[0]),
        "validation_rows": int(X_va.shape[0]),
        "classes_train": len(classes),
        "dimensions": DIMS,
        "seed": SEED,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
