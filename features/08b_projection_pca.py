"""Module 08b: unsupervised PCA projection on source training data (ablation, proposal section 8).

The proposal asks what the supervised projection (module 08) hides, by comparing it with a healthy-only or
unsupervised alternative. This module fits PCA on the training split of the source datasets (all images, no labels)
and keeps the first 16 components, in the same file format as module 08 (mean and projection matrix W), so the
benchmark modules can use it through --projection.

Output: results/08b_projection_pca/projection_k16.npz, manifest.json

CPU module, runs locally.

Usage:
    python 08b_projection_pca.py --features results/05_features_dinov2/remote/features_dinov2 --output results/08b_projection_pca
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

SOURCES = ["plantvillage_color", "plantseg_v6"]
K = 16


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
    log("step 1/4: reading the feature index")
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    log(f"  {len(index)} index rows")
    log("step 2/4: loading training-split features of the source datasets")
    X = []
    for ds in SOURCES:
        a = np.load(feats / f"features_{ds}.npy").astype(np.float32)
        rows = [r for r in index if r["dataset"] == ds]
        X.append(a[[k for k, r in enumerate(rows) if r["split"] == "train"]])
        log(f"  {ds}: {len(X[-1])} training rows")
    X = np.vstack(X)
    log(f"step 3/4: PCA by SVD on {X.shape[0]} x {X.shape[1]}")
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    W = Vt[:K].T
    log("step 4/4: writing projection and manifest")
    # scale components to unit variance so distances are comparable with the supervised projection
    W = W / (S[:K] / np.sqrt(len(X) - 1))
    np.savez(out / "projection_k16.npz", mean=mean.astype(np.float32), W=W.astype(np.float32))
    explained = float((S[:K] ** 2).sum() / (S ** 2).sum())
    (out / "manifest.json").write_text(json.dumps({
        "module": "08b_projection_pca", "train_rows": int(len(X)), "components": K,
        "explained_variance_share": round(explained, 4), "seconds": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    log(f"done: {len(X)} rows, {K} components, explained variance {explained:.3f}")


if __name__ == "__main__":
    main()
