"""Module 10: Gaussianity check of healthy features per source crop (proposal section 5, step 2).

For each source crop with enough healthy images, the healthy features (16-dimensional projection,
module 08) are compared under two models, fitted on the training split and scored on the validation
split:
  - a single Gaussian (shrunk covariance)
  - a two-component Gaussian mixture
The gain is the mean held-out log-likelihood per image of the mixture minus the single Gaussian.
A gain near zero means the single Gaussian is adequate. A large gain means the healthy distribution
has structure the affine map does not capture.

The result is reported only. The registered rules do not change with it (proposal section 5).

Note: no plant IDs exist in PlantVillage, so the split is the dataset's own train and validation split,
not a plant-level split. This limit is stated in the report.

CPU module, runs locally.

Usage:
    python 10_gaussianity.py --features results/05_features_dinov2/remote/features_dinov2
        --labelled results/01_crop_disease_table/labelled_images.csv
        --projection results/08_projection/projection_k16.npz --output results/10_gaussianity
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

DATASET = "plantvillage_color"
MIN_TRAIN = 200
MIN_VAL = 50
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading projection, features and healthy rows by crop")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = [r for r in csv.DictReader(f) if r["dataset"] == DATASET]
    feat = np.load(feats / f"features_{DATASET}.npy").astype(np.float64)
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == DATASET and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    Z = (feat - mean) @ W
    by_crop = {}
    for i, r in enumerate(index):
        if r["is_healthy"] != "1":
            continue
        c = crop_of.get(r["path"])
        if c is None or r["split"] not in ("train", "val"):
            continue
        by_crop.setdefault(c, {"train": [], "val": []})[r["split"]].append(i)

    log("step 2/3: fitting single Gaussian and two-component mixture per crop")
    rows = []
    for crop, v in sorted(by_crop.items()):
        if len(v["train"]) < MIN_TRAIN or len(v["val"]) < MIN_VAL:
            continue
        Xtr, Xva = Z[v["train"]], Z[v["val"]]
        m = Xtr.mean(axis=0)
        C = np.cov(Xtr, rowvar=False)
        C = 0.9 * C + 0.1 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        ll_single = multivariate_normal(mean=m, cov=C).logpdf(Xva)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=SEED,
                              reg_covar=1e-4 * np.trace(C) / C.shape[0]).fit(Xtr)
        ll_mix = gmm.score_samples(Xva)
        gain = float(np.mean(ll_mix) - np.mean(ll_single))
        rows.append({"crop": crop, "train": len(v["train"]), "validation": len(v["val"]),
                     "loglik_single_per_image": round(float(np.mean(ll_single)), 3),
                     "loglik_mixture_per_image": round(float(np.mean(ll_mix)), 3),
                     "gain_mixture_minus_single": round(gain, 3)})
        log(f"  {crop}: gain {gain:.3f} nats per image")

    log("step 3/3: writing outputs")
    if not rows:
        raise SystemExit("no crop has enough healthy images")
    with open(out / "gaussianity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    gains = [r["gain_mixture_minus_single"] for r in rows]
    summary = {"crops": len(rows), "median_gain": round(float(np.median(gains)), 3),
               "max_gain": round(float(np.max(gains)), 3), "min_gain": round(float(np.min(gains)), 3),
               "seed": SEED, "min_train": MIN_TRAIN, "min_validation": MIN_VAL,
               "limit": "no plant IDs; dataset's own train and validation split used",
               "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
