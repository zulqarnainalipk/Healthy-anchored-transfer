"""Module 13: push source disease models through the affine host maps (proposal section 5, step 3).

For each source crop A with a host map to a target crop B (module 11), and each disease d of A:
  1. fit a shrunk-covariance Gaussian mixture (2 components) to the disease's training features in the
     16-dimensional projection (module 08); diseases with fewer than MIN_TRAIN images are skipped
  2. push each component through the affine map x -> m_B + T (x - m_A). The pushed component has mean
     m_B + T (mu_k - m_A) and covariance T Sigma_k T'.

The pushed mixture is the predicted distribution of disease d on the target crop B. Module 14 scores
target leaves under these mixtures.

Outputs:
  - pushed_models.npz   per (A, B, d): weights, means, covariances (stored as flat arrays with keys)
  - pushed_models.csv   one row per (A, B, d): counts, component weights, skip reasons
  - manifest.json

CPU module, runs locally.

Usage:
    python 13_push_forward.py --features results/05_features_dinov2/remote/features_dinov2
        --labelled results/01_crop_disease_table/labelled_images.csv
        --projection results/08_projection/projection_k16.npz
        --maps results/11_host_maps_affine --output results/13_push_forward
"""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

SOURCE_DATASETS = ["plantvillage_color", "plantseg_v6"]
MIN_TRAIN = 30
SHRINK = 0.1
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrunk_cov(Z):
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    return (1 - SHRINK) * C + SHRINK * np.trace(C) / d * np.eye(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/4: loading projection, host maps and features")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    M = np.load(Path(args.maps) / "host_maps.npz")
    quality = {}
    with open(Path(args.maps) / "host_map_quality.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            quality[(r["source_crop"], r["target"])] = float(r["frechet_error"])
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in SOURCE_DATASETS and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    # Projected features for each source dataset, and the position of each index row in its file.
    Zs = {ds: (np.load(feats / f"features_{ds}.npy").astype(np.float64) - mean) @ W
          for ds in SOURCE_DATASETS}
    rows_of = {ds: [i for i, r in enumerate(index) if r["dataset"] == ds] for ds in SOURCE_DATASETS}
    pos = {}
    for ds in SOURCE_DATASETS:
        for k, i in enumerate(rows_of[ds]):
            pos[i] = (ds, k)

    # Map pairs available from module 11 (keys like "apple__fig_T").
    pairs = sorted({k[:-2] for k in M.files if k.endswith("_T")})
    log(f"  {len(pairs)} host maps: {pairs}")

    log("step 2/4: fitting disease mixtures on source training data")
    disease_models = {}   # (source crop, disease) -> (weights, means, covs, n)
    rows = []
    by_key = {}
    for i, r in enumerate(index):
        if r["dataset"] not in SOURCE_DATASETS or r["split"] != "train" or r["is_healthy"] == "1":
            continue
        crop = crop_of.get(r["path"])
        if crop is None:
            continue
        by_key.setdefault((crop, r["class"]), []).append(pos[i])
    for (crop, disease), locs in sorted(by_key.items()):
        if not any(p.startswith(crop + "__") for p in pairs):
            continue  # no host map for this crop: it has no healthy reference (not a HATC source)
        if len(locs) < MIN_TRAIN:
            rows.append({"source_crop": crop, "disease": disease, "train_images": len(locs),
                         "status": f"skipped: fewer than {MIN_TRAIN} training images"})
            continue
        X = np.vstack([Zs[ds][k][None, :] for ds, k in locs])
        ncomp = 2 if len(locs) >= 60 else 1
        gmm = GaussianMixture(n_components=ncomp, covariance_type="full", random_state=SEED,
                              reg_covar=1e-4 * np.trace(shrunk_cov(X)) / X.shape[1]).fit(X)
        covs = np.array([(1 - SHRINK) * c + SHRINK * np.trace(c) / c.shape[0] * np.eye(c.shape[0])
                         for c in gmm.covariances_])
        disease_models[(crop, disease)] = (gmm.weights_, gmm.means_, covs, len(locs))
        rows.append({"source_crop": crop, "disease": disease, "train_images": len(locs),
                     "status": f"fitted: {ncomp} components"})
    log(f"  fitted {len(disease_models)} source disease models")

    log("step 3/4: pushing models through each host map")
    pushed = {}
    for pair in pairs:
        scrop, tkey = pair.split("__")
        tcrop = tkey.replace("_T", "")
        T = M[f"{pair}_T"]
        mA, mB = M[f"{pair}_m"][0], M[f"{pair}_m"][1]
        target = "fig" if tcrop == "fig" else "rice"
        n_push = 0
        for (crop, disease), (w, mu, C, n) in disease_models.items():
            if crop != scrop:
                continue
            mu_p = mB + (mu - mA) @ T.T
            C_p = np.array([T @ c @ T.T for c in C])
            key = f"{scrop}__{target}__{disease}"
            pushed[key] = {"weights": w, "means": mu_p, "covs": C_p, "train_images": n}
            n_push += 1
        log(f"  {scrop} -> {target}: {n_push} disease models pushed")

    log("step 4/4: writing outputs")
    with open(out / "pushed_models.pkl", "wb") as f:
        pickle.dump(pushed, f)
    with open(out / "pushed_models.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_crop", "disease", "train_images", "status"])
        for r in rows:
            w.writerow([r["source_crop"], r["disease"], r["train_images"], r["status"]])
    manifest = {"module": "13_push_forward", "projection_dimensions": int(W.shape[1]),
                "maps_used": pairs, "pushed_models": len(pushed),
                "source_disease_models": len(disease_models), "min_train": MIN_TRAIN,
                "shrinkage": SHRINK, "seed": SEED, "seconds": round(time.time() - t0, 1),
                "note": "pickle written by this run; the map quality is in module 11"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()
