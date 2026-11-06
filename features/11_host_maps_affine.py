"""Module 11: affine host maps from source crops to target crops (proposal section 5, step 2).

For each source crop A (PlantVillage crops with enough healthy images) and each target crop B
(rice, fig Original), fit the affine Monge map between the Gaussians of their healthy features in the
16-dimensional projection (module 08, main setting). The map sends A's healthy Gaussian to B's.

Fitting uses training splits only. Quality is the Frechet distance between the transported A Gaussian
and B's held-out (validation split) healthy Gaussian, reported beside a baseline of the B-to-B
distance between two halves of B's training split, so the map's error can be read against noise.

Outputs:
  - host_maps.npz        one matrix T and mean for each (A, B), in one archive (keys "A__B_T", "A__B_m")
  - host_map_quality.csv per pair: Frechet error of the map, baseline, counts
  - manifest.json

The Brenier map (for B with 200 or more healthy images) is a separate module and is not fitted here.

CPU module, runs locally.

Usage:
    python 11_host_maps_affine.py --features results/05_features_dinov2/remote/features_dinov2
        --labelled results/01_crop_disease_table/labelled_images.csv
        --projection results/08_projection/projection_k16.npz --output results/11_host_maps_affine
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

SOURCE_DATASET = "plantvillage_color"
TARGETS = {"rice_leaf_disease": "rice", "fig_sartak_sfld": "fig"}
MIN_TRAIN = 200
MIN_VAL = 50
SHRINK = 0.1
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrunk_cov(Z):
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    return (1 - SHRINK) * C + SHRINK * np.trace(C) / d * np.eye(d)


def frechet(m1, C1, m2, C2):
    s1 = sqrtm(C1).real
    cross = sqrtm(s1 @ C2 @ s1).real
    return float(np.sum((m1 - m2) ** 2) + np.trace(C1 + C2 - 2 * cross))


def affine_map(C1, C2):
    s1 = sqrtm(C1).real
    s1_inv = np.linalg.inv(s1)
    mid = sqrtm(s1 @ C2 @ s1).real
    return s1_inv @ mid @ s1_inv


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

    log("step 1/4: loading projection, features and healthy rows by crop")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["role"] == "image":
                crop_of[r["path"]] = (r["crop"].strip().lower() if r["dataset"] != "fig_sartak_sfld"
                                      else "fig")
    groups = {}  # (dataset, crop) -> {"train": [...], "val": [...]} of feature rows
    feat_cache = {}
    for ds in [SOURCE_DATASET] + list(TARGETS):
        feat_cache[ds] = (np.load(feats / f"features_{ds}.npy").astype(np.float64) - mean) @ W
    for i, r in enumerate(index):
        if r["is_healthy"] != "1" or r["split"] not in ("train", "val"):
            continue
        ds = r["dataset"]
        if ds not in feat_cache:
            continue
        crop = crop_of.get(r["path"], "fig" if ds == "fig_sartak_sfld" else None)
        if crop is None:
            continue
        # row index within this dataset's feature file: rows of the index for this dataset, in order
        groups.setdefault((ds, crop), {"train": [], "val": []})
    # Rebuild the per-dataset row numbers in index order (matches the feature file order).
    row_no = {}
    counter = {}
    for i, r in enumerate(index):
        ds = r["dataset"]
        row_no[i] = counter.get(ds, 0)
        counter[ds] = counter.get(ds, 0) + 1
    for i, r in enumerate(index):
        ds = r["dataset"]
        if r["is_healthy"] != "1" or r["split"] not in ("train", "val") or ds not in feat_cache:
            continue
        crop = crop_of.get(r["path"], "fig" if ds == "fig_sartak_sfld" else None)
        if crop is None:
            continue
        groups[(ds, crop)][r["split"]].append(row_no[i])

    sources = sorted((k, v) for k, v in groups.items() if k[0] == SOURCE_DATASET
                     and len(v["train"]) >= MIN_TRAIN and len(v["val"]) >= MIN_VAL)
    targets = sorted((k, v) for k, v in groups.items() if k[0] in TARGETS
                     and len(v["train"]) >= MIN_TRAIN and len(v["val"]) >= MIN_VAL)
    log(f"  sources: {[k[1] for k, _ in sources]}")
    log(f"  targets: {[k[1] for k, _ in targets]}")

    log("step 2/4: fitting affine maps and checking quality")
    maps, rows = {}, []
    rng = np.random.default_rng(SEED)
    for (tds, tcrop), tv in targets:
        Zt_all = feat_cache[tds][tv["train"]]
        Zt_val = feat_cache[tds][tv["val"]]
        Ct_val = shrunk_cov(Zt_val)
        mt_val = Zt_val.mean(axis=0)
        # Baseline: distance between two random halves of B's training healthy images.
        perm = rng.permutation(len(Zt_all))
        half1, half2 = Zt_all[perm[: len(perm) // 2]], Zt_all[perm[len(perm) // 2:]]
        baseline = frechet(half1.mean(axis=0), shrunk_cov(half1), half2.mean(axis=0), shrunk_cov(half2))
        for (sds, scrop), sv in sources:
            Za = feat_cache[sds][sv["train"]]
            Zb = Zt_all
            mA, CA = Za.mean(axis=0), shrunk_cov(Za)
            mB, CB = Zb.mean(axis=0), shrunk_cov(Zb)
            T = affine_map(CA, CB)
            # Affine Monge map x -> mB + T (x - mA) sends A's Gaussian to N(mB, T CA T').
            # The map is fitted on training images only. Quality is measured on HELD-OUT source
            # images: map A's validation healthy features and compare with B's held-out Gaussian.
            Za_val = feat_cache[sds][sv["val"]]
            Zmapped = mB + (Za_val - mA) @ T.T
            err = frechet(Zmapped.mean(axis=0), shrunk_cov(Zmapped), mt_val, Ct_val)
            key = f"{scrop}__{tcrop}"
            maps[f"{key}_T"] = T
            maps[f"{key}_m"] = np.vstack([mA, mB])
            rows.append({"source_crop": scrop, "target": tcrop, "frechet_error": round(err, 4),
                         "target_baseline_half_split": round(baseline, 4),
                         "source_train": len(sv["train"]), "target_train": len(tv["train"]),
                         "target_validation": len(tv["val"])})
        log(f"  target {tcrop}: {len(sources)} maps, baseline {baseline:.4f}")

    log("step 3/4: writing maps and quality table")
    np.savez(out / "host_maps.npz", **maps)
    with open(out / "host_map_quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log("step 4/4: manifest")
    manifest = {"module": "11_host_maps_affine", "projection_dimensions": int(W.shape[1]),
                "maps": len(rows), "sources": [k[1] for k, _ in sources],
                "targets": [k[1] for k, _ in targets], "min_train": MIN_TRAIN, "min_validation": MIN_VAL,
                "shrinkage": SHRINK, "seed": SEED, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()
