"""Module 60: background-only control for the real-pair test (host versus acquisition effects).

For every PlantWild image (same rows and order as the DINOv2 features) we describe only the outer frame of the picture, the
outer 12.5% on each side, which is mostly background: mean and standard deviation of red, green, blue, hue, saturation and
value (12 values) and an 8-bin hue histogram (8 values), 20 values in all. The saved matrix goes into the real-pair test
(module 53) in place of the DINOv2 features. If the gain of HATC over mean-centring came from photographic style and not from
leaves, it should also appear with these features. A second output gives, for each crop, the cross-validated AUROC of a
logistic classifier that separates healthy from diseased images from the frame alone, next to the same number for the DINOv2 features.

Usage:
    python 60_background_control.py --data-root <data>/plantwild --features results/09_features_plantwild/remote/features_plantwild --output results/60_background_control
Then:
    python features/53_real_pair_test.py --dataset plantwild --features results/60_background_control --features-name features_plantwild_border.npy \
        --projection pca --tag border --output results/53_real_pair_test
CPU module, about 10 minutes.
"""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

SIZE, FRAME = 128, 16        # images are reduced to 128 x 128; the frame is the outer 16 pixels (12.5%) on each side


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describe(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
        h = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32) / 255.0
    ring = np.ones((SIZE, SIZE), dtype=bool)
    ring[FRAME:-FRAME, FRAME:-FRAME] = False
    px = np.concatenate([a[ring], h[ring]], axis=1)                    # rows: ring pixels; columns: R G B H S V
    hist = np.histogram(h[ring][:, 0], bins=8, range=(0, 1))[0] / ring.sum()
    return np.concatenate([px.mean(axis=0), px.std(axis=0), hist]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features", required=True, help="folder with features_plantwild.npy and features_index.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fdir = Path(args.features)
    log("step 1/4: reading the feature index")
    meta = list(csv.DictReader(open(fdir / "features_index.csv", encoding="utf-8")))
    log(f"  {len(meta)} images")
    log("step 2/4: describing the image frames")
    root = Path(args.data_root) / "images"
    F = np.zeros((len(meta), 20), dtype=np.float32)
    bad = []
    for i, m in enumerate(meta):
        try:
            F[i] = describe(root / m["path"])
        except Exception:
            bad.append(m["path"])
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            log(f"    {i + 1}/{len(meta)} images, finish about {time.strftime('%H:%M', time.localtime(time.time() + (len(meta) - i - 1) / rate))}")
    np.save(out / "features_plantwild_border.npy", F)
    shutil.copy(fdir / "features_index.csv", out / "features_index.csv")
    log(f"  unreadable {len(bad)}")

    log("step 3/4: healthy versus diseased from the frame alone, per crop (5-fold cross-validated AUROC)")
    X = np.load(fdir / "features_plantwild.npy").astype(np.float32)
    crop = np.array([m["crop"] for m in meta])
    isd = np.array([m["role"] == "disease" for m in meta]).astype(int)
    rows = []
    for c in sorted(set(crop) - {""}):
        k = crop == c
        if isd[k].sum() < 40 or (1 - isd[k]).sum() < 40:
            continue
        res = {"crop": c, "healthy": int((1 - isd[k]).sum()), "disease": int(isd[k].sum())}
        for name, M in (("frame", F), ("dinov2", X)):
            Z = M[k]
            Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-6)
            p = cross_val_predict(LogisticRegression(max_iter=500, C=0.1, class_weight="balanced"), Z, isd[k], cv=5, method="predict_proba")[:, 1]
            res[f"auroc_{name}"] = round(float(roc_auc_score(isd[k], p)), 3)
        rows.append(res)
        log(f"    {c}: frame {res['auroc_frame']}, dinov2 {res['auroc_dinov2']}")
    with open(out / "frame_vs_dinov2_auroc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log("step 4/4: manifest")
    (out / "manifest.json").write_text(json.dumps({
        "module": "60_background_control", "images": len(meta), "unreadable": bad, "frame_fraction": FRAME / SIZE, "values": 20,
        "mean_auroc_frame": round(float(np.mean([r["auroc_frame"] for r in rows])), 3),
        "mean_auroc_dinov2": round(float(np.mean([r["auroc_dinov2"] for r in rows])), 3), "crops": len(rows),
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
