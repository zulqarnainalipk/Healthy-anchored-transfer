"""Module 57: spot check of the PlantWild features (and the row-to-file alignment).

Recomputes, locally and independently of the remote run, the DINOv2 ViT-B/14 features (pinned revision, float32 on CPU, same
224 x 224 bicubic resize, same [CLS | mean patch] layout) for a random sample of images of features_index.csv and compares them
with the stored rows: cosine similarity and the largest absolute difference. Also checks that the stored class of each sampled
row equals the folder name of its file and that every file listed in the index exists.

Outputs: results/57_feature_spotcheck/spotcheck.csv and a summary line. CPU module, about 2 minutes for 40 images.
Usage: python 57_feature_spotcheck.py --data-root <data>/plantwild --features <features dir> --sample 40
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL, REV = "facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--output", default="results/57_feature_spotcheck")
    ap.add_argument("--sample", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    root, feats = Path(args.data_root), Path(args.features)
    log("step 1/3: index and existence of every listed file")
    meta = list(csv.DictReader(open(feats / "features_index.csv", encoding="utf-8")))
    missing = [m["path"] for m in meta if not (root / "images" / m["path"]).exists()]
    wrong_class = [m["path"] for m in meta if m["path"].split("/")[0] != m["class"]]
    log(f"  {len(meta)} rows, missing files {len(missing)}, class differs from folder {len(wrong_class)}")
    F = np.load(feats / "features_plantwild.npy").astype(np.float32)
    assert len(F) == len(meta)

    log("step 2/3: loading the encoder")
    proc = AutoImageProcessor.from_pretrained(MODEL, revision=REV)
    model = AutoModel.from_pretrained(MODEL, revision=REV).eval()
    rng = np.random.default_rng(3)
    pick = sorted(rng.choice(len(meta), size=args.sample, replace=False))

    log("step 3/3: recomputing features")
    rows = []
    for k, i in enumerate(pick, 1):
        with Image.open(root / "images" / meta[i]["path"]) as im:
            img = im.convert("RGB").resize((224, 224), Image.BICUBIC)
        px = proc(images=[img], return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
        with torch.no_grad():
            h = model(pixel_values=px).last_hidden_state
        v = torch.cat([h[:, 0, :], h[:, 1:, :].mean(dim=1)], dim=1)[0].numpy()
        s = F[i]
        cos = float(v @ s / (np.linalg.norm(v) * np.linalg.norm(s)))
        rows.append({"row": i, "path": meta[i]["path"], "cosine": round(cos, 6), "max_abs_diff": round(float(np.abs(v - s).max()), 4),
                     "rel_diff": round(float(np.linalg.norm(v - s) / np.linalg.norm(v)), 5)})
        if k % 10 == 0:
            log(f"    {k}/{len(pick)}")
    with open(out / "spotcheck.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {"rows": len(meta), "missing_files": len(missing), "class_differs_from_folder": len(wrong_class), "sampled": len(rows),
            "min_cosine": min(r["cosine"] for r in rows), "max_rel_diff": max(r["rel_diff"] for r in rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(f"summary {summ}")


if __name__ == "__main__":
    main()
