"""Module 05: cached DINOv2 features (proposal section 5, step 1).

One frozen encoder pass over every eligible image. Output is cached so that all later modules run
on features only. Encoder: facebook/dinov2-base (ViT-B/14), revision recorded in the manifest.

Per image:
  - resized to 224 x 224 (full frame, no crop, so no leaf edge is cut off)
  - features = [CLS token | mean of patch tokens], 768 + 768 = 1536 values, stored as float16

Inputs:
  - the dataset folder (read only)
  - feature_index.csv: one row per eligible image, with dataset, path, split, class, is_healthy
    (built from results/02b_splits/splits.csv and results/01_crop_disease_table/labelled_images.csv)

Outputs (in the output folder):
  - features_<dataset>.npy       float16, shape (n_images, 1536), rows in index order
  - features_index.csv           the index with a row number for each feature row
  - manifest.json                encoder id and revision, input sizes, checksums, timing

GPU module when run remotely. Logs step, dataset and batch progress.

Usage (remote):
    python 05_features_dinov2.py --dataset-root <folder> --index <feature_index.csv> --output <folder>
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"   # latest commit of the repository (a README edit of 2024-01-17); the weights are unchanged since 2023-09-04. The original run did not pass a revision.
SIZE = 224
BATCH = 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/4: reading feature index")
    with open(args.index, encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    by_ds = {}
    for i, r in enumerate(index):
        by_ds.setdefault(r["dataset"], []).append(i)
    log(f"  {len(index)} images in {len(by_ds)} datasets")

    log("step 2/4: loading encoder " + MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval().half() if device == "cuda" \
        else AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    log(f"  loaded, config commit: {revision}")

    log("step 3/4: extracting features, dataset by dataset")
    file_info = {}
    for ds, rows in sorted(by_ds.items()):
        log(f"  {ds}: {len(rows)} images")
        feats = np.zeros((len(rows), 1536), dtype=np.float16)
        for start in range(0, len(rows), BATCH):
            batch_ids = rows[start:start + BATCH]
            imgs = [load_rgb(root / index[i]["path"]) for i in batch_ids]
            px = processor(images=imgs, return_tensors="pt", do_resize=False, do_center_crop=False)
            pixel_values = px["pixel_values"].to(device)
            if device == "cuda":
                pixel_values = pixel_values.half()
            with torch.no_grad():
                hidden = model(pixel_values=pixel_values).last_hidden_state
            cls = hidden[:, 0, :]
            mean_patch = hidden[:, 1:, :].mean(dim=1)
            vec = torch.cat([cls, mean_patch], dim=1).float().cpu().numpy().astype(np.float16)
            feats[start:start + len(batch_ids)] = vec
            if (start // BATCH) % 50 == 0:
                log(f"    {ds}: {start + len(batch_ids)}/{len(rows)}")
        path = out / f"features_{ds}.npy"
        np.save(path, feats)
        file_info[ds] = {"file": path.name, "rows": len(rows), "sha256": sha256_of(path)}
        log(f"  saved {path.name} {feats.shape}")

    log("step 4/4: writing index and manifest")
    row_of = {}
    for ds, rows in by_ds.items():
        for k, i in enumerate(rows):
            row_of[i] = (ds, k)
    with open(out / "features_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "row", "path", "split", "class", "is_healthy"])
        for i, r in enumerate(index):
            ds, k = row_of[i]
            w.writerow([ds, k, r["path"], r["split"], r["class"], r["is_healthy"]])
    manifest = {
        "module": "05_features_dinov2",
        "encoder": MODEL_ID,
        "encoder_commit": revision,
        "encoder_revision_requested": MODEL_REVISION,
        "input_size": SIZE,
        "feature_layout": "CLS (768) then mean of patch tokens (768), float16",
        "images": len(index),
        "files": file_info,
        "seconds": round(time.time() - t0, 1),
        "device": device,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
