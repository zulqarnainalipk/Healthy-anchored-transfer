"""Module 06: cached BioCLIP features (second encoder, sensitivity check; proposal 7.7).

One frozen pass of BioCLIP (ViT-B/16, imageomics/bioclip, trained on biological images) over every eligible image,
same index and same 224 x 224 full-frame resize as module 05. Features = the 512-d image embedding (float16).

Inputs:  --dataset-root  folder that holds the dataset folders (read only)
         --index         feature_index.csv (module 05a): dataset, path, split, class, is_healthy
Outputs (in --output): features_<dataset>.npy, features_index.csv, manifest.json (checksums, versions).

GPU module, run once in a remote GPU environment. Logs step, dataset and batch progress with time stamps.
Note: the run that produced results/06_features_bioclip/ used the same logic with the input and output folders
fixed in the script; here they are command-line arguments. Requires the package open_clip_torch.

Usage (remote):
    python 06_features_bioclip.py --dataset-root <folder> --index <feature_index.csv> --output <folder>
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

MODEL_ID = "hf-hub:imageomics/bioclip"
# Repository state at the time of the run (open_clip loads the latest commit and cannot pin one through this interface): latest commit
# ce901ab3c6a913f9e9ef94ce6d27761069f4f01c (2026-02-26, a device-placement fix); the weights were last changed before 2025.
SIZE, BATCH = 224, 64
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    log("step 0/4: importing open_clip and torch")
    import open_clip
    import torch
    from PIL import Image

    root, out = Path(args.dataset_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"inputs: {root} | {args.index}")

    log("step 1/4: reading feature index")
    with open(args.index, encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    by_ds = {}
    for i, r in enumerate(index):
        by_ds.setdefault(r["dataset"], []).append(i)
    log(f"  {len(index)} images in {len(by_ds)} datasets")

    log("step 2/4: loading encoder " + MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    log(f"  device {device}")
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)

    def load(path):
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)

    log("step 3/4: extracting features, dataset by dataset")
    file_info, dim = {}, None
    for ds, rows in sorted(by_ds.items()):
        log(f"  {ds}: {len(rows)} images")
        feats = None
        for start in range(0, len(rows), BATCH):
            ids = rows[start:start + BATCH]
            x = (torch.stack([load(root / index[i]["path"]) for i in ids]) - mean) / std
            x = x.to(device)
            if device == "cuda":
                x = x.half()
            with torch.no_grad():
                v = model.encode_image(x).float().cpu().numpy().astype(np.float16)
            if feats is None:
                dim = v.shape[1]
                feats = np.zeros((len(rows), dim), dtype=np.float16)
            feats[start:start + len(ids)] = v
            if (start // BATCH) % 50 == 0:
                log(f"    {ds}: {start + len(ids)}/{len(rows)}")
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
    manifest = {"module": "06_features_bioclip", "encoder": MODEL_ID, "input_size": SIZE, "dimensions": dim,
                "images": len(index), "files": file_info, "seconds": round(time.time() - t0, 1), "device": device,
                "open_clip_version": open_clip.__version__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done in {manifest['seconds']} s")


if __name__ == "__main__":
    main()
