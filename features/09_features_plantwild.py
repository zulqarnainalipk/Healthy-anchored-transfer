"""Module 09 and 10: frozen image features of PlantWild v1 (DINOv2 ViT-B/14, or BioCLIP ViT-B/16 with --encoder bioclip).

Reads the PlantWild images (images/<class>/<file>, classes.txt, trainval.txt), drops the images listed in the duplicate-rule
exclusion file (results/plantwild_audit/dedupe_exclude.csv, module 08b), and writes one feature vector per remaining image:
  dinov2   resize to 224 x 224 (full frame, bicubic), ImageNet normalisation, [CLS | mean patch token], 1,536 values, float16,
           model facebook/dinov2-base pinned to revision f9e44c814b77203eaa57a6bdbbd535f21ede1415
  bioclip  resize to 224 x 224, CLIP normalisation, image embedding, 512 values, float16 (open_clip, hf-hub:imageomics/bioclip)
Outputs in --output: features_plantwild.npy (dinov2) or features_plantwild_bioclip.npy, features_index.csv (row, path, class, role,
crop, disease, split_mode of the authors' trainval.txt), manifest.json, and a partial checkpoint every 2,000 images.
Runs on a CPU (about 1 to 2 hours for 17,328 images) or a GPU. Every step and the progress with an estimated finishing time are logged.

Usage:
    python 09_features_plantwild.py --data-root <data>/plantwild --exclude results/plantwild_audit/dedupe_exclude.csv --output <dir>
    python 09_features_plantwild.py --data-root ... --exclude ... --output <dir> --encoder bioclip
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DINO, REVISION = "facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
BIOCLIP = "hf-hub:imageomics/bioclip"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
SIZE, BATCH = 224, 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Images(torch.utils.data.Dataset):
    def __init__(self, root, items):
        self.root, self.items = root, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        try:
            with Image.open(self.root / self.items[i]["path"]) as im:
                a = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
            return torch.from_numpy(a.copy()), i
        except Exception:
            return torch.zeros((SIZE, SIZE, 3), dtype=torch.uint8), -1 - i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--exclude", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--encoder", choices=["dinov2", "bioclip"], default="dinov2")
    args = ap.parse_args()
    t0 = time.time()
    base, out = Path(args.data_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    root = base / "images"

    log("step 1/5: image list")
    exclude = set(Path(args.exclude).read_text(encoding="utf-8").splitlines()[1:])
    classes = [ln.strip().split(" ", 1)[1] for ln in (base / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    mode = {}
    for ln in (base / "trainval.txt").read_text(encoding="utf-8").splitlines():
        if "=" in ln:
            p, _, m = ln.strip().rsplit("=", 2)
            mode[p] = m
    healthy = {c for c in classes if c.endswith(" leaf")}
    crops = sorted({c[:-5] for c in healthy}, key=len, reverse=True)
    items, dropped = [], 0
    for c in classes:
        role = "healthy" if c in healthy else "disease"
        crop = c[:-5] if role == "healthy" else next((x for x in crops if c.startswith(x + " ")), "")
        disease = "" if role == "healthy" else (c[len(crop) + 1:] if crop else c)
        for q in sorted((root / c).iterdir()):
            rel = f"{c}/{q.name}"
            if rel in exclude:
                dropped += 1
                continue
            items.append({"path": rel, "class": c, "role": role, "crop": crop, "disease": disease, "split_mode": mode.get(rel, "")})
    log(f"  {len(items)} images kept, {dropped} dropped by the duplicate rule")

    log("step 2/5: loading the encoder " + args.encoder)
    torch.set_num_threads(os.cpu_count() or 4)
    if args.encoder == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(DINO, revision=REVISION)
        model = AutoModel.from_pretrained(DINO, revision=REVISION).eval()
        dim, fname = 1536, "features_plantwild.npy"
    else:
        import open_clip
        model = open_clip.create_model_and_transforms(BIOCLIP)[0].eval()
        mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
        dim, fname = 512, "features_plantwild_bioclip.npy"

    log("step 3/5: extracting features")
    feats = np.zeros((len(items), dim), dtype=np.float16)
    failed, done, tl = [], 0, time.time()
    for batch, idx in torch.utils.data.DataLoader(Images(root, items), batch_size=BATCH, num_workers=2):
        with torch.no_grad():
            if args.encoder == "dinov2":
                px = proc(images=list(batch.numpy()), return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"]
                h = model(pixel_values=px).last_hidden_state
                v = torch.cat([h[:, 0, :], h[:, 1:, :].mean(dim=1)], dim=1)
            else:
                v = model.encode_image((batch.permute(0, 3, 1, 2).float() / 255.0 - mean) / std)
        v = v.float().numpy().astype(np.float16)
        for k, i in enumerate(idx.tolist()):
            if i < 0:
                failed.append(items[-1 - i]["path"])
            else:
                feats[i] = v[k]
        done += len(idx)
        if (done // BATCH) % 20 == 0:
            rate = done / (time.time() - tl)
            log(f"    {done}/{len(items)} images, {rate:.1f} img/s, finish about {time.strftime('%H:%M', time.localtime(time.time() + (len(items) - done) / max(rate, 1e-6)))}")
        if done % 2000 < BATCH:
            np.save(out / "partial_features.npy", feats[:done])

    log("step 4/5: writing features and index")
    np.save(out / fname, feats)
    with open(out / "features_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "path", "class", "role", "crop", "disease", "split_mode"])
        w.writeheader()
        for i, it in enumerate(items):
            w.writerow({"row": i, **it})
    if (out / "partial_features.npy").exists():
        (out / "partial_features.npy").unlink()
    log("step 5/5: manifest")
    (out / "manifest.json").write_text(json.dumps({
        "module": "09_features_plantwild", "encoder": args.encoder, "model": DINO if args.encoder == "dinov2" else BIOCLIP,
        "revision": REVISION if args.encoder == "dinov2" else "latest at run time", "images": len(items), "dropped_by_duplicate_rule": dropped,
        "failed_to_read": failed, "dtype": "float16", "dimensions": dim, "input_size": SIZE, "seconds": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s; failed {len(failed)}")


if __name__ == "__main__":
    main()
