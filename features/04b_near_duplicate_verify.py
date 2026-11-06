"""Module 04b: verification of the closest target-source pairs (proposal 7.9).

Takes the top candidate pairs from module 04 (nearest.csv), and checks each with a pixel-level
comparison that does not depend on the encoder:
  - both images at 64 x 64 grayscale, normalised to zero mean and unit variance
  - correlation of the target with the source under 8 flips and rotations (90-degree steps)
A pair is a copy if the correlation is at least 0.9.

Outputs:
  - verify_pairs.csv     every candidate pair with its pixel correlation and copy flag
  - verify_summary.txt   counts
  - the top N pairs are also written as side-by-side pictures in the scratchpad (not the repo),
    because they contain dataset images

CPU module, runs locally.

Usage:
    python 04b_near_duplicate_verify.py --nearest results/04_near_duplicates/nearest.csv --data-root "<data-root>" --output results/04b_verify --top 200 --sheet-dir <folder>
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
COPY = 0.9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gray(path):
    with Image.open(path) as im:
        return im.convert("L").resize((SIZE, SIZE), Image.BILINEAR)


def norm(im):
    a = np.asarray(im, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def best_corr(t_img, s_img):
    base_t = norm(t_img)
    best, label = -1.0, ""
    for flip in (False, True):
        s = s_img.transpose(Image.FLIP_LEFT_RIGHT) if flip else s_img
        for k in range(4):
            c = float(np.mean(base_t * norm(s.rotate(90 * k))))
            if c > best:
                best, label = c, f"rot{90 * k}{'+flip' if flip else ''}"
    return best, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nearest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--sheet-dir", default=None)
    args = ap.parse_args()
    root = Path(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/3: selecting top candidate pairs")
    with open(args.nearest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["similarity"]), reverse=True)
    top = rows[:args.top]
    log(f"  {len(top)} candidates, similarity {top[-1]['similarity']} to {top[0]['similarity']}")

    log("step 2/3: pixel-level check of each pair")
    results = []
    for i, r in enumerate(top, 1):
        if i % 25 == 0:
            log(f"  {i}/{len(top)}")
        t = gray(root / r["target_path"])
        s = gray(root / r["source_path"])
        c, label = best_corr(t, s)
        results.append({**r, "pixel_correlation": round(c, 4), "transform": label,
                        "copy": int(c >= COPY)})

    log("step 3/3: writing outputs")
    with open(out / "verify_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    copies = sum(r["copy"] for r in results)
    best = max(float(r["pixel_correlation"]) for r in results)
    summary = (f"candidates: {len(results)}; pixel copies (correlation >= {COPY}): {copies}; "
               f"highest pixel correlation: {best}")
    (out / "verify_summary.txt").write_text(summary + "\n", encoding="utf-8")
    log(f"done: {summary}")

    if args.sheet_dir:
        sd = Path(args.sheet_dir)
        sd.mkdir(parents=True, exist_ok=True)
        tile, pad = 240, 10
        n = min(12, len(results))
        sheet = Image.new("RGB", (2 * (tile + pad) + pad + 260, n * (tile + pad) + pad + 20), "white")
        d = ImageDraw.Draw(sheet)
        d.text((pad, 4), "target (left) | closest source (right) | similarity | pixel correlation", fill="black")
        for i, r in enumerate(results[:n]):
            y = 20 + pad + i * (tile + pad)
            for j, p in enumerate([r["target_path"], r["source_path"]]):
                try:
                    im = Image.open(root / p).convert("RGB")
                    im.thumbnail((tile, tile))
                except Exception:
                    im = Image.new("RGB", (tile, tile), "white")
                x = pad + j * (tile + pad)
                sheet.paste(im, (x, y))
                d.rectangle([x, y, x + tile, y + tile], outline="black")
            d.text((2 * (tile + pad) + pad, y + 10),
                   f"{i + 1}. sim {r['similarity']}\npixel corr {r['pixel_correlation']}\n"
                   f"{Path(r['target_path']).name[:26]}\n{Path(r['source_path']).name[:26]}", fill="black")
        sheet.save(sd / "top_pairs.png")
        log(f"sheet written: {sd / 'top_pairs.png'}")


if __name__ == "__main__":
    main()
