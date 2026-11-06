"""Fig augmentation check (audit item D3).

Fig file names have the form Class_NNNN_Transform.png (for example Mosaic_0001_Brightness.png).
For each augmented image, find the Original of the same class and number, and test whether it is
a copy of that Original under rotation, flip, or brightness and contrast change. A match means the
augmented image carries no new information for evaluation.

Method: each image is converted to 48 x 48 grayscale and normalised to zero mean and unit variance
(this removes brightness and contrast changes). The original is rotated in 5-degree steps and
flipped; the highest correlation with the augmented image is kept. A correlation of at least MATCH
counts as a match. Rotations are done on the small images, so the run is fast.

Inputs: the fig dataset folder (read only). Outputs: one CSV row per augmented image, and a summary.

CPU, runs locally. Progress is logged every 500 images.

Usage:
    python fig_augmentation_check.py --input "<data-root>/fig_sartak_sfld" --output results/fig_augmentation_check
"""

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

NAME = re.compile(r"^(.+?)_(\d{4})")
SIZE = 48
MATCH = 0.9
ANGLES = range(0, 360, 5)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def small_gray(path):
    """Load an image as a 48 x 48 grayscale PIL image."""
    with Image.open(path) as im:
        return im.convert("L").resize((SIZE, SIZE), Image.BILINEAR)


def normalised(im):
    a = np.asarray(im, dtype=np.float64)
    return (a - a.mean()) / (a.std() + 1e-9)


def best_match(aug_norm, orig_small):
    """Highest correlation between the augmented image and rotated and flipped originals."""
    best, label = -1.0, ""
    for flip in (False, True):
        base = orig_small.transpose(Image.FLIP_LEFT_RIGHT) if flip else orig_small
        for ang in ANGLES:
            rot = base.rotate(ang, resample=Image.BILINEAR, fillcolor=0)
            c = float(np.mean(aug_norm * normalised(rot)))
            if c > best:
                best, label = c, f"rot{ang}{'+flip' if flip else ''}"
    return best, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("step 1/3: listing Original and Augmented images")
    originals, augmented = {}, []
    for p in root.rglob("*.png"):
        m = NAME.match(p.name)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if "Original" in p.parts:
            originals[key] = p
        elif "Augmented" in p.parts:
            augmented.append((key, p))
    log(f"  {len(originals)} Original, {len(augmented)} Augmented")

    log("step 2/3: matching augmented images to their originals")
    orig_cache = {}
    rows = []
    for i, (key, p) in enumerate(augmented, 1):
        if i % 500 == 0:
            log(f"  {i}/{len(augmented)}")
        rel = str(p.relative_to(root))
        if key not in originals:
            rows.append({"augmented": rel, "class": key[0], "number": key[1], "original_found": 0,
                         "correlation": "", "transform": "", "match": 0})
            continue
        if key not in orig_cache:
            orig_cache[key] = small_gray(originals[key])
        c, t = best_match(normalised(small_gray(p)), orig_cache[key])
        rows.append({"augmented": rel, "class": key[0], "number": key[1], "original_found": 1,
                     "correlation": round(c, 4), "transform": t, "match": int(c >= MATCH)})

    log("step 3/3: writing results")
    with open(out / "fig_augmentation_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["augmented"])
        w.writeheader()
        w.writerows(rows)
    matched = sum(r["match"] for r in rows)
    no_orig = sum(1 for r in rows if not r["original_found"])
    summary = (f"augmented: {len(rows)}; matched to an original (correlation >= {MATCH}): {matched}; "
               f"no original with same class and number: {no_orig}; unmatched: {len(rows) - matched - no_orig}")
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    log(f"done: {summary}")


if __name__ == "__main__":
    main()
