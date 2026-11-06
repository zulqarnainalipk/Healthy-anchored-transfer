"""Audit the plant disease source and target datasets.

Reads the dataset folder (read only). It walks the folder tree, reads image headers, hashes
every file, and writes CSV tables, two figures and a summary report. Nothing from the dataset
is copied to the output folder.

Usage:
    python audit_dataset.py --input path/to/plant-paper-all-sources --output path/to/out
"""

import argparse
import collections
import csv
import hashlib
import os
import re
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_OUTPUT = "results/code_audit_result"
BACKGROUND_SAMPLE = 200       # images per dataset decoded for background and colour checks
TREE_DEPTH = 4
HEALTHY_WORDS = ("healthy", "health_leaf", "health leaf")
LICENCE_WORDS = ("licen", "readme", "license", "citation")


DATASET_NAME = "plant-paper-all-sources"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_input(arg):
    """Return the dataset root (the folder named plant-paper-all-sources).

    The data may sit at different depths under the mount point, so search for the name.
    """
    if arg:
        return Path(arg)
    env = os.environ.get("PLANT_DATA_ROOT")
    if env:
        return Path(env)
    raise FileNotFoundError("pass --input (or set PLANT_DATA_ROOT) to the dataset folder")


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def background_kind(img):
    """Classify the background from the four corner pixels of a small thumbnail."""
    small = img.convert("RGB")
    small.thumbnail((128, 128))
    w, h = small.size
    corners = [small.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    mean = sum(sum(c) for c in corners) / (3 * len(corners))
    if mean < 30:
        return "black"
    if mean > 225:
        return "white"
    return "other"


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path.name, len(rows), "rows")


def folder_tree(root, out_path):
    lines = []

    def walk(d, depth):
        if depth > TREE_DEPTH:
            return
        subs = sorted(p for p in d.iterdir() if p.is_dir())
        files = [p for p in d.iterdir() if p.is_file()]
        lines.append("  " * depth + f"{d.name}/  ({len(files)} files, {len(subs)} folders)")
        for s in subs:
            walk(s, depth + 1)

    walk(root, 0)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out_path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--no-hash", action="store_true", help="skip file hashing (headers and sizes only)")
    args = ap.parse_args()
    do_hash = not args.no_hash
    t0 = time.time()
    root = find_input(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log(f"step 1/6: input {root}")

    inventory = []      # one row per file
    corrupt = []
    ext_count = collections.Counter()
    ext_bytes = collections.Counter()
    sampled_bg = collections.defaultdict(list)
    sampled_mode = collections.defaultdict(list)

    log("step 2/6: walking files, hashing, reading image headers")
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if len(inventory) % 5000 == 0:
                log(f"  processed {len(inventory)} files, {time.time() - t0:.0f} s elapsed")
            path = Path(dirpath) / name
            rel = path.relative_to(root)
            parts = rel.parts
            dataset = parts[0] if len(parts) > 1 else "(root)"
            ext = path.suffix.lower()
            size = path.stat().st_size
            ext_count[(dataset, ext)] += 1
            ext_bytes[(dataset, ext)] += size
            label = parts[-2] if len(parts) > 2 else ""
            is_aug = any("augment" in p.lower() for p in parts)
            row = {
                "dataset": dataset,
                "path": str(rel).replace("\\", "/"),
                "top_folder": parts[1] if len(parts) > 2 else "",
                "class_folder": label,
                "is_healthy": int(any(w in label.lower() for w in HEALTHY_WORDS)),
                "is_augmented_path": int(is_aug),
                "ext": ext,
                "bytes": size,
                "width": "",
                "height": "",
                "mode": "",
                "md5": md5_of(path) if do_hash else "",
                "is_image": int(ext in IMAGE_EXT),
            }
            if ext in IMAGE_EXT:
                try:
                    with Image.open(path) as img:
                        row["width"], row["height"], row["mode"] = img.width, img.height, img.mode
                        if len(sampled_bg[dataset]) < BACKGROUND_SAMPLE:
                            img.load()
                            sampled_bg[dataset].append(background_kind(img))
                            sampled_mode[dataset].append(img.mode)
                except Exception as err:  # noqa: BLE001
                    corrupt.append({"dataset": dataset, "path": row["path"], "error": str(err)[:200]})
            inventory.append(row)

    log(f"step 3/6: walk done, {len(inventory)} files, {len(corrupt)} unreadable images")

    # ---- inventory and file types
    write_csv(out / "inventory.csv", inventory, list(inventory[0].keys()) if inventory else ["dataset"])
    write_csv(out / "file_types.csv",
              [{"dataset": d, "ext": e, "files": c, "bytes": ext_bytes[(d, e)]}
               for (d, e), c in sorted(ext_count.items())],
              ["dataset", "ext", "files", "bytes"])
    write_csv(out / "corrupt_files.csv", corrupt, ["dataset", "path", "error"])

    datasets = sorted({r["dataset"] for r in inventory})
    images = [r for r in inventory if r["is_image"]]

    # ---- dataset summary
    summary = []
    for d in datasets:
        rows = [r for r in inventory if r["dataset"] == d]
        imgs = [r for r in rows if r["is_image"]]
        names = [str(Path(r["path"]).name).lower() for r in rows]
        has_doc = any(any(w in n for w in LICENCE_WORDS) for n in names)
        summary.append({
            "dataset": d,
            "files": len(rows),
            "images": len(imgs),
            "gigabytes": round(sum(r["bytes"] for r in rows) / 1e9, 3),
            "class_folders": len({r["class_folder"] for r in imgs}),
            "augmented_path_images": sum(r["is_augmented_path"] for r in imgs),
            "has_licence_or_readme": int(has_doc),
        })
    write_csv(out / "dataset_summary.csv", summary,
              ["dataset", "files", "images", "gigabytes", "class_folders",
               "augmented_path_images", "has_licence_or_readme"])

    # ---- class and healthy/disease counts
    class_rows = []
    hd_rows = []
    for d in datasets:
        imgs = [r for r in images if r["dataset"] == d]
        counts = collections.Counter((r["class_folder"], r["is_healthy"]) for r in imgs)
        for (cls, healthy), c in sorted(counts.items()):
            class_rows.append({"dataset": d, "class_folder": cls, "is_healthy": healthy, "images": c})
        h = sum(r["is_healthy"] for r in imgs)
        hd_rows.append({"dataset": d, "healthy_images": h, "other_images": len(imgs) - h})
    write_csv(out / "class_counts.csv", class_rows, ["dataset", "class_folder", "is_healthy", "images"])
    write_csv(out / "healthy_disease_counts.csv", hd_rows,
              ["dataset", "healthy_images", "other_images"])

    # ---- resolution, colour mode, background
    res_rows = []
    bg_rows = []
    for d in datasets:
        imgs = [r for r in images if r["dataset"] == d and r["width"] != ""]
        if not imgs:
            continue
        ws = [r["width"] for r in imgs]
        hs = [r["height"] for r in imgs]
        modes = collections.Counter(r["mode"] for r in imgs)
        res_rows.append({
            "dataset": d,
            "images_with_size": len(imgs),
            "width_min": min(ws), "width_median": statistics.median(ws), "width_max": max(ws),
            "height_min": min(hs), "height_median": statistics.median(hs), "height_max": max(hs),
            "modes": "; ".join(f"{m}:{c}" for m, c in modes.most_common()),
        })
        bgs = collections.Counter(sampled_bg[d])
        bg_rows.append({
            "dataset": d,
            "sampled": len(sampled_bg[d]),
            "black": bgs.get("black", 0), "white": bgs.get("white", 0), "other": bgs.get("other", 0),
        })
    write_csv(out / "resolution_summary.csv", res_rows,
              ["dataset", "images_with_size", "width_min", "width_median", "width_max",
               "height_min", "height_median", "height_max", "modes"])
    write_csv(out / "background_summary.csv", bg_rows,
              ["dataset", "sampled", "black", "white", "other"])

    # ---- exact duplicates
    by_hash = collections.defaultdict(list)
    for r in inventory:
        by_hash[r["md5"]].append(r)
    dup_groups = []
    dup_rows = []
    for md5, members in by_hash.items():
        if len(members) < 2:
            continue
        sets = sorted({m["dataset"] for m in members})
        dup_groups.append({
            "md5": md5, "copies": len(members), "datasets": "; ".join(sets),
            "paths": " | ".join(m["path"] for m in members[:10]),
        })
    write_csv(out / "exact_duplicate_groups.csv", dup_groups,
              ["md5", "copies", "datasets", "paths"])
    for d in datasets:
        in_d = [g for g in dup_groups if d in g["datasets"].split("; ")]
        cross = [g for g in in_d if "; " in g["datasets"]]
        dup_rows.append({
            "dataset": d,
            "duplicate_groups_touching_dataset": len(in_d),
            "groups_shared_with_other_datasets": len(cross),
        })
    write_csv(out / "duplicate_summary.csv", dup_rows,
              ["dataset", "duplicate_groups_touching_dataset", "groups_shared_with_other_datasets"])

    # ---- id patterns in file names (plant or photo ids)
    pattern_rows = []
    for d in datasets:
        names = [Path(r["path"]).stem for r in images if r["dataset"] == d]
        with_digits = [n for n in names if re.search(r"\d", n)]
        patterns = collections.Counter(re.sub(r"\d+", "#", n) for n in names)
        pattern_rows.append({
            "dataset": d,
            "images": len(names),
            "names_with_digits": len(with_digits),
            "top_patterns": "; ".join(f"{p} ({c})" for p, c in patterns.most_common(3)),
        })
    write_csv(out / "id_pattern_summary.csv", pattern_rows,
              ["dataset", "images", "names_with_digits", "top_patterns"])

    folder_tree(root, out / "folder_tree.txt")

    # ---- figures (black and white)
    if res_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for d in datasets:
            ws = [r["width"] for r in images if r["dataset"] == d and r["width"] != ""]
            if ws:
                ax.hist(ws, bins=30, histtype="step", color="black", linestyle="-", label=d)
        ax.set_xlabel("image width (px)")
        ax.set_ylabel("images")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_resolution_hist.png", dpi=200, facecolor="white")
        plt.close(fig)
        print("wrote fig_resolution_hist.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["dataset"] for r in hd_rows],
           [r["healthy_images"] for r in hd_rows], color="white", edgecolor="black", hatch="//",
           label="healthy")
    ax.bar([r["dataset"] for r in hd_rows],
           [r["other_images"] for r in hd_rows], bottom=[r["healthy_images"] for r in hd_rows],
           color="white", edgecolor="black", label="other")
    ax.set_ylabel("images")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_healthy_disease_counts.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote fig_healthy_disease_counts.png")

    # ---- short report
    lines = ["# Code audit result (automatic)", "",
             f"Input: `{root}`", f"Files: {len(inventory)}, images: {len(images)}, "
             f"corrupt images: {len(corrupt)}, duplicate groups: {len(dup_groups)}", "",
             "## Datasets", "", "| dataset | images | GB | class folders | augmented-path images | licence/readme |",
             "|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['dataset']} | {s['images']} | {s['gigabytes']} | {s['class_folders']} | "
                     f"{s['augmented_path_images']} | {'yes' if s['has_licence_or_readme'] else 'no'} |")
    (out / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote audit_report.md")
    print("done")


if __name__ == "__main__":
    main()
