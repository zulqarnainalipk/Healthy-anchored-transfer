"""Audit of PlantWild v1 for real source-target crop pairs (module 08b).

Question: does PlantWild provide usable real pairs (two crops that share a disease, with healthy images of the target crop)?

Reads the dataset folder (images/<class>/..., classes.txt, trainval.txt, url_record.json) and writes, to --output:
  class_table.csv        class, role (healthy or disease), crop, disease, images, unreadable files, image size statistics
  expected_check.csv     folders against classes.txt
  duplicates.csv         exact (MD5) and near duplicates (64-bit difference hash, Hamming distance <= 3)
  duplicate_summary.csv  per class: images in duplicate pairs and their share
  source_urls.csv        per class: images, distinct source pages, most common source domain and its share
  pair_table.csv         ordered source-target crop pairs per shared disease type, tier, counts and count-rule flag
  pair_summary.csv       pairs and target crops passing the count rule, per tier
  samples/*.jpg          contact sheets of key classes
  report.md, manifest.json
Tiers: 1 = same pathogen species (tomato and potato early blight and late blight); 2 = same specific disease type with
different species (rust, mosaic virus, powdery mildew, downy mildew, early blight, late blight, black rot, anthracnose,
sheath blight, alternaria leaf spot); 3 = generic leaf spot or leaf blight (weak).
Count rule: target healthy images >= 50, source disease images >= 30, target disease images >= 30.

CPU module. Logs every step and progress.
Usage: python audit_plantwild.py --data-root <data>/plantwild --output results/plantwild_audit
"""

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
SPECIFIC = ["rust", "mosaic virus", "powdery mildew", "downy mildew", "early blight", "late blight", "black rot", "anthracnose",
            "sheath blight", "alternaria leaf spot"]
GENERIC = ["leaf spot", "leaf blight"]
TIER1 = {"early blight": {"potato", "tomato"}, "late blight": {"potato", "tomato"}}
MIN_HEALTHY, MIN_SRC, MIN_TGT = 50, 30, 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dhash(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    v = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.output)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/7: classes and folders")
    classes = [ln.strip().split(" ", 1)[1] for ln in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    folders = {p.name: p for p in (root / "images").iterdir() if p.is_dir()}
    with open(out / "expected_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "in_classes_txt", "folder_exists"])
        for n in sorted(set(classes) | set(folders)):
            w.writerow([n, n in classes, n in folders])
    log(f"  classes.txt {len(classes)}, folders {len(folders)}, matching {len(set(classes) & set(folders))}")

    healthy = {c for c in classes if c.endswith(" leaf")}
    crops = sorted({c[:-5] for c in healthy}, key=len, reverse=True)
    urls = json.loads((root / "url_record.json").read_text(encoding="utf-8"))

    log("step 2/7: reading every image (size, MD5, difference hash)")
    rows, per_image = [], []
    for ci, c in enumerate(classes, 1):
        role = "healthy" if c in healthy else "disease"
        crop = c[:-5] if role == "healthy" else next((x for x in crops if c.startswith(x + " ")), "")
        disease = "" if role == "healthy" else (c[len(crop) + 1:] if crop else c)
        files = [q for q in folders[c].rglob("*") if q.is_file() and q.suffix.lower() in IMG_EXT]
        ok, bad, ms, kb = 0, 0, [], []
        for q in files:
            try:
                data = q.read_bytes()
                with Image.open(q) as im:
                    im.load()
                    w_, h_ = im.size
                    dh = dhash(im)
                per_image.append((c, str(q.relative_to(root / "images")).replace("\\", "/"), hashlib.md5(data).hexdigest(), dh))
                ms.append(min(w_, h_)); kb.append(len(data) / 1024); ok += 1
            except Exception:
                bad += 1
        rows.append({"class": c, "role": role, "crop": crop, "disease": disease, "images": ok, "unreadable": bad,
                     "min_side_median": int(np.median(ms)) if ms else 0, "min_side_min": min(ms) if ms else 0,
                     "median_kb": int(np.median(kb)) if kb else 0})
        if ci % 10 == 0 or ci == len(classes):
            log(f"    {ci}/{len(classes)} classes, {len(per_image)} images")
    with open(out / "class_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["images"] for r in rows)
    log(f"  {total} readable images, {sum(r['unreadable'] for r in rows)} unreadable")

    log("step 3/7: duplicates")
    by_md5 = defaultdict(list)
    for i, r in enumerate(per_image):
        by_md5[r[2]].append(i)
    dups, involved = [], defaultdict(set)
    for idx in by_md5.values():
        for k in idx[1:]:
            dups.append({"kind": "exact", "a": per_image[idx[0]][1], "b": per_image[k][1], "class_a": per_image[idx[0]][0], "class_b": per_image[k][0], "hamming": 0})
            involved[per_image[k][0]].add(k); involved[per_image[idx[0]][0]].add(idx[0])
    seg = [defaultdict(list) for _ in range(4)]
    for i, r in enumerate(per_image):
        for s in range(4):
            seg[s][(r[3] >> (16 * s)) & 0xFFFF].append(i)
    seen = set()
    for s in range(4):
        for idx in seg[s].values():
            if 2 <= len(idx) <= 300:
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        i, j = idx[a], idx[b]
                        if (i, j) in seen:
                            continue
                        seen.add((i, j))
                        ham = bin(per_image[i][3] ^ per_image[j][3]).count("1")
                        if ham <= 3 and per_image[i][2] != per_image[j][2]:
                            dups.append({"kind": "near", "a": per_image[i][1], "b": per_image[j][1], "class_a": per_image[i][0], "class_b": per_image[j][0], "hamming": ham})
                            involved[per_image[i][0]].add(i); involved[per_image[j][0]].add(j)
        log(f"    hash segment {s + 1}/4 done, {len(dups)} duplicate pairs")
    with open(out / "duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "a", "b", "class_a", "class_b", "hamming"])
        w.writeheader()
        w.writerows(dups)
    cnt = {r["class"]: r["images"] for r in rows}
    with open(out / "duplicate_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "images", "images_in_duplicate_pairs", "share"])
        for c in classes:
            k = len(involved.get(c, ()))
            w.writerow([c, cnt[c], k, round(k / max(cnt[c], 1), 3)])
    cross = sum(1 for r in dups if r["class_a"] != r["class_b"])
    log(f"  duplicate pairs {len(dups)} (exact {sum(1 for r in dups if r['kind'] == 'exact')}), across classes {cross}")

    log("step 4/7: source pages and domains")
    src = defaultdict(list)
    for path, url in urls.items():
        parts = path.split("/")
        if len(parts) >= 3:
            src[parts[2]].append(url)
    with open(out / "source_urls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "urls", "distinct_pages", "top_domain", "top_domain_share"])
        for c in classes:
            u = src.get(c, [])
            dom = Counter(urlparse(x).netloc for x in u)
            top = dom.most_common(1)[0] if dom else ("", 0)
            w.writerow([c, len(u), len(set(u)), top[0], round(top[1] / max(len(u), 1), 3)])

    log("step 5/7: pair table")
    hcount = {r["crop"]: r["images"] for r in rows if r["role"] == "healthy"}
    dcount = {(r["crop"], r["disease"]): r["images"] for r in rows if r["role"] == "disease" and r["crop"]}
    pairs = []
    for dtype in SPECIFIC + GENERIC:
        members = {k: n for k, n in dcount.items() if k[1] == dtype or k[1].endswith(" " + dtype)}
        cs = sorted({c for c, _ in members})
        for s in cs:
            for t in cs:
                if s == t:
                    continue
                ns = sum(n for (c, d), n in members.items() if c == s)
                nt = sum(n for (c, d), n in members.items() if c == t)
                nh = hcount.get(t, 0)
                tier = 1 if (dtype in TIER1 and {s, t} <= TIER1[dtype]) else (3 if dtype in GENERIC else 2)
                pairs.append({"disease_type": dtype, "source": s, "target": t, "tier": tier, "source_disease_images": ns,
                              "target_disease_images": nt, "target_healthy_images": nh,
                              "passes_counts": int(nh >= MIN_HEALTHY and ns >= MIN_SRC and nt >= MIN_TGT)})
    with open(out / "pair_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        w.writeheader()
        w.writerows(pairs)
    summ = []
    for tier in (1, 2, 3):
        ok = [p for p in pairs if p["tier"] == tier and p["passes_counts"]]
        summ.append({"tier": tier, "pairs_all": sum(1 for p in pairs if p["tier"] == tier), "pairs_passing_counts": len(ok),
                     "target_crops": len({p["target"] for p in ok}), "disease_types": len({p["disease_type"] for p in ok})})
    with open(out / "pair_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    for s in summ:
        log(f"  tier {s['tier']}: {s['pairs_passing_counts']} of {s['pairs_all']} pairs pass; {s['target_crops']} target crops; {s['disease_types']} disease types")

    log("step 6/7: contact sheets")
    random.seed(1)
    keys = ["tomato leaf", "potato leaf", "tomato early blight", "potato early blight", "tomato late blight", "potato late blight",
            "squash leaf", "squash powdery mildew", "cucumber powdery mildew", "apple rust", "bean rust", "corn rust"]
    for k in keys:
        files = [q for q in folders[k].rglob("*") if q.suffix.lower() in IMG_EXT]
        sheet = Image.new("RGB", (8 * 160, 160), "white")
        for c, q in enumerate(random.sample(files, min(8, len(files)))):
            try:
                with Image.open(q) as im:
                    im = im.convert("RGB")
                    im.thumbnail((160, 160))
                    sheet.paste(im, (c * 160, 0))
            except Exception:
                pass
        sheet.save(out / "samples" / (k.replace(" ", "_") + ".jpg"), quality=80)

    log("step 7/7: report")
    hr = [r for r in rows if r["role"] == "healthy"]
    lines = ["# PlantWild v1 audit", "", f"- class folders {len(folders)} (classes.txt {len(classes)}); readable images {total}",
             f"- healthy classes {len(hr)} ({sum(r['images'] for r in hr)} images, smallest {min(r['images'] for r in hr)}); disease classes {len(rows) - len(hr)}",
             f"- duplicate pairs {len(dups)} (exact {sum(1 for r in dups if r['kind'] == 'exact')}), across classes {cross}", ""]
    lines += [f"- tier {s['tier']}: {s['pairs_passing_counts']} of {s['pairs_all']} ordered pairs pass the count rule; {s['target_crops']} target crops; {s['disease_types']} disease types" for s in summ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"module": "08b_audit_plantwild", "images": total, "classes": len(classes),
                                                   "duplicate_pairs": len(dups), "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
