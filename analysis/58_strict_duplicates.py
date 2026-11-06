"""Module 58: a stricter near-duplicate rule for the PlantWild real-pair test (sensitivity check).

The registered rule (module 08b) used a 64-bit difference hash with Hamming distance <= 3. Here the distance is <= 8. Groups of
images connected by such pairs are formed; a group that spans more than one class is removed completely, and from a group inside
one class only the lexicographically first path is kept (the same logic as before). The images that are removed in addition to
those already absent from the features are written to results/58_strict_duplicates/extra_exclude.csv (column path), which
module 53 can read with --exclude-file.

CPU module, about 8 minutes (reads the 17,328 images). Usage:
    python 58_strict_duplicates.py --data-root <data>/plantwild --features <features dir> --distance 8
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dhash(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--distance", type=int, default=8)
    ap.add_argument("--output", default="results/58_strict_duplicates")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = list(csv.DictReader(open(Path(args.features) / "features_index.csv", encoding="utf-8")))
    log(f"step 1/3: hashing {len(meta)} images")
    hashes = np.zeros(len(meta), dtype=np.uint64)
    for i, m in enumerate(meta):
        with Image.open(Path(args.data_root) / "images" / m["path"]) as im:
            hashes[i] = np.uint64(dhash(im))
        if (i + 1) % 3000 == 0:
            log(f"    {i + 1}/{len(meta)}")
    log("step 2/3: pairs within the distance (pigeonhole buckets)")
    nseg = args.distance + 1
    bounds = np.linspace(0, 64, nseg + 1).astype(int)
    pairs = set()
    pop = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)
    for s in range(nseg):
        lo, hi = bounds[s], bounds[s + 1]
        seg = (hashes >> np.uint64(lo)) & np.uint64((1 << (hi - lo)) - 1)
        buckets = defaultdict(list)
        for i, v in enumerate(seg):
            buckets[int(v)].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            h = hashes[idx]
            x = h[:, None] ^ h[None, :]
            d = pop[(x & np.uint64(0xFFFF)).astype(np.int64)] + pop[((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.int64)] \
                + pop[((x >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64)] + pop[((x >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.int64)]
            a, b = np.where(np.triu(d <= args.distance, 1))
            for u, v in zip(a, b):
                pairs.add((idx[u], idx[v]))
        log(f"    segment {s + 1}/{nseg}, {len(pairs)} pairs so far")
    log("step 3/3: groups and exclusions")
    par = list(range(len(meta)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in pairs:
        par[find(a)] = find(b)
    comp = defaultdict(list)
    for i in range(len(meta)):
        comp[find(i)].append(i)
    excl = []
    for members in comp.values():
        if len(members) < 2:
            continue
        classes = {meta[i]["class"] for i in members}
        if len(classes) > 1:
            excl += [meta[i]["path"] for i in members]
        else:
            excl += sorted(meta[i]["path"] for i in members)[1:]
    with open(out / "extra_exclude.csv", "w", encoding="utf-8") as f:
        f.write("path\n" + "\n".join(sorted(excl)) + "\n")
    log(f"  {len(pairs)} pairs, {sum(1 for m in comp.values() if len(m) > 1)} groups, {len(excl)} images to remove from {len(meta)}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
