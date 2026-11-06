"""Module 16: synthetic worlds with known truth (proposal 7.4, primary benchmark).

Each world is defined by how a target crop is related to a source crop, in the 16-dimensional
projected space (module 08). Source samples are real features (PlantVillage healthy and disease,
training split). Target samples are generated from the source samples through the world's known
operator, so the true relationship is known exactly.

Worlds (proposal 7.4):
  a    affine relation: target = M x + t (a transport map; a tie is expected)
  a2   additive shift: target = x + s, not an affine transport (operator transfer expected to win)
  b    exact radial relation: target = m + r (x - m) for every pair (best case)
  b2   maps that do not compose: a different random affine per source crop
  c    graded departure: target = M x + lambda * phi(x), lambda in {0.25, 0.5, 1}
  d    all sources wrong in the same way: sources share a common wrong shift, the target does not
  e    rotation plus shear that is not a transport map (general affine registration expected to win)
  f    real features with a known operator: target = real features of another crop + known shift

Each world writes, per replicate, the target healthy sample and target disease samples (per class),
with sample sizes drawn from the real datasets (healthy 50 and 200; disease 10 and 50 per class), and the
true operator parameters.

Outputs (results/16_worlds/):
  - world_<name>.npz     true operator and generated samples for the first replicate
  - worlds.csv           world, parameters, sample sizes, replicate count
  - manifest.json

CPU module, runs locally.

Usage:
    python 16_world_generator.py --features results/05_features_dinov2/remote/features_dinov2
        --labelled results/01_crop_disease_table/labelled_images.csv
        --projection results/08_projection/projection_k16.npz --output results/16_worlds
"""

import argparse
import csv
import json
import time
import zlib
from pathlib import Path

import numpy as np

SOURCE_DATASETS = ["plantvillage_color", "plantseg_v6"]
if __import__("os").environ.get("SOURCE_ONLY"):   # ablation: restrict the source crops to one dataset (module 51)
    SOURCE_DATASETS = [s for s in SOURCE_DATASETS if s == __import__("os").environ["SOURCE_ONLY"]]
REPS = 200
SEED = 2026
HEALTHY_N = [50, 200]
DISEASE_N = [10, 50]
MIN_TRAIN = 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


MIN_GROUP = 60          # smallest crop-disease group used in the benchmark (so each part has at least 15)
SPLIT = (0.5, 0.25, 0.25)  # source models, target test samples, probe training samples


ALIASES = {"cedar apple rust": "rust", "bacterial leaf spot": "bacterial spot"}


def canonical_class(crop, cls):
    """Harmonised class name: lower case, underscores to spaces, crop prefix removed, a few same-disease aliases.

    The two source datasets name the same disease differently (for example 'Early_blight' and 'potato early blight'),
    which splits one disease into two classes inside a crop. Used by the harmonised sensitivity runs (module 03b)."""
    name = cls.lower().replace("_", " ").strip()
    first = crop.lower().replace("_", " ").split()[0]
    if name.startswith(first + " "):
        name = name[len(first) + 1:]
    return ALIASES.get(name, name)


def harmonise_samples(samples):
    """Merge groups of the same crop whose harmonised class names agree (healthy is kept as it is)."""
    merged = {}
    for (crop, cls), arr in samples.items():
        key = (crop, "healthy" if cls == "healthy" else canonical_class(crop, cls))
        merged.setdefault(key, []).append(arr)
    return {k: np.vstack(v) for k, v in merged.items()}


def split_samples(samples):
    """Split every group into three DISJOINT parts: source models, target test, probe training.

    Used by the benchmark modules (17, 18, 20, 26) so that no image used to fit a source model or to train
    a probe appears among the test samples. Returns (source, test, train) dicts with the same keys."""
    src, test, train = {}, {}, {}
    for key, arr in samples.items():
        if len(arr) < MIN_GROUP:
            continue
        rng = np.random.default_rng(SEED + zlib.crc32(str(key).encode("utf-8")) % 100000)
        perm = rng.permutation(len(arr))
        a = int(SPLIT[0] * len(arr))
        b = a + int(SPLIT[1] * len(arr))
        src[key], test[key], train[key] = arr[perm[:a]], arr[perm[a:b]], arr[perm[b:]]
    return src, test, train


def random_affine(rng, scale=0.15, dim=16):
    """A random invertible dim x dim matrix near the identity."""
    M = np.eye(dim) + scale * rng.standard_normal((dim, dim)) / 4.0
    return M


def operators(name, rng, mu, n_crops, s=1.0):
    """Return a function(x, crop_index) -> transformed x, and the true parameters for the world.

    s multiplies the strength of every operator (shifts and affine departures from the identity).
    The default 1.0 is the first-draft strength; the strength used in the benchmark is calibrated
    on the no-adaptation score alone (see docs/modules/16_world_generator.md)."""
    dim = len(mu)
    if name == "a":
        M = random_affine(rng, 0.15 * s, dim)
        t = 0.3 * s * rng.standard_normal(dim)
        return (lambda x, k: x @ M.T + t), {"M": M.tolist(), "t": t.tolist()}
    if name == "a2":
        sh = 0.5 * s * rng.standard_normal(dim)
        return (lambda x, k: x + sh), {"shift": sh.tolist()}
    if name == "b":
        r = 1.0 + 0.3 * s
        return (lambda x, k: mu + r * (x - mu)), {"radius_factor": r}
    if name == "b2":
        Ms = [random_affine(rng, 0.15 * s, dim) for _ in range(n_crops)]
        return (lambda x, k: x @ Ms[k].T), {"per_crop_M": [m.tolist() for m in Ms]}
    if name.startswith("c"):
        lam = {"c25": 0.25, "c50": 0.5, "c100": 1.0}[name]
        M = random_affine(rng, 0.15 * s, dim)
        def phi(x):
            return np.tanh(x)
        return (lambda x, k: x @ M.T + lam * s * phi(x)), {"M": M.tolist(), "lambda": lam}
    if name == "d":
        shift = 0.5 * s * rng.standard_normal(dim)
        return (lambda x, k: x + shift), {"common_wrong_shift": shift.tolist(), "target_true_map": "identity"}
    if name == "e":
        R = np.linalg.qr(rng.standard_normal((dim, dim)))[0]
        Sh = np.eye(dim) + 0.4 * s * np.triu(rng.standard_normal((dim, dim)), 1)
        M = R @ Sh
        return (lambda x, k: x @ M.T), {"M_rotation_shear": M.tolist()}
    if name == "f":
        sh = 0.4 * s * rng.standard_normal(dim)
        return (lambda x, k: x + sh), {"known_shift": sh.tolist()}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading source features and labels")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in SOURCE_DATASETS and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    samples = {}   # (crop, disease or 'healthy') -> array of projected training features
    rows_of = {ds: [i for i, r in enumerate(index) if r["dataset"] == ds] for ds in SOURCE_DATASETS}
    for ds in SOURCE_DATASETS:
        Z = (np.load(feats / f"features_{ds}.npy").astype(np.float64) - mean) @ W
        for k, i in enumerate(rows_of[ds]):
            r = index[i]
            if r["split"] != "train":
                continue
            crop = crop_of.get(r["path"])
            if crop is None:
                continue
            key = (crop, "healthy" if r["is_healthy"] == "1" else r["class"])
            samples.setdefault(key, []).append(Z[k])
    samples = {k: np.array(v) for k, v in samples.items() if len(v) >= MIN_TRAIN}
    crops = sorted({k[0] for k in samples if k[1] == "healthy"})
    log(f"  {len(samples)} crop-disease groups; {len(crops)} crops with healthy samples")

    log("step 2/3: generating worlds")
    rng = np.random.default_rng(SEED)
    mu = np.vstack(list(samples.values())).mean(axis=0)
    world_names = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]
    worlds = []
    for name in world_names:
        op, truth = operators(name, rng, mu, len(crops))
        crop_index = {c: i for i, c in enumerate(crops)}
        healthy_src = {c: samples[(c, "healthy")] for c in crops}
        first = {}
        for hn in HEALTHY_N:
            for dn in DISEASE_N:
                reps = []
                for rep in range(REPS):
                    r2 = np.random.default_rng(SEED + rep)
                    crop = crops[rep % len(crops)]
                    k = crop_index[crop]
                    H = op(healthy_src[crop][r2.choice(len(healthy_src[crop]), size=hn, replace=True)], k)
                    D = {}
                    for (c, dis), arr in samples.items():
                        if dis == "healthy" or c != crop:
                            continue
                        D[dis] = op(arr[r2.choice(len(arr), size=dn, replace=True)], k)
                    reps.append({"healthy": H, "disease": D})
                first[(hn, dn)] = reps[0]
        worlds.append({"world": name, "truth": json.dumps(truth)[:200], "healthy_n": HEALTHY_N,
                       "disease_n": DISEASE_N, "replicates": REPS})
        np.savez(out / f"world_{name}.npz", **{f"H_{hn}_{dn}": first[(hn, dn)]["healthy"]
                                               for hn in HEALTHY_N for dn in DISEASE_N},
                 truth=np.array(json.dumps(truth)))
        log(f"  world {name}: samples written")

    log("step 3/3: writing table and manifest")
    with open(out / "worlds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(worlds[0].keys()))
        w.writeheader()
        w.writerows(worlds)
    manifest = {"module": "16_world_generator", "worlds": world_names, "replicates": REPS,
                "healthy_n": HEALTHY_N, "disease_n": DISEASE_N, "seed": SEED,
                "crops_with_healthy": crops, "seconds": round(time.time() - t0, 1),
                "note": "minimum-detectable-difference and power are computed in module 36b"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"done: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()


