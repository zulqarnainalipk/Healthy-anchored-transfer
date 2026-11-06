"""Module 51: projection fairness and calibration checks (design check; exploratory).

Question. The supervised projection of the main benchmark (module 08) is fitted on source classes that include the
classes of the crop later used as the target. That helps the few-shot probe and fixes the geometry in which HATC works.
Here the projection is fitted WITHOUT the target crop (leave-one-crop-out, LOCO), with LDA (supervised) or PCA
(unsupervised), and the benchmark is repeated. Operator strength can be calibrated separately for each projection with
the same rule as before (smallest strength with the mean no-adaptation accuracy in 0.40 to 0.65).

Modes
  fit        fit and cache the LOCO projections (one per usable target crop and kind)
  calibrate  no-adaptation accuracy against operator strength for each projection (to choose the strength)
  run        benchmark at one strength: NA, MC, STD, CORAL, HATC, S50, ZFGA-style (two readings), probe with 5, 10, 25 labels
Projections: base (the module 08 projection, includes the target crop), loco_lda, loco_pca (both 16-d, target crop excluded).

ZFGA-style baseline (our reading of Khan et al., arXiv 2609.37302, a technical note): Fisher information of the source
classifier's posterior with respect to the features, averaged over healthy source images (reference) and over healthy
target images (test), A = (I_te + eps I)^(-1/2) (I_ref + eps I)^(1/2), target features z' = A z ('ZFGA'), and the same
after mean-centring ('ZFGA_MC': z' = mu_A + A (z - mu_B)). It is not the authors' code.

CPU module, runs locally. LOCO LDA fitting takes about 2 to 3 minutes per crop.

Usage:
    python 51_loco_projection.py --mode fit --features ... --labelled ... --output results/51_review_checks/loco
    python 51_loco_projection.py --mode calibrate --features ... --labelled ... --base-projection ... --output ... --scales 1 1.5 2 3 4
    python 51_loco_projection.py --mode run --scale 2 --projections base loco_lda loco_pca --healthy-n 200 --reps-per-crop 17 ...
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from scipy.linalg import sqrtm

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w16 = _load("w16", "16_world_generator.py")
w17 = _load("w17", "17_benchmark_hatc.py")
w50 = _load("w50", "50_hatc_select.py")
m46 = _load("m46", "46b_metrics_cost.py")

WORLDS = ["a", "a2", "b", "b2", "c25", "c50", "c100", "d", "e", "f"]
DISEASE_N = 50
DIM = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_raw(feats, labelled):
    """Raw 1,536-d training-split features of the source datasets with crop, class and LDA label."""
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in w16.SOURCE_DATASETS and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    X, crop, key, lda_label = [], [], [], []
    for ds in w16.SOURCE_DATASETS:
        a = np.load(feats / f"features_{ds}.npy").astype(np.float32)
        rows = [r for r in index if r["dataset"] == ds]
        for r, vec in zip(rows, a):
            if r["split"] != "train":
                continue
            X.append(vec)
            crop.append(crop_of.get(r["path"], ""))
            key.append("healthy" if r["is_healthy"] == "1" else r["class"])
            lda_label.append(f"{ds}:{r['class']}")
    return np.vstack(X), np.array(crop), np.array(key), np.array(lda_label)


def fit_projection(X, lda_label, mask, kind):
    Xm = X[mask]
    mean = Xm.mean(axis=0)
    if kind == "lda":
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Xm, lda_label[mask])
        W = lda.scalings_[:, :DIM]
    else:
        W = PCA(n_components=DIM, random_state=0).fit(Xm).components_.T
    return mean.astype(np.float64), W.astype(np.float64)


def groups_from(Z, crop, key, only_crop=None):
    samples = {}
    for c in (sorted(set(crop) - {""}) if only_crop is None else [only_crop]):
        for k in set(key[crop == c]):
            arr = Z[(crop == c) & (key == k)]
            if len(arr) >= w16.MIN_TRAIN:
                samples[(c, k)] = arr
    return samples


def fisher_mean(Z, models):
    """Average Fisher information of the Gaussian-class posterior with respect to the features (ZFGA-style)."""
    K, d = len(models), Z.shape[1]
    G, L = [], []
    for m, C in models:
        Ci = np.linalg.inv(C)
        G.append(-(Z - m) @ Ci.T)
        L.append(w17.gauss_loglik(Z, m, C))
    G, L = np.stack(G, axis=1), np.stack(L, axis=1)               # n x K x d, n x K
    L = L - L.max(axis=1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    Gc = G - np.einsum("nk,nkd->nd", P, G)[:, None, :]
    return np.einsum("nk,nkd,nke->de", P, Gc, Gc) / len(Z)


def msqrt(M, power):
    w, v = np.linalg.eigh((M + M.T) / 2)
    return (v * np.maximum(w, 1e-12) ** power) @ v.T


def run_cell(op, k, samples_c, test_c, probe_c, hn, rep_seed, methods, probes):
    r2 = np.random.default_rng(rep_seed)
    classes = sorted(d for (c, d) in samples_c if d != "healthy")
    if len(classes) < 2 or "healthy" not in {d for (_, d) in samples_c}:
        return None
    crop = next(iter(samples_c))[0]
    th = test_c[(crop, "healthy")]
    H_tgt = op(th[r2.choice(len(th), size=hn, replace=True)], k)
    cls_data = {d: samples_c[(crop, d)] for d in classes}
    Xt, yt = [], []
    for ci, d in enumerate(classes):
        arr = test_c[(crop, d)]
        Xt.append(op(arr[r2.choice(len(arr), size=DISEASE_N, replace=True)], k))
        yt += [ci] * DISEASE_N
    Xt, yt = np.vstack(Xt), np.array(yt)
    src_h = samples_c[(crop, "healthy")]
    mA, CA = src_h.mean(axis=0), w17.shrunk_cov(src_h)
    mB, CB = H_tgt.mean(axis=0), w17.shrunk_cov(H_tgt)
    out = {}
    for m in methods:
        if m in ("NA", "MC", "STD", "CORAL", "HATC", "S50"):
            models = m46.fit_models(w50.make_point_map(m, mA, CA, mB, CB), cls_data, classes)
            out[m] = float(np.mean(m46.scores(Xt, models).argmax(axis=1) == yt))
        elif m in ("ZFGA", "ZFGA_MC"):
            models = m46.fit_models(lambda X: X, cls_data, classes)
            Iref, Ite = fisher_mean(src_h, models), fisher_mean(H_tgt, models)
            eps = 1e-3 * np.trace(Iref) / Iref.shape[0]
            A = msqrt(Ite + eps * np.eye(len(Ite)), -0.5) @ msqrt(Iref + eps * np.eye(len(Iref)), 0.5)
            Xz = Xt @ A.T if m == "ZFGA" else mA + (Xt - mB) @ A.T
            out[m] = float(np.mean(m46.scores(Xz, models).argmax(axis=1) == yt))
    if probes:
        r3 = np.random.default_rng(rep_seed + 7000)
        for n_lab in probes:
            Xtr, ytr = [], []
            for ci, d in enumerate(classes):
                arr = probe_c[(crop, d)]
                Xtr.append(op(arr[r3.choice(len(arr), size=n_lab, replace=True)], k))
                ytr += [ci] * n_lab
            clf = LogisticRegression(max_iter=300).fit(np.vstack(Xtr), np.array(ytr))
            out[f"probe{n_lab}"] = float(np.mean(clf.predict(Xt) == yt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fit", "calibrate", "run"], required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--base-projection", default="results/08_projection/projection_k16.npz")
    ap.add_argument("--output", required=True)
    ap.add_argument("--projections", nargs="+", default=["base", "loco_lda", "loco_pca"])
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--scales", nargs="+", type=float, default=[1, 1.5, 2, 3, 4])
    ap.add_argument("--healthy-n", type=int, default=200)
    ap.add_argument("--reps-per-crop", type=int, default=17)
    ap.add_argument("--worlds", nargs="+", default=WORLDS)
    ap.add_argument("--world-seed", type=int, default=2026)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    feats, out = Path(args.features), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1: loading raw training-split features")
    X, crop, key, lda_label = load_raw(feats, args.labelled)
    log(f"  {X.shape[0]} rows, {len(set(crop) - {''})} crops")
    P0 = np.load(args.base_projection)
    base = (P0["mean"].astype(np.float64), P0["W"].astype(np.float64))
    Zb = (X - base[0]) @ base[1]
    groups_b = w16.harmonise_samples(groups_from(Zb, crop, key))
    # usable target crops: a healthy group and at least two disease groups after harmonisation
    crops_all = sorted({c for (c, k) in groups_b if k == "healthy"})
    usable = [c for c in crops_all if sum(1 for (cc, k) in groups_b if cc == c and k != "healthy") >= 2]
    log(f"  crops with healthy images {len(crops_all)}; usable targets {len(usable)}: {usable}")

    if args.mode == "fit":
        for c in usable:
            for kind in ("pca", "lda"):
                p = out / f"loco_{kind}_{c}.npz"
                if p.exists():
                    continue
                t1 = time.time()
                mean, W = fit_projection(X, lda_label, crop != c, kind)
                np.savez(p, mean=mean, W=W)
                log(f"  fitted {kind} without crop {c} ({time.time() - t1:.0f} s)")
        log("done")
        return

    def projection_of(name, c):
        if name == "base":
            return base
        p = np.load(out.parent / "loco" / f"{name}_{c}.npz") if (out.parent / "loco").exists() else np.load(out / f"{name}_{c}.npz")
        return p["mean"], p["W"]

    scales = args.scales if args.mode == "calibrate" else [args.scale]
    methods = ["NA"] if args.mode == "calibrate" else ["NA", "MC", "STD", "CORAL", "HATC", "S50", "ZFGA", "ZFGA_MC"]
    probes = [] if args.mode == "calibrate" else [5, 10, 25]
    rows_out = []
    for pname in args.projections:
        log(f"step 2: projection {pname}")
        per_crop = {}
        for c in usable:
            mean, W = projection_of(pname, c)
            Zc = (X - mean) @ W
            samples_all = w16.harmonise_samples(groups_from(Zc, crop, key))
            mu = np.vstack(list(samples_all.values())).mean(axis=0)
            src_s, test_s, probe_s = w16.split_samples(samples_all)
            sel = lambda D: {k: v for k, v in D.items() if k[0] == c}
            per_crop[c] = (mu, sel(src_s), sel(test_s), sel(probe_s))
        for scale in scales:
            for wname in args.worlds:
                accs = {m: [] for m in methods + [f"probe{p}" for p in probes]}
                for ci, c in enumerate(usable):
                    mu, src_c, test_c, probe_c = per_crop[c]
                    rng = np.random.default_rng(args.world_seed)
                    op, _ = w16.operators(wname, rng, mu, len(crops_all), s=scale)
                    k = crops_all.index(c)
                    for rep in range(args.reps_per_crop):
                        res = run_cell(op, k, src_c, test_c, probe_c, args.healthy_n, args.world_seed + 100 + ci * 1000 + rep, methods, probes)
                        if res:
                            for m, v in res.items():
                                accs[m].append(v)
                row = {"projection": pname, "scale": scale, "world": wname, "healthy_n": args.healthy_n,
                       "replicates": len(accs["NA"])}
                for m, v in accs.items():
                    row[f"acc_{m}"] = round(float(np.mean(v)), 4) if v else ""
                if args.mode == "run":
                    best = max(row["acc_NA"], row["acc_MC"], row["acc_STD"])
                    row["HATC_minus_best_simple"] = round(row["acc_HATC"] - best, 4)
                    row["probe5_minus_HATC"] = round(row["acc_probe5"] - row["acc_HATC"], 4)
                rows_out.append(row)
            log(f"  {pname} scale {scale} done ({time.time() - t0:.0f} s)")
    path = out / f"{args.mode}_{args.tag}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (out / f"{args.mode}_{args.tag}_manifest.json").write_text(json.dumps({
        "module": "51_loco_projection", "mode": args.mode, "projections": args.projections, "scales": scales,
        "healthy_n": args.healthy_n, "reps_per_crop": args.reps_per_crop, "usable_crops": usable,
        "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s, wrote {path}")


if __name__ == "__main__":
    main()
