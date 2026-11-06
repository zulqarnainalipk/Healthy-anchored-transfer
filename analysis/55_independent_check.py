"""Module 55: independent re-implementation check of the real-pair test (module 53).

Re-implements, from scratch and without importing the benchmark code, the covariance shrinkage, the Bures-Wasserstein map, the
alignments NA, MC, STD, CORAL, HATC, the Gaussian classification (Cholesky), a leave-one-crop-out PCA (exact SVD) and the
data splits, for a random sample of units, and compares the balanced accuracy of every method with units.csv of module 53
(means over the same 20 replicate seeds). Differences should be of the order of the difference between the randomised PCA of
module 53 and the exact SVD used here.

Outputs: results/55_independent_check/comparison.csv and a summary line. CPU module, a few minutes.
Usage: python 55_independent_check.py --features <plantwild features dir> --units results/53_real_pair_test/pw_pca/units.csv
       --output results/55_independent_check --sample 15
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

SHRINK, DIM, MAP_MAX, PROBE_MAX = 0.1, 16, 200, 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def shrink(Z):
    C = np.cov(Z, rowvar=False)
    d = C.shape[0]
    return (1 - SHRINK) * C + SHRINK * np.trace(C) / d * np.eye(d)


def psd_pow(C, p):
    w, v = np.linalg.eigh((C + C.T) / 2)
    return (v * np.maximum(w, 1e-12) ** p) @ v.T


def bw_map(CA, CB):
    a_half, a_mhalf = psd_pow(CA, 0.5), psd_pow(CA, -0.5)
    return a_mhalf @ psd_pow(a_half @ CB @ a_half, 0.5) @ a_mhalf


def apply_map(name, X, mA, CA, mB, CB):
    if name == "NA":
        return X
    if name == "MC":
        return X + (mB - mA)
    if name == "STD":
        return mB + (X - mA) * np.sqrt(np.diag(CB) / np.diag(CA))
    if name == "CORAL":
        return mB + (X - mA) @ (psd_pow(CB, 0.5) @ psd_pow(CA, -0.5)).T
    if name == "HATC":
        return mB + (X - mA) @ bw_map(CA, CB).T
    raise ValueError(name)


def loglik(X, m, C):
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, (X - m).T)
    return -0.5 * ((z ** 2).sum(axis=0) + 2 * np.log(np.diag(L)).sum())


def balanced(pred, y, k):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(k)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = Path(args.features)
    X = np.load(feats / "features_plantwild.npy").astype(np.float64)
    meta = list(csv.DictReader(open(feats / "features_index.csv", encoding="utf-8")))
    crop = np.array([m["crop"] for m in meta])
    role = np.array([m["role"] for m in meta])
    disease = np.array([m["disease"] for m in meta])
    units = list(csv.DictReader(open(args.units, encoding="utf-8")))
    all_units = [(i + 1, u) for i, u in enumerate(units)]                  # module 53 seeds by 1-based unit position
    rng = np.random.default_rng(args.seed)
    pick = [all_units[i] for i in sorted(rng.choice(len(all_units), size=min(args.sample, len(all_units)), replace=False))]
    log(f"checking {len(pick)} of {len(units)} units")
    proj = {}
    rows = []
    for ui, u in pick:
        s, t, d = u["source"], u["target"], u["disease"]
        if t not in proj:
            keep = crop != t                                              # exact leave-one-crop-out PCA by SVD
            mean = X[keep].mean(axis=0)
            _, _, Vt = np.linalg.svd(X[keep] - mean, full_matrices=False)
            proj[t] = (X - mean) @ Vt[:DIM].T
        Z = proj[t]
        isd = lambda c: (crop == c) & (role == "disease") & np.array([(x == d or x.endswith(" " + d)) for x in disease])
        Zs_h, Zs_d = Z[(crop == s) & (role == "healthy")], Z[isd(s)]
        Zt_h, Zt_d = Z[(crop == t) & (role == "healthy")], Z[isd(t)]
        acc = {m: [] for m in ["NA", "MC", "STD", "CORAL", "HATC"]}
        for rep in range(args.reps):
            r = np.random.default_rng(1000 * ui + rep)                    # same split procedure as module 53
            perm = r.permutation(len(Zt_h))
            n_map = min(MAP_MAX, len(Zt_h) // 2)
            map_idx, rest_h = perm[:n_map], perm[n_map:]
            n_lab = min(PROBE_MAX, len(rest_h) // 3, len(Zt_d) // 3)
            test_h = rest_h[n_lab:]
            p = r.permutation(len(Zt_d))
            test_d = p[n_lab:]
            H = Zt_h[map_idx]
            mA, CA, mB, CB = Zs_h.mean(axis=0), shrink(Zs_h), H.mean(axis=0), shrink(H)
            Xte = np.vstack([Zt_h[test_h], Zt_d[test_d]])
            yte = np.concatenate([np.zeros(len(test_h), int), np.ones(len(test_d), int)])
            for m in acc:
                Y = apply_map(m, Zs_d, mA, CA, mB, CB)
                L = np.stack([loglik(Xte, mB, CB), loglik(Xte, Y.mean(axis=0), shrink(Y))], axis=1)
                acc[m].append(balanced(L.argmax(axis=1), yte, 2))
        for m, v in acc.items():
            rows.append({"unit": ui, "source": s, "target": t, "disease": d, "method": m,
                         "independent": round(float(np.mean(v)), 4), "module53": float(u[f"ba_{m}"]),
                         "difference": round(float(np.mean(v)) - float(u[f"ba_{m}"]), 4)})
        log(f"  unit {ui} ({s} -> {t}, {d}) done")
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    diff = np.array([abs(r["difference"]) for r in rows])
    log(f"max abs difference {diff.max():.4f}, mean abs difference {diff.mean():.4f}, share within 0.01: {float(np.mean(diff <= 0.01)):.3f}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
