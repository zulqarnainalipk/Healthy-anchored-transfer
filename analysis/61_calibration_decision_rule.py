"""Module 61: (a) calibration of the healthy-versus-disease decision after alignment and (b) a healthy-only predictor of when alignment helps.

(a) For every tier-2 unit of the real-pair test (module 53) and 5 replicates per unit (same splits as module 53), the posterior of
'disease' under equal priors is computed for no adaptation (NA), mean-centring (MC), CORAL and HATC. We report the expected
calibration error (ECE, 10 equal-width bins), the Brier score, the balanced accuracy at the 0.5 threshold and the AUROC. Then a
Platt recalibration (one logistic regression on the log-likelihood ratio, fitted on 10 labelled target images per class that are
not in the map set or the test set) is applied and the same numbers are computed again. The recalibration uses target labels; it
is a diagnostic of whether the ranking of the methods comes from the decision threshold and not from the ordering of the scores.

(b) For every tier-2 unit, the size of the healthy-set shift is measured from healthy images only: the Bures-Wasserstein distance
between the shrunk Gaussians of the source and target healthy images in the 16-dimensional leave-one-crop-out projection, split
into a mean part and a covariance part, and scaled by the trace of the target covariance. The Spearman correlation of each
measure with the per-unit gain of HATC over no adaptation and over mean-centring (from module 53 results) is reported for DINOv2
and BioCLIP, with a leave-one-target-crop-out check of the sign. This is exploratory: it asks for a simple decision rule.

Usage:
    python 61_calibration_decision_rule.py --features-dino results/09_features_plantwild/remote/features_plantwild \
        --features-bio results/10_features_plantwild_bioclip/remote/features_plantwild_bioclip --results results/53_real_pair_test --output results/61_calibration_decision_rule
CPU module, a few minutes.
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m53", HERE.parent / "features" / "53_real_pair_test.py")
m53 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m53)
w17, w50 = m53.w17, m53.w50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def bw_parts(mA, CA, mB, CB):
    sA = np.real(sqrtm(CA))
    cross = np.real(sqrtm(sA @ CB @ sA))
    cov = float(np.trace(CA) + np.trace(CB) - 2 * np.trace(cross))
    return float(np.sum((mA - mB) ** 2)), max(cov, 0.0)


def unit_arrays(Z, crop, role, disease, s, t, dtype):
    sel = lambda c: (crop == c) & (role == "disease") & np.array([(x == dtype or x.endswith(" " + dtype)) for x in disease])
    return Z[(crop == s) & (role == "healthy")], Z[sel(s)], Z[(crop == t) & (role == "healthy")], Z[sel(t)]


def posteriors(Zs_h, Zs_d, Zt_h, Zt_d, seed, k_lab=10):
    """Mirror of module 53 splits; returns, per method, (llr on test, y test, llr on the labelled pool, y labelled)."""
    r = np.random.default_rng(seed)
    perm = r.permutation(len(Zt_h))
    n_map = min(m53.MAP_MAX, len(Zt_h) // 2)
    map_idx, rest_h = perm[:n_map], perm[n_map:]
    n_lab = min(max(m53.PROBES), len(rest_h) // 3, len(Zt_d) // 3)
    lab_h, test_h = rest_h[:n_lab], rest_h[n_lab:]
    p = r.permutation(len(Zt_d))
    lab_d, test_d = p[:n_lab], p[n_lab:]
    H = Zt_h[map_idx]
    mA, CA, mB, CB = Zs_h.mean(axis=0), w17.shrunk_cov(Zs_h), H.mean(axis=0), w17.shrunk_cov(H)
    Xte = np.vstack([Zt_h[test_h], Zt_d[test_d]])
    yte = np.concatenate([np.zeros(len(test_h)), np.ones(len(test_d))])
    Xlb = np.vstack([Zt_h[lab_h[:k_lab]], Zt_d[lab_d[:k_lab]]])
    ylb = np.concatenate([np.zeros(k_lab), np.ones(k_lab)])
    out = {}
    for m in ("NA", "MC", "CORAL", "HATC"):
        g = w50.make_point_map(m, mA, CA, mB, CB)
        Y = g(Zs_d)
        md, Cd = Y.mean(axis=0), w17.shrunk_cov(Y)
        f = lambda X: w17.gauss_loglik(X, md, Cd) - w17.gauss_loglik(X, mB, CB)
        out[m] = (f(Xte), yte, f(Xlb), ylb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dino", required=True)
    ap.add_argument("--features-bio", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    enc = {"dinov2": (Path(args.features_dino), "features_plantwild.npy", "pw_pca"),
           "bioclip": (Path(args.features_bio), "features_plantwild_bioclip.npy", "pw_bioclip_pca")}
    cal_rows, rule_rows = [], []
    for ei, (name, (fdir, fname, tag)) in enumerate(enc.items(), 1):
        log(f"step {ei}/2: encoder {name}")
        X, meta = m53.load_plantwild(fdir, fname)
        crop = np.array([m["crop"] for m in meta]); role = np.array([m["role"] for m in meta]); disease = np.array([m["disease"] for m in meta])
        units = [r for r in csv.DictReader(open(Path(args.results) / tag / "units.csv", encoding="utf-8")) if r["tier"] == "2"]
        cache = {}
        agg = {}
        for ui, u in enumerate(units, 1):
            s, t, dt = u["source"], u["target"], u["disease"]
            if t not in cache:
                mean, W = m53.fit_projection(X, meta, t, "pca")
                cache[t] = (X - mean) @ W
            Zs_h, Zs_d, Zt_h, Zt_d = unit_arrays(cache[t], crop, role, disease, s, t, dt)
            mA, CA, mB, CB = Zs_h.mean(axis=0), w17.shrunk_cov(Zs_h), Zt_h.mean(axis=0), w17.shrunk_cov(Zt_h)
            dm, dc = bw_parts(mA, CA, mB, CB)
            sc = float(np.trace(CB))
            rule_rows.append({"encoder": name, "source": s, "target": t, "disease": dt, "mean_part": round(np.sqrt(dm / sc), 4),
                              "cov_part": round(np.sqrt(dc / sc), 4), "bw_total": round(np.sqrt((dm + dc) / sc), 4),
                              "gain_HATC_minus_NA": round(float(u["ba_HATC"]) - float(u["ba_NA"]), 4),
                              "gain_HATC_minus_MC": round(float(u["ba_HATC"]) - float(u["ba_MC"]), 4), "ba_NA": u["ba_NA"]})
            if name != "dinov2":
                continue
            for rep in range(args.reps):
                res = posteriors(Zs_h, Zs_d[0] if isinstance(Zs_d, list) else Zs_d, Zt_h, Zt_d, 1000 * ui + rep)
                for m, (llr, y, llr_l, y_l) in res.items():
                    pr = 1 / (1 + np.exp(-np.clip(llr, -50, 50)))
                    plat = LogisticRegression(C=1e6, max_iter=500).fit(llr_l.reshape(-1, 1).clip(-50, 50), y_l)
                    pp = plat.predict_proba(llr.reshape(-1, 1).clip(-50, 50))[:, 1]
                    for lab, q in (("raw", pr), ("platt", pp)):
                        ba = float(np.mean([np.mean((q[y == c] > 0.5) == c) for c in (0, 1)]))
                        agg.setdefault((m, lab), []).append((ece(q, y), float(np.mean((q - y) ** 2)), ba, float(roc_auc_score(y, q))))
            if ui % 20 == 0:
                log(f"    {ui}/{len(units)} units")
        if name == "dinov2":
            for (m, lab), v in sorted(agg.items()):
                a = np.array(v).mean(axis=0)
                cal_rows.append({"method": m, "scores": lab, "ece": round(a[0], 4), "brier": round(a[1], 4), "balanced_accuracy": round(a[2], 4), "auroc": round(a[3], 4)})
    with open(out / "calibration.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cal_rows[0])); w.writeheader(); w.writerows(cal_rows)
    with open(out / "decision_rule_units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rule_rows[0])); w.writeheader(); w.writerows(rule_rows)

    log("correlations")
    corr = []
    for name in enc:
        R = [r for r in rule_rows if r["encoder"] == name]
        for feat in ("mean_part", "cov_part", "bw_total", "ba_NA"):
            for gain in ("gain_HATC_minus_NA", "gain_HATC_minus_MC"):
                x = np.array([float(r[feat]) for r in R]); y = np.array([r[gain] for r in R])
                rho = spearmanr(x, y)[0]
                tg = sorted({r["target"] for r in R})
                loo = [spearmanr(np.array([float(r[feat]) for r in R if r["target"] != t]), np.array([r[gain] for r in R if r["target"] != t]))[0] for t in tg]
                corr.append({"encoder": name, "measure": feat, "gain": gain, "units": len(R), "spearman": round(float(rho), 3),
                             "leave_one_target_out_min": round(float(min(loo)), 3), "leave_one_target_out_max": round(float(max(loo)), 3)})
    with open(out / "decision_rule_correlations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(corr[0])); w.writeheader(); w.writerows(corr)
    (out / "manifest.json").write_text(json.dumps({"module": "61_calibration_decision_rule", "reps": args.reps, "platt_labels_per_class": 10,
                                                   "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
