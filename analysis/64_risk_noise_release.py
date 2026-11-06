"""Module 64: label-free risk proxies, a label-noise proxy and the replication lists (registration section 10).

(a) Risk proxies (DINOv2, tier-2 units of module 53): three quantities that need no disease labels of the target crop are
    computed per unit as the mean over 3 replicates with the splits of module 61: the Bures-Wasserstein distance between the
    source and target healthy sets (total, from module 61 logic), the disagreement between the predictions of the partial maps
    S25 and S75 on the unlabelled test images, and the false-alarm rate of HATC on held-out healthy target images (a healthy-only
    calibration error, since every held-out healthy image should be called healthy). Each is correlated (Spearman) with the HATC
    balanced accuracy and with HATC minus mean-centring over the units, with the range over leave-one-target-crop-out.
    A gate that refuses the third of units with the worst proxy is evaluated: mean HATC balanced accuracy on the kept units.
(b) Label-noise proxy: in each crop, every image is compared with its 10 nearest neighbours (cosine, DINOv2, same crop,
    leave-self-out); the share of neighbours with the other role (healthy versus disease) is averaged by disease type. High values
    point to images that are hard or possibly mislabelled; this is a proxy, not an annotator study.
(c) Replication lists: kept images with the harmonised crop, role, disease and class (the index of the feature files), the
    removed near-duplicates, the class-name table, and the leave-one-crop-out projection (mean and 16 PCA axes per held-out crop)
    for both encoders.

Usage: python 64_risk_noise_release.py --features-dino ... --features-bio ... --results results/53_real_pair_test --output results/64_risk_noise_release
"""

import argparse
import csv
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m53", HERE.parent / "features" / "53_real_pair_test.py")
m53 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m53)
spec61 = importlib.util.spec_from_file_location("m61", HERE / "61_calibration_decision_rule.py")
m61 = importlib.util.module_from_spec(spec61)
spec61.loader.exec_module(m61)
w17, w50 = m53.w17, m53.w50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def proxies(Zs_h, Zs_d, Zt_h, Zt_d, seed):
    r = np.random.default_rng(seed)
    perm = r.permutation(len(Zt_h))
    n_map = min(m53.MAP_MAX, len(Zt_h) // 2)
    map_idx, rest_h = perm[:n_map], perm[n_map:]
    n_lab = min(max(m53.PROBES), len(rest_h) // 3, len(Zt_d) // 3)
    test_h = rest_h[n_lab:]
    p = r.permutation(len(Zt_d))
    test_d = p[n_lab:]
    H = Zt_h[map_idx]
    mA, CA, mB, CB = Zs_h.mean(axis=0), w17.shrunk_cov(Zs_h), H.mean(axis=0), w17.shrunk_cov(H)
    Xte = np.vstack([Zt_h[test_h], Zt_d[test_d]])
    pred = {}
    for m in ("S25", "S75", "HATC"):
        Y = w50.make_point_map(m, mA, CA, mB, CB)(Zs_d)
        L = np.stack([w17.gauss_loglik(Xte, mB, CB), w17.gauss_loglik(Xte, Y.mean(axis=0), w17.shrunk_cov(Y))], axis=1)
        pred[m] = L.argmax(axis=1)
    nh = len(test_h)
    return float(np.mean(pred["S25"] != pred["S75"])), float(np.mean(pred["HATC"][:nh] == 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dino", required=True)
    ap.add_argument("--features-bio", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--audit", default="results/plantwild_audit")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fd = Path(args.features_dino)
    X, meta = m53.load_plantwild(fd, "features_plantwild.npy")
    crop = np.array([m["crop"] for m in meta]); role = np.array([m["role"] for m in meta]); disease = np.array([m["disease"] for m in meta])

    log("step 1/3: risk proxies")
    units = [r for r in csv.DictReader(open(Path(args.results) / "pw_pca" / "units.csv", encoding="utf-8")) if r["tier"] == "2"]
    cache, rows = {}, []
    for ui, u in enumerate(units, 1):
        s, t, dt = u["source"], u["target"], u["disease"]
        if t not in cache:
            mean, W = m53.fit_projection(X, meta, t, "pca")
            cache[t] = (X - mean) @ W
        Zs_h, Zs_d, Zt_h, Zt_d = m61.unit_arrays(cache[t], crop, role, disease, s, t, dt)
        mA, CA, mB, CB = Zs_h.mean(axis=0), w17.shrunk_cov(Zs_h), Zt_h.mean(axis=0), w17.shrunk_cov(Zt_h)
        dm, dc = m61.bw_parts(mA, CA, mB, CB)
        v = np.array([proxies(Zs_h, Zs_d, Zt_h, Zt_d, 1000 * ui + k) for k in range(args.reps)]).mean(axis=0)
        rows.append({"source": s, "target": t, "disease": dt, "bw_total": round(float(np.sqrt((dm + dc) / np.trace(CB))), 4),
                     "s25_s75_disagreement": round(float(v[0]), 4), "healthy_false_alarm": round(float(v[1]), 4),
                     "ba_HATC": float(u["ba_HATC"]), "gain_HATC_minus_MC": round(float(u["ba_HATC"]) - float(u["ba_MC"]), 4)})
        if ui % 20 == 0:
            log(f"    {ui}/{len(units)} units")
    write(out / "risk_proxy_units.csv", rows)
    corr = []
    for feat in ("bw_total", "s25_s75_disagreement", "healthy_false_alarm"):
        x = np.array([r[feat] for r in rows])
        for tgt in ("ba_HATC", "gain_HATC_minus_MC"):
            y = np.array([r[tgt] for r in rows])
            loo = [spearmanr(np.array([r[feat] for r in rows if r["target"] != t]), np.array([r[tgt] for r in rows if r["target"] != t]))[0]
                   for t in sorted({r["target"] for r in rows})]
            k = x <= np.percentile(x, 66.7)             # refuse the third of units with the largest proxy value
            corr.append({"proxy": feat, "outcome": tgt, "units": len(rows), "spearman": round(float(spearmanr(x, y)[0]), 3),
                         "loo_min": round(float(min(loo)), 3), "loo_max": round(float(max(loo)), 3),
                         "mean_outcome_all": round(float(y.mean()), 4), "mean_outcome_kept_two_thirds": round(float(y[k].mean()), 4)})
            log(f"  {feat} vs {tgt}: rho {corr[-1]['spearman']}")
    write(out / "risk_proxy_correlations.csv", corr)

    log("step 2/3: label-noise proxy")
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    share = np.zeros(len(meta)); k = 10
    for c in sorted(set(crop)):
        idx = np.where(crop == c)[0]
        if len(idx) <= k:
            share[idx] = np.nan
            continue
        S = Xn[idx] @ Xn[idx].T
        np.fill_diagonal(S, -np.inf)
        nn = np.argsort(-S, axis=1)[:, :k]
        r_ = role[idx]
        share[idx] = (r_[nn] != r_[:, None]).mean(axis=1)
    rows = []
    for d in list(m53.TYPES) + list(m53.GENERIC):
        sel = (role == "disease") & np.array([(x == d or x.endswith(" " + d)) for x in disease])
        rows.append({"group": d, "images": int(sel.sum()), "mean_other_role_share": round(float(np.nanmean(share[sel])), 3),
                     "images_majority_other_role": round(float(np.nanmean(share[sel] > 0.5)), 3)})
    rows.append({"group": "healthy (all crops)", "images": int((role == "healthy").sum()), "mean_other_role_share": round(float(np.nanmean(share[role == "healthy"])), 3),
                 "images_majority_other_role": round(float(np.nanmean(share[role == "healthy"] > 0.5)), 3)})
    write(out / "label_noise_proxy.csv", rows)

    log("step 3/3: replication lists")
    rel = out / "replication"
    rel.mkdir(exist_ok=True)
    shutil.copy(Path(args.features_dino) / "features_index.csv", rel / "kept_images.csv")
    shutil.copy(Path(args.audit) / "dedupe_exclude.csv", rel / "removed_near_duplicates.csv")
    shutil.copy(Path(args.audit) / "class_table.csv", rel / "class_name_harmonisation.csv")
    for name, fdir, fname in (("dinov2", Path(args.features_dino), "features_plantwild.npy"), ("bioclip", Path(args.features_bio), "features_plantwild_bioclip.npy")):
        Xe, me = m53.load_plantwild(fdir, fname)
        arrs = {}
        for c in sorted(set(m["crop"] for m in me)):
            mean, W = m53.fit_projection(Xe, me, c, "pca")
            arrs[f"{c}__mean"] = mean.astype(np.float32); arrs[f"{c}__axes"] = W.astype(np.float32)
        np.savez_compressed(rel / f"loco_pca_projection_{name}.npz", **arrs)
        log(f"  {name}: {len(arrs) // 2} held-out crops")
    (out / "manifest.json").write_text(json.dumps({"module": "64_risk_noise_release", "reps": args.reps, "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log(f"done in {time.time() - t0:.0f} s")


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), restval="")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
