"""Module 54: agreement check on the real PlantWild pairs (registered in docs/registration.md, section 7).

For every target crop T and disease type D that at least three crops share (the target and at least two sources that pass the
count rule), and for every pair of source crops (S1, S2): both source disease sets are moved to T by the healthy-anchored map
(each through its own healthy images), a logistic discriminator is trained on half of the pushed samples and tested on the other
half, and the 95% Wilson lower bound of its accuracy q gives the bound TV >= 2q - 1 (as in module 18). Replicates: 20 splits.
The outcome of interest is whether the bound predicts failure of HATC on T:
  failure of a triple = the mean balanced accuracy of HATC over S1 and S2 is more than 0.02 below that of no adaptation
  (units from results/53_real_pair_test/<tag>/units.csv, same projection and splits).
Scores compared: the bound, ensemble disagreement (share of target test images on which the classifiers built from S1 and S2
differ) and low confidence (one minus the mean maximum posterior of the classifier built from S1). Metrics: AUROC for failure
(only meaningful with enough failures) and Spearman correlation with the HATC gain over NA (continuous).

Outputs (results/54_real_agreement/<tag>/): triples.csv, summary.json. CPU module, a few minutes.
Usage: python 54_real_agreement.py --features <plantwild features dir> --units results/53_real_pair_test/pw_pca/units.csv
       --projection pca --tag pca --output results/54_real_agreement
"""

import argparse
import csv
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m53 = _load("m53", "53_real_pair_test.py")
w17, w50 = m53.w17, m53.w50
REPS, PUSH_N = 20, 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wilson_lower(k, n, z=1.645):
    p = k / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--projection", choices=["pca", "lda"], default="pca")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading features and units")
    X, meta = m53.load_plantwild(Path(args.features))
    crop = np.array([m["crop"] for m in meta])
    role = np.array([m["role"] for m in meta])
    disease = np.array([m["disease"] for m in meta])
    units = {(u["source"], u["target"], u["disease"]): u for u in csv.DictReader(open(args.units, encoding="utf-8")) if u["tier"] in ("1", "2")}
    by_td = {}
    for (s, t, d) in units:
        by_td.setdefault((t, d), []).append(s)
    triples_src = [(t, d, s1, s2) for (t, d), ss in by_td.items() if len(ss) >= 2 for s1, s2 in itertools.combinations(sorted(ss), 2)]
    log(f"  {len(by_td)} (target, disease) cells, {len(triples_src)} source pairs")

    log("step 2/3: agreement bound, disagreement and confidence per triple")
    cache, rows = {}, []
    for ti, (t, d, s1, s2) in enumerate(triples_src, 1):
        if t not in cache:
            mean, W = m53.fit_projection(X, meta, t, args.projection)
            cache[t] = (X - mean) @ W
        Z = cache[t]
        isd = lambda c: Z[(crop == c) & (role == "disease") & np.array([(x == d or x.endswith(" " + d)) for x in disease])]
        Zt_h, Zt_d = Z[(crop == t) & (role == "healthy")], isd(t)
        S = {s: (Z[(crop == s) & (role == "healthy")], isd(s)) for s in (s1, s2)}
        bounds, dis, lowc = [], [], []
        for rep in range(REPS):
            r = np.random.default_rng(5000 * ti + rep)
            perm = r.permutation(len(Zt_h))
            n_map = min(m53.MAP_MAX, len(Zt_h) // 2)
            H, test_h = Zt_h[perm[:n_map]], Zt_h[perm[n_map:]]
            mB, CB = H.mean(axis=0), w17.shrunk_cov(H)
            pushed, models = {}, {}
            for s, (Zh, Zd) in S.items():
                mA, CA = Zh.mean(axis=0), w17.shrunk_cov(Zh)
                g = w50.make_point_map("HATC", mA, CA, mB, CB)
                Y = g(Zd)
                pushed[s] = Y[r.choice(len(Y), size=PUSH_N, replace=len(Y) < PUSH_N)]
                models[s] = [(mB, CB), (Y.mean(axis=0), w17.shrunk_cov(Y))]
            Xd = np.vstack([pushed[s1], pushed[s2]])
            yd = np.array([0] * PUSH_N + [1] * PUSH_N)
            idx = r.permutation(len(yd))
            tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
            clf = LogisticRegression(max_iter=300).fit(Xd[tr], yd[tr])
            q = float(np.mean(clf.predict(Xd[te]) == yd[te]))
            bounds.append(max(0.0, 2 * wilson_lower(int(round(q * len(te))), len(te)) - 1))
            Xte = np.vstack([test_h, Zt_d])
            Ls = {s: np.stack([w17.gauss_loglik(Xte, m, C) for m, C in models[s]], axis=1) for s in (s1, s2)}
            p1, p2 = Ls[s1].argmax(axis=1), Ls[s2].argmax(axis=1)
            dis.append(float(np.mean(p1 != p2)))
            P = np.exp(Ls[s1] - Ls[s1].max(axis=1, keepdims=True))
            lowc.append(float(1 - np.mean((P / P.sum(axis=1, keepdims=True)).max(axis=1))))
        u1, u2 = units[(s1, t, d)], units[(s2, t, d)]
        hatc = (float(u1["ba_HATC"]) + float(u2["ba_HATC"])) / 2
        na = (float(u1["ba_NA"]) + float(u2["ba_NA"])) / 2
        rows.append({"target": t, "disease": d, "source1": s1, "source2": s2, "bound": round(float(np.mean(bounds)), 4),
                     "ens_disagree": round(float(np.mean(dis)), 4), "low_conf": round(float(np.mean(lowc)), 4),
                     "hatc_ba": round(hatc, 4), "na_ba": round(na, 4), "gain_over_na": round(hatc - na, 4),
                     "failure": int(hatc < na - 0.02)})
        if ti % 25 == 0 or ti == len(triples_src):
            log(f"    {ti}/{len(triples_src)} source pairs")
    with open(out / "triples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log("step 3/3: scores against failure")
    fail = np.array([r["failure"] for r in rows])
    gain = np.array([r["gain_over_na"] for r in rows])
    summ = {"triples": len(rows), "failures": int(fail.sum()), "tag": args.tag}
    for sc in ("bound", "ens_disagree", "low_conf"):
        v = np.array([r[sc] for r in rows])
        summ[f"auroc_{sc}"] = round(float(roc_auc_score(fail, v)), 3) if 0 < fail.sum() < len(fail) else None
        rho = spearmanr(v, gain)
        summ[f"spearman_with_gain_{sc}"] = round(float(rho.statistic), 3)
        summ[f"spearman_p_{sc}"] = round(float(rho.pvalue), 4)
    summ["mean_bound"] = round(float(np.mean([r["bound"] for r in rows])), 3)
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    log(f"summary {summ}")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
