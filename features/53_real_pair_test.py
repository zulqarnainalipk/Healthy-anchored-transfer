"""Module 53: real-pair test (PlantWild v1 and the PlantVillage tomato-potato pilot). Specified in docs/registration.md, section 7.

Unit = (source crop S, target crop T, disease type D). Task: healthy versus disease D on crop T (binary), or healthy versus
several diseases (three-class, tomato and potato). The healthy model is the Gaussian of the target healthy images used for the
map (no disease label); the disease model is the source disease Gaussian moved to the target by the method under test:
NA, MC, STD, CORAL, HATC (affine Monge map between the healthy Gaussians), interpolated maps S25, S50, S75. Labelled arm:
logistic probe on k labelled target images per class (k = 5, 10, 25). Projection: 16 dimensions fitted without the target
crop (leave-one-crop-out): PCA (primary) or LDA (secondary).

Per replicate (20 random splits per unit): the target healthy images are split into a map set (min(200, half)), a labelled
pool for the probe (up to 25 per class, taken from images not in the map set) and a test set (the rest); target disease
images of type D are split into the labelled pool and the test set (all remaining images).
Outputs (results/53_real_pair_test/<tag>/): units.csv (per unit: balanced accuracy and AUROC per method, mean over replicates),
primary.csv (HATC minus the best simple method with the two-way cluster bootstrap), manifest.json.

CPU module. Usage:
    python 53_real_pair_test.py --dataset plantwild --features <dir with features_plantwild.npy and features_index.csv>
        --projection pca --tag pca --output results/53_real_pair_test
    python 53_real_pair_test.py --dataset pv_pilot --features results/05_features_dinov2/remote/features_dinov2 \
        --labelled results/01_crop_disease_table/labelled_images.csv --projection pca --tag pilot --output results/53_real_pair_test
"""

import argparse
import csv
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w17 = _load("w17", "17_benchmark_hatc.py")
w50 = _load("w50", "50_hatc_select.py")

TYPES = ["rust", "mosaic virus", "powdery mildew", "downy mildew", "early blight", "late blight", "black rot", "anthracnose",
         "sheath blight", "alternaria leaf spot"]
GENERIC = ["leaf spot", "leaf blight"]
TIER1 = {"early blight": {"potato", "tomato"}, "late blight": {"potato", "tomato"}}
MIN_HEALTHY, MIN_SRC, MIN_TGT = 50, 30, 30
METHODS = ["NA", "MC", "STD", "CORAL", "HATC", "S25", "S50", "S75"]
PROBES = [5, 10, 25]
SELECT_ON = False   # --select: add HATC-S (alignment chosen from target healthy images only; post hoc)
SELECTED = defaultdict(int)   # how often each candidate alignment was chosen by HATC-S
FAILURES = []   # methods whose likelihoods were not finite in a replicate (counted per unit, reported)
REPS, DIM, MAP_MAX = 20, 16, 200
SIMPLE = ["NA", "MC", "STD"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_plantwild(feats, name="features_plantwild.npy"):
    X = np.load(feats / name).astype(np.float32)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return X, meta


def load_pilot(feats, labelled):
    """PlantVillage tomato and potato (healthy, early blight, late blight) plus all other PlantVillage rows for the projection."""
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = [r for r in csv.DictReader(f) if r["dataset"] == "plantvillage_color"]
    lab = {}
    with open(labelled, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "plantvillage_color" and r["role"] == "image":
                lab[r["path"]] = (r["crop"].strip().lower(), r["disease"].lower())
    A = np.load(feats / "features_plantvillage_color.npy").astype(np.float32)
    meta, keep = [], []
    for i, r in enumerate(index):
        if r["path"] not in lab:
            continue
        crop, dis = lab[r["path"]]
        if r["is_healthy"] == "1":
            role, d = "healthy", ""
        elif "early blight" in dis or "early_blight" in dis:
            role, d = "disease", "early blight"
        elif "late blight" in dis or "late_blight" in dis:
            role, d = "disease", "late blight"
        else:
            role, d = "other", dis
        meta.append({"crop": crop, "role": role, "disease": d, "class": f"{crop} {d}".strip()})
        keep.append(i)
    return A[keep], meta


def fit_projection(X, meta, target, kind):
    mask = np.array([m["crop"] != target for m in meta])
    mean = X[mask].mean(axis=0)
    if kind == "pca":
        W = PCA(n_components=DIM, random_state=0).fit(X[mask]).components_.T
    else:
        y = np.array([m["class"] for m in meta])[mask]
        W = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X[mask], y).scalings_[:, :DIM]
    return mean.astype(np.float64), W.astype(np.float64)


EXTRA_ON = False
EXTRA_METHODS = ["EIG", "UDA_CORAL", "UDA_BW"]


def eig_map(mA, CA, mB, CB):
    """Affine map with covariance C_B whose rotation comes from matching the eigenvectors of C_A and C_B in order of eigenvalue.
    A = V_B diag(sqrt(l_B / l_A)) S V_A^T with S = diag(sign(v_A,i . v_B,i)) (a Procrustes-like choice from healthy images only)."""
    wA, VA = np.linalg.eigh(CA)
    wB, VB = np.linalg.eigh(CB)
    VA, VB, wA, wB = VA[:, ::-1], VB[:, ::-1], wA[::-1], wB[::-1]
    S = np.sign(np.sum(VA * VB, axis=0))
    S[S == 0] = 1
    A = (VB * (np.sqrt(wB / wA) * S)) @ VA.T
    return lambda X: mB + (X - mA) @ A.T


def balanced_acc(pred, y, k):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(k)]))


def eval_unit(Zs_h, Zs_d, Zt_h, Zt_d, rep_seed):
    """Zs_h: source healthy; Zs_d: list of source disease arrays; Zt_h: target healthy; Zt_d: list of target disease arrays."""
    r = np.random.default_rng(rep_seed)
    k_cls = 1 + len(Zt_d)
    perm = r.permutation(len(Zt_h))
    n_map = min(MAP_MAX, len(Zt_h) // 2)
    map_idx, rest_h = perm[:n_map], perm[n_map:]
    n_lab = min(max(PROBES), len(rest_h) // 3, *[len(z) // 3 for z in Zt_d])
    lab_h, test_h = rest_h[:n_lab], rest_h[n_lab:]
    lab_d, test_d = [], []
    for z in Zt_d:
        p = r.permutation(len(z))
        lab_d.append(p[:n_lab]); test_d.append(p[n_lab:])
    H = Zt_h[map_idx]
    mA, CA = Zs_h.mean(axis=0), w17.shrunk_cov(Zs_h)
    mB, CB = H.mean(axis=0), w17.shrunk_cov(H)
    Xte = np.vstack([Zt_h[test_h]] + [z[i] for z, i in zip(Zt_d, test_d)])
    yte = np.concatenate([[0] * len(test_h)] + [[c + 1] * len(i) for c, i in enumerate(test_d)])
    out = {}
    for m in METHODS + (EXTRA_METHODS if EXTRA_ON else []):
        try:
            if m == "EIG":                                   # covariance-matching map with the rotation fixed by eigenvector matching (healthy images only)
                g = eig_map(mA, CA, mB, CB)
            elif m in ("UDA_CORAL", "UDA_BW"):               # reference arm with unlabelled target images of both kinds (transductive; uses more than healthy images)
                Ps, Pt = np.vstack([Zs_h] + list(Zs_d)), np.vstack([H, Xte])
                g = w50.make_point_map("CORAL" if m == "UDA_CORAL" else "HATC", Ps.mean(axis=0), w17.shrunk_cov(Ps), Pt.mean(axis=0), w17.shrunk_cov(Pt))
            else:
                g = w50.make_point_map(m, mA, CA, mB, CB)
            models = [(mB, CB)] + [(lambda Y: (Y.mean(axis=0), w17.shrunk_cov(Y)))(g(zs)) for zs in Zs_d]
            L = np.stack([w17.gauss_loglik(Xte, mm, CC) for mm, CC in models], axis=1)
        except Exception as e:                              # e.g. matrix square root fails on a degenerate covariance
            FAILURES.append(f"{m}:{type(e).__name__}")
            out[m] = (float("nan"), float("nan"))
            continue
        if not np.isfinite(L).all():                       # numerical failure of the map or of a Gaussian: recorded, not hidden
            FAILURES.append(m)
            out[m] = (float("nan"), float("nan"))
            continue
        pred = L.argmax(axis=1)
        out[m] = (balanced_acc(pred, yte, k_cls), float(roc_auc_score((yte > 0).astype(int), L[:, 1:].max(axis=1) - L[:, 0])))
        if m == "HATC":                                   # per-image refusal by confidence: AUROC for detecting errors, accuracy at 80% coverage
            P = np.exp(L - L.max(axis=1, keepdims=True))
            conf = (P / P.sum(axis=1, keepdims=True)).max(axis=1)
            err = (pred != yte).astype(int)
            if 0 < err.sum() < len(err):
                keep = np.argsort(-conf)[: int(0.8 * len(err))]
                out["conf"] = (float(roc_auc_score(err, -conf)), float(1 - err[keep].mean()))
    if SELECT_ON:                                        # HATC-S (module 50) on real pairs: choose the alignment from target healthy images only
        order = ["NA", "MC", "STD", "S25", "S50", "S75", "CORAL", "HATC"]    # simplest first
        half = len(H) // 2
        err = {}
        try:
            for cand in order:
                e = []
                for fit, held in ((H[:half], H[half:]), (H[half:], H[:half])):
                    g = w50.make_point_map(cand, mA, CA, fit.mean(axis=0), w17.shrunk_cov(fit))
                    mp = g(Zs_h)
                    e.append(w50.frechet(mp.mean(axis=0), w17.shrunk_cov(mp), held.mean(axis=0), w17.shrunk_cov(held)))
                err[cand] = float(np.mean(e))
            low = min(err.values())
            chosen = next(c for c in order if err[c] <= 1.05 * low)
            SELECTED[chosen] += 1
            out["SELECT"] = out[chosen]
        except Exception as e:
            FAILURES.append(f"SELECT:{type(e).__name__}")
            out["SELECT"] = (float("nan"), float("nan"))
    if k_cls == 2:                                       # one-class baseline: Mahalanobis distance to the target healthy Gaussian, no source disease used
        Ci = np.linalg.inv(CB)
        d2 = lambda Y: np.einsum("ij,jk,ik->i", Y - mB, Ci, Y - mB)
        thr = np.percentile(d2(H), 95)
        s_te = d2(Xte)
        out["OC"] = (balanced_acc((s_te > thr).astype(int), yte, 2), float(roc_auc_score((yte > 0).astype(int), s_te)))
    for kk in PROBES:
        if kk > n_lab:
            continue
        Xl = np.vstack([Zt_h[lab_h[:kk]]] + [z[i[:kk]] for z, i in zip(Zt_d, lab_d)])
        yl = np.concatenate([[0] * kk] + [[c + 1] * kk for c in range(len(Zt_d))])
        clf = LogisticRegression(max_iter=300).fit(Xl, yl)
        s = clf.predict_proba(Xte)
        out[f"probe{kk}"] = (balanced_acc(s.argmax(axis=1), yte, k_cls), float(roc_auc_score((yte > 0).astype(int), 1 - s[:, 0])))
    return out


def two_way_bootstrap(units, key, resamples=10000, seed=2026):
    """units: list of (target, source, value). Mean value with a two-way cluster bootstrap over targets and sources."""
    rng = np.random.default_rng(seed)
    tg = sorted({u[0] for u in units})
    sr = sorted({u[1] for u in units})
    ti = {t: i for i, t in enumerate(tg)}
    si = {s: i for i, s in enumerate(sr)}
    T = np.array([ti[u[0]] for u in units])
    S = np.array([si[u[1]] for u in units])
    v = np.array([u[2] for u in units], dtype=float)
    boots = np.empty(resamples)
    for b in range(resamples):
        wt = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))
        ws = np.bincount(rng.integers(0, len(sr), len(sr)), minlength=len(sr))
        w = wt[T] * ws[S]
        boots[b] = (w * v).sum() / w.sum() if w.sum() > 0 else np.nan
    boots = boots[~np.isnan(boots)]
    return float(v.mean()), float(np.percentile(boots, 5)), float(np.percentile(boots, 95)), len(tg), len(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["plantwild", "pv_pilot"], required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", default="")
    ap.add_argument("--features-name", default="features_plantwild.npy", help="feature file inside --features (BioCLIP: features_plantwild_bioclip.npy)")
    ap.add_argument("--projection", choices=["pca", "lda"], default="pca")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--tiers", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--multiclass", action="store_true", help="three-class tomato and potato task instead of binary units")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--map-max", type=int, default=200, help="largest number of target healthy images used for the map (200 registered; 50 sensitivity)")
    ap.add_argument("--domain-match", default="", help="path of url_record.json: keep only images whose source domain occurs in both the healthy and the disease class of a crop (confound control; needs at least 40 healthy and 25 disease images per crop after matching)")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to every replicate seed (multi-seed verification)")
    ap.add_argument("--control", choices=["none", "wrong_disease"], default="none",
                    help="wrong_disease: the source disease model is built from another disease of the source crop (negative control)")
    ap.add_argument("--exclude-file", default="", help="csv with a column 'path' of images to drop before the analysis (stricter duplicate rule)")
    ap.add_argument("--cov", choices=["shrink", "lw", "oas", "lowrank"], default="shrink",
                    help="covariance estimator for every Gaussian (post hoc sensitivity): shrink = fixed lambda towards the scaled identity "
                         "(registered, lambda 0.1); lw = Ledoit-Wolf; oas = oracle approximating shrinkage; lowrank = rank-r factor model plus isotropic noise")
    ap.add_argument("--lam", type=float, default=0.1, help="lambda of --cov shrink (registered 0.1)")
    ap.add_argument("--rank", type=int, default=4, help="rank of --cov lowrank")
    ap.add_argument("--select", action="store_true", help="add HATC-S, the alignment chosen from target healthy images only (module 50 rule); post hoc")
    ap.add_argument("--extra", action="store_true", help="add EIG (eigenvector-matched rotation) and the unlabelled-target reference arms UDA_CORAL and UDA_BW; post hoc (registration section 10)")
    args = ap.parse_args()
    globals()["EXTRA_ON"] = args.extra
    globals()["MAP_MAX"] = args.map_max
    globals()["SELECT_ON"] = args.select
    if args.cov != "shrink" or args.lam != 0.1:
        from sklearn.covariance import ledoit_wolf, oas as _oas
        lam_, rank_, kind_ = args.lam, args.rank, args.cov

        def alt_cov(Z):
            d = Z.shape[1]
            if kind_ == "lw":
                return ledoit_wolf(Z)[0]
            if kind_ == "oas":
                return _oas(Z)[0]
            C = np.cov(Z, rowvar=False)
            if kind_ == "lowrank":                           # top-r eigenpairs plus the mean of the remaining eigenvalues as isotropic noise
                w, V = np.linalg.eigh(C)
                w, V = w[::-1], V[:, ::-1]
                noise = max(float(w[rank_:].mean()), 1e-8)
                return (V[:, :rank_] * (w[:rank_] - noise)) @ V[:, :rank_].T + noise * np.eye(d)
            return (1 - lam_) * C + lam_ * np.trace(C) / d * np.eye(d)
        w17.shrunk_cov = alt_cov
        log(f"covariance estimator: {args.cov} (lambda {args.lam}, rank {args.rank})")
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/4: loading features")
    feats = Path(args.features)
    if args.dataset == "plantwild":
        X, meta = load_plantwild(feats, args.features_name)
    else:
        X, meta = load_pilot(feats, args.labelled)
    if args.exclude_file:
        drop = {ln.strip() for ln in Path(args.exclude_file).read_text(encoding="utf-8").splitlines()[1:]}
        keep = np.array([m.get("path", "") not in drop for m in meta])
        X, meta = X[keep], [m for m, k in zip(meta, keep) if k]
        log(f"  dropped {int((~keep).sum())} images listed in {args.exclude_file}")
    log(f"  {len(meta)} images")
    domain = None
    if args.domain_match:
        import json as _json
        from urllib.parse import urlparse
        urls = _json.loads(Path(args.domain_match).read_text(encoding="utf-8"))
        domain = np.array([urlparse(urls.get("plantwild/images/" + m["path"], "")).netloc.lower().replace("www.", "") for m in meta])
    crop = np.array([m["crop"] for m in meta])
    role = np.array([m["role"] for m in meta])
    disease = np.array([m["disease"] for m in meta])

    log("step 2/4: units")
    units = []
    if args.multiclass:
        for s, t in (("tomato", "potato"), ("potato", "tomato")):
            units.append((s, t, ["early blight", "late blight"], 1))
    else:
        for dtype in TYPES + GENERIC:
            have = sorted({c for c, d, r in zip(crop, disease, role) if r == "disease" and (d == dtype or d.endswith(" " + dtype))})
            for s in have:
                for t in have:
                    if s == t:
                        continue
                    tier = 1 if (dtype in TIER1 and {s, t} <= TIER1[dtype]) else (3 if dtype in GENERIC else 2)
                    if tier in args.tiers:
                        units.append((s, t, [dtype], tier))
    ok = []
    for s, t, ds, tier in units:
        sel = lambda c, dd: (crop == c) & (role == "disease") & np.array([(x == dd or x.endswith(" " + dd)) for x in disease])
        nh = int(((crop == t) & (role == "healthy")).sum())
        nhs = int(((crop == s) & (role == "healthy")).sum())      # the source crop needs healthy images too (they define the map)
        if s and t and nh >= MIN_HEALTHY and nhs >= MIN_HEALTHY and all(sel(s, d).sum() >= MIN_SRC and sel(t, d).sum() >= MIN_TGT for d in ds):
            ok.append((s, t, ds, tier))
    log(f"  {len(units)} candidate units, {len(ok)} pass the count rule")

    log("step 3/4: running units")
    cache, rows = {}, []
    for ui, (s, t, ds, tier) in enumerate(ok, 1):
        if t not in cache:
            mean, W = fit_projection(X, meta, t, args.projection)
            cache[t] = (X - mean) @ W
        Z = cache[t]
        sel = lambda c, dd: Z[(crop == c) & (role == "disease") & np.array([(x == dd or x.endswith(" " + dd)) for x in disease])]
        Zs_h, Zt_h = Z[(crop == s) & (role == "healthy")], Z[(crop == t) & (role == "healthy")]
        Zs_d, Zt_d = [sel(s, d) for d in ds], [sel(t, d) for d in ds]
        if domain is not None:                              # keep only images from domains shared by the healthy and the disease class of the crop
            isdis = lambda c, dd: (crop == c) & (role == "disease") & np.array([(x == dd or x.endswith(" " + dd)) for x in disease])
            def shared(c, dd):
                mh, md = (crop == c) & (role == "healthy"), isdis(c, dd)
                keep = set(domain[mh]) & set(domain[md]) - {""}
                return mh & np.isin(domain, list(keep)), md & np.isin(domain, list(keep))
            (mh_s, md_s), (mh_t, md_t) = shared(s, ds[0]), shared(t, ds[0])
            Zs_h, Zt_h, Zs_d, Zt_d = Z[mh_s], Z[mh_t], [Z[md_s]], [Z[md_t]]
            if len(Zs_h) < 40 or len(Zt_h) < 40 or len(Zs_d[0]) < 25 or len(Zt_d[0]) < 25:
                continue
        control_note = ""
        if args.control == "wrong_disease":
            others = sorted({d for c, d, r in zip(crop, disease, role) if c == s and r == "disease" and d != "" and
                             not any(d == x or d.endswith(" " + x) for x in ds)})
            pick = next((d for d in others if ((crop == s) & (disease == d)).sum() >= MIN_SRC), None)
            if pick is None:
                continue
            Zs_d = [Z[(crop == s) & (disease == pick)]]
            control_note = pick
        acc = defaultdict(lambda: ([], []))
        n_fail0 = len(FAILURES)
        for rep in range(args.reps):
            res = eval_unit(Zs_h, Zs_d, Zt_h, Zt_d, 1000 * ui + rep + args.seed_offset)
            for m, (b, a) in res.items():
                acc[m][0].append(b); acc[m][1].append(a)
        row = {"tier": tier, "source": s, "target": t, "disease": "+".join(ds), "target_healthy": len(Zt_h),
               "source_disease": sum(len(z) for z in Zs_d), "target_disease": sum(len(z) for z in Zt_d), "control_source_disease": control_note}
        for m, (b, a) in acc.items():
            row[f"ba_{m}"] = round(float(np.nanmean(b)), 4) if not np.all(np.isnan(b)) else float("nan")
            row[f"auroc_{m}"] = round(float(np.nanmean(a)), 4) if not np.all(np.isnan(a)) else float("nan")
            row[f"fail_{m}"] = int(np.sum(np.isnan(b)))
        rows.append(row)
        if ui % 10 == 0 or ui == len(ok):
            log(f"    {ui}/{len(ok)} units")
    with open(out / "units.csv", "w", newline="", encoding="utf-8") as f:
        names = list(dict.fromkeys(k for r in rows for k in r))      # union: a unit with few images may lack the larger probes
        w = csv.DictWriter(f, fieldnames=names, restval="")
        w.writeheader()
        w.writerows(rows)

    log("step 4/4: primary outcome and bootstrap")
    prim = []
    for tier in sorted({r["tier"] for r in rows}):
        sub = [r for r in rows if r["tier"] == tier]
        best = max(SIMPLE, key=lambda m: np.mean([r[f"ba_{m}"] for r in sub]))
        for name in ["HATC", "SELECT", "EIG", "UDA_CORAL", "UDA_BW", "CORAL", "S25", "S50", "S75", "OC", "probe5", "probe10", "probe25"]:
            if f"ba_{name}" not in sub[0]:
                continue
            vals = [(r["target"], r["source"], r[f"ba_{name}"] - r[f"ba_{best}"]) for r in sub if f"ba_{name}" in r and np.isfinite(r[f"ba_{name}"]) and np.isfinite(r[f"ba_{best}"])]
            if len(vals) < 2:
                continue
            mean, lo, hi, nt, ns = two_way_bootstrap(vals, name, resamples=10000 if len(vals) > 2 else 1000)
            outcome = "equivalent" if (lo > -0.02 and hi < 0.02) else ("different" if (hi < -0.02 or lo > 0.02) else "inconclusive")
            prim.append({"tier": tier, "contrast": f"{name} minus {best}", "units": len(vals), "target_crops": nt, "source_crops": ns,
                         "mean_diff": round(mean, 4), "ci90_low": round(lo, 4), "ci90_high": round(hi, 4), "outcome": outcome,
                         "status": "ok" if nt >= 8 and ns >= 8 else "exploratory (fewer than 8 clusters)"})
    if prim:
        with open(out / "primary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prim[0].keys()))
            w.writeheader()
            w.writerows(prim)
    (out / "manifest.json").write_text(json.dumps({"module": "53_real_pair_test", "dataset": args.dataset, "projection": args.projection,
                                                    "units": len(rows), "reps": args.reps, "seed_offset": args.seed_offset, "control": args.control, "cov": args.cov, "lam": args.lam, "rank": args.rank, "exclude_file": args.exclude_file, "map_max": MAP_MAX, "seconds": round(time.time() - t0, 1)}, indent=2),
                                       encoding="utf-8")
    from collections import Counter
    if SELECT_ON:
        (out / "select_counts.json").write_text(json.dumps(dict(SELECTED), indent=2), encoding="utf-8")
        log(f"HATC-S choices: {dict(SELECTED)}")
    log(f"numerical failures (method: replicates with non-finite likelihoods): {dict(Counter(FAILURES))}")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
