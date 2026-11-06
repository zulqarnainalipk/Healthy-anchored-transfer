"""Module 48b: redesigned data panels for the pipeline figure (panels c, d, e), all from real stored features.

Writes to figures/fig4_pipeline/assets/:
  space_source.png, space_target.png  framed 2-D views (axis 1 = healthy shift, axis 2 = main spread of the tomato
                                      disease means) with filled 2-sigma ellipses; pushed tomato models in the target
  map_T.png                           heat map of the 16 x 16 affine map T (tomato healthy to fig healthy)
  posterior.json                      class probabilities of one real fig anthracnose test image under the three pushed
                                      tomato disease models (equal priors, shrunk-covariance Gaussians)
  disc_hist.png                       agreement check on the real pair: score distributions of a logistic discriminator
                                      separating early blight of tomato and of potato, both pushed to fig, with
                                      accuracy a and the bound TV >= 2a - 1 (Wilson lower bound on a)
CPU module, runs locally in a few seconds. Illustrative pair (tomato to fig).
Usage: python analysis/48b_pipeline_panels_v2.py --results results --assets figures/fig4_pipeline/assets
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("w17", HERE.parent / "features" / "17_benchmark_hatc.py")
w17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w17)

TEAL = {"healthy": "#00634A", "early": "#009E73", "septoria": "#4DC4A1", "mold": "#99DCC6"}
AMBER, NAVY, GREY = "#E69F00", "#0072B2", "#8A97A8"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ell(ax, pts, color, lw=1.6, ls="-", fill=0.10):
    m, C = pts.mean(axis=0), np.cov(pts.T)
    w, v = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(m, 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, fill=fill > 0, fc=color,
                         ec=color, lw=lw, ls=ls, zorder=2))
    if fill > 0:
        ax.patches[-1].set_facecolor((*matplotlib.colors.to_rgb(color), fill))


def ell_pts(pts):
    m, C = pts.mean(axis=0), np.cov(pts.T)
    w, v = np.linalg.eigh(C)
    th = np.linspace(0, 2 * np.pi, 60)
    return m + (v @ (2 * np.sqrt(w)[:, None] * np.vstack([np.cos(th), np.sin(th)]))).T


def frame(ax, sets):
    pts = np.vstack(sets)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.07 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color="#E3E8EF", lw=0.5)
    for s in ax.spines.values():
        s.set_color("#AEB8C6")
        s.set_linewidth(0.8)
    ax.set_xlabel("axis 1: healthy shift", fontsize=7, color=NAVY, labelpad=2)
    ax.set_ylabel("axis 2: disease spread", fontsize=7, color=NAVY, labelpad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--assets", default="figures/fig4_pipeline/assets")
    args = ap.parse_args()
    R, A = Path(args.results), Path(args.assets)
    feats = R / "05_features_dinov2/remote/features_dinov2"
    t0 = time.time()
    rng = np.random.default_rng(1)

    log("step 1/5: loading features")
    P = np.load(R / "08_projection/projection_k16.npz")
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of, dis_of = {}, {}
    with open(R / "01_crop_disease_table/labelled_images.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "plantvillage_color" and r["role"] == "image":
                crop_of[r["path"]], dis_of[r["path"]] = r["crop"].strip().lower(), r["disease"].lower()
    Zpv = (np.load(feats / "features_plantvillage_color.npy").astype(np.float64) - mean) @ W
    groups = {"healthy": [], "early": [], "septoria": [], "mold": []}
    for k, r in enumerate([r for r in index if r["dataset"] == "plantvillage_color"]):
        p = r["path"]
        if r["split"] != "train" or crop_of.get(p) != "tomato":
            continue
        d = "healthy" if r["is_healthy"] == "1" else dis_of.get(p, "")
        key = "healthy" if d == "healthy" else "early" if "early" in d else "septoria" if "septoria" in d else "mold" if "mold" in d else None
        if key:
            groups[key].append(Zpv[k])
    groups = {k: np.array(v) for k, v in groups.items()}
    rows_f = [r for r in index if r["dataset"] == "fig_sartak_sfld"]
    Zf = (np.load(feats / "features_fig_sartak_sfld.npy").astype(np.float64) - mean) @ W
    ix = lambda split, cls=None, healthy=None: [i for i, r in enumerate(rows_f) if r["split"] == split and
                                                (cls is None or r["class"] == cls) and (healthy is None or (r["is_healthy"] == "1") == healthy)]
    fh, fh_val, fa = Zf[ix("train", healthy=True)], Zf[ix("test", healthy=True) + ix("val", healthy=True)], Zf[ix("test", "anthracnose")]

    log("step 2/5: affine map and pushed models")
    mA, CA = groups["healthy"].mean(axis=0), w17.shrunk_cov(groups["healthy"])
    mB, CB = fh.mean(axis=0), w17.shrunk_cov(fh)
    T = w17.affine_T(CA, CB)
    pushed = {k: mB + (groups[k] - mA) @ T.T for k in ("early", "septoria", "mold")}
    pushed_h = mB + (groups["healthy"] - mA) @ T.T

    e1 = (mB - mA) / np.linalg.norm(mB - mA)
    dm = np.array([groups[k].mean(axis=0) - mA for k in ("early", "septoria", "mold")])
    dm = dm - np.outer(dm @ e1, e1)
    e2 = np.linalg.svd(dm, full_matrices=False)[2][0]
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    V = np.stack([e1, e2], axis=1)
    c0 = 0.5 * (mA + mB)
    proj = lambda Z: (Z - c0) @ V

    log("step 3/5: framed feature-space panels")
    fig, ax = plt.subplots(figsize=(3.0, 2.55), dpi=300)
    frame(ax, [ell_pts(proj(g)) for g in groups.values()])
    for k in ("mold", "septoria", "early", "healthy"):
        Z2 = proj(groups[k])
        sel = rng.choice(len(Z2), size=min(140, len(Z2)), replace=False)
        ax.scatter(Z2[sel, 0], Z2[sel, 1], s=5, color=TEAL[k], alpha=0.6, linewidths=0, zorder=3)
        ell(ax, Z2, NAVY if k == "healthy" else TEAL[k], lw=1.8 if k == "healthy" else 1.4, fill=0.0 if k == "healthy" else 0.14)
    fig.savefig(A / "space_source.png", transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.55), dpi=300)
    A2, F2 = proj(fa), proj(fh)
    frame(ax, [ell_pts(F2)] + [ell_pts(proj(p)) for p in pushed.values()] + [A2])
    sel = rng.choice(len(F2), size=140, replace=False)
    ax.scatter(F2[sel, 0], F2[sel, 1], s=5, color=AMBER, alpha=0.6, linewidths=0, zorder=3)
    ell(ax, F2, NAVY, lw=1.8, fill=0.0)
    for k in ("early", "septoria", "mold"):
        ell(ax, proj(pushed[k]), TEAL[k], lw=1.4, ls=(0, (4, 2.2)), fill=0.0)
    ax.scatter(A2[:, 0], A2[:, 1], s=14, facecolor="none", edgecolor="#D55E00", linewidths=0.8, zorder=4)
    fig.savefig(A / "space_target.png", transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    log("step 4/5: map T heat map and posterior of one real fig leaf")
    cm = LinearSegmentedColormap.from_list("tw", [TEAL["early"], "#FFFFFF", AMBER])
    fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=300)
    lim = np.abs(T).max()
    ax.imshow(T, cmap=cm, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#AEB8C6")
    fig.savefig(A / "map_T.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    models = [(pushed[k].mean(axis=0), w17.shrunk_cov(pushed[k])) for k in ("early", "septoria", "mold")]
    x = fa[3:4]
    L = np.array([w17.gauss_loglik(x, m, C)[0] for m, C in models])
    pr = np.exp(L - L.max())
    pr /= pr.sum()
    (A / "posterior.json").write_text(json.dumps({"probabilities": [round(float(v), 3) for v in pr],
                                                  "classes": ["early blight", "septoria", "leaf mold"]}), encoding="utf-8")
    log(f"  posterior of one fig anthracnose image under the pushed tomato models: {np.round(pr, 3)}")

    log("step 5/5: agreement check between two source crops on the real pair (tomato and potato early blight pushed to fig)")
    pot = {"healthy": [], "early": []}
    for k, r in enumerate([r for r in index if r["dataset"] == "plantvillage_color"]):
        pth = r["path"]
        if r["split"] != "train" or crop_of.get(pth) != "potato":
            continue
        d = "healthy" if r["is_healthy"] == "1" else dis_of.get(pth, "")
        key = "healthy" if d == "healthy" else "early" if "early" in d else None
        if key:
            pot[key].append(Zpv[k])
    pot = {k: np.array(v) for k, v in pot.items()}
    mP, CP = pot["healthy"].mean(axis=0), w17.shrunk_cov(pot["healthy"])
    TP = w17.affine_T(CP, CB)
    from_tomato = mB + (groups["early"] - mA) @ T.T          # tomato early blight pushed to fig
    from_potato = mB + (pot["early"] - mP) @ TP.T           # potato early blight pushed to fig
    n = min(len(from_tomato), len(from_potato))
    Xa, Xb = from_tomato[rng.permutation(len(from_tomato))[:n]], from_potato[rng.permutation(len(from_potato))[:n]]
    half = n // 2
    Xtr = np.vstack([Xa[:half], Xb[:half]])
    ytr = np.array([0] * half + [1] * half)
    Xte = np.vstack([Xa[half:], Xb[half:]])
    yte = np.array([0] * (n - half) + [1] * (n - half))
    clf = LogisticRegression(max_iter=1000, C=0.5).fit(Xtr, ytr)
    sc = clf.decision_function(Xte)
    acc = float(np.mean((sc > 0) == yte))
    k, m = int(round(acc * len(yte))), len(yte)
    z = 1.645
    p_ = k / m
    wl = (p_ + z * z / (2 * m) - z * np.sqrt(p_ * (1 - p_) / m + z * z / (4 * m * m))) / (1 + z * z / m)
    bound = max(0.0, 2 * wl - 1)
    fig, ax = plt.subplots(figsize=(2.3, 1.25), dpi=300)
    bins = np.linspace(sc.min(), sc.max(), 28)
    ax.hist(sc[yte == 0], bins=bins, color=TEAL["early"], alpha=0.75, label="tomato early blight pushed")
    ax.hist(sc[yte == 1], bins=bins, color=AMBER, alpha=0.75, label="potato early blight pushed")
    ax.set_yticks([])
    ax.set_xlabel("discriminator score", fontsize=7, color=NAVY, labelpad=1)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.savefig(A / "disc_hist.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    (A / "agreement.json").write_text(json.dumps({"accuracy": round(acc, 3), "wilson_lower": round(float(wl), 3),
                                                   "tv_bound": round(float(bound), 3), "test_images": m,
                                                   "pair": "tomato and potato early blight pushed to fig"}), encoding="utf-8")
    log(f"  tomato vs potato early blight pushed to fig: discriminator accuracy {acc:.3f} on {m} images, Wilson lower {wl:.3f}, TV bound {bound:.3f}")
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
