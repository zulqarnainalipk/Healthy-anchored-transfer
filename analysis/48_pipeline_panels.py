"""Module 48: data-driven panels for the pipeline figure (figures/fig4_pipeline).

Builds, from the stored DINOv2 features, the real content shown in the pipeline figure:
  space_source.png  2-D view of the 16-d projected features of tomato (healthy, early blight, septoria, leaf mold, train
                    split) with 2-sigma ellipses of the class Gaussians
  space_target.png  the same 2-D view for fig: healthy train images, their Gaussian, the tomato disease models pushed
                    through the healthy-anchored affine map, and fig anthracnose test images
  feature_strip.png the 1,536 feature values of one fig anthracnose image as a heat strip
  tokens.png        patch tiles cut from the target photograph (the encoder's input tokens)
The 2-D view is the first two principal axes of tomato plus fig healthy samples in the 16-d projected space; the pushed
models are computed in 16-d and only drawn in 2-D. Illustrative: the pair (tomato to fig) is used for display only.

CPU module, runs locally in a few seconds.
Usage: python analysis/48_pipeline_panels.py --results results --assets figures/fig4_pipeline/assets
"""

import argparse
import csv
import importlib.util
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("w17", HERE.parent / "features" / "17_benchmark_hatc.py")
w17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w17)

TEAL = {"healthy": "#00634A", "early": "#009E73", "septoria": "#4DC4A1", "mold": "#99DCC6"}
AMBER, NAVY = "#E69F00", "#0072B2"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ellipse(ax, pts, **kw):
    m, C = pts.mean(axis=0), np.cov(pts.T)
    w, v = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(m, 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--assets", default="figures/fig4_pipeline/assets")
    args = ap.parse_args()
    R, A = Path(args.results), Path(args.assets)
    feats = R / "05_features_dinov2/remote/features_dinov2"
    t0 = time.time()

    log("step 1/4: loading features and projection")
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
    rows_pv = [r for r in index if r["dataset"] == "plantvillage_color"]
    groups = {"healthy": [], "early": [], "septoria": [], "mold": []}
    for k, r in enumerate(rows_pv):
        p = r["path"]
        if r["split"] != "train" or crop_of.get(p) != "tomato":
            continue
        d = "healthy" if r["is_healthy"] == "1" else dis_of.get(p, "")
        key = "healthy" if d == "healthy" else "early" if "early" in d else "septoria" if "septoria" in d else "mold" if "mold" in d else None
        if key:
            groups[key].append(Zpv[k])
    groups = {k: np.array(v) for k, v in groups.items()}
    log("  tomato: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()))
    Xf = np.load(feats / "features_fig_sartak_sfld.npy").astype(np.float64)
    rows_f = [r for r in index if r["dataset"] == "fig_sartak_sfld"]
    Zf = (Xf - mean) @ W
    fh = Zf[[i for i, r in enumerate(rows_f) if r["split"] == "train" and r["is_healthy"] == "1"]]
    fa_idx = [i for i, r in enumerate(rows_f) if r["split"] == "test" and r["class"] == "anthracnose"]
    fa = Zf[fa_idx]
    log(f"  fig: healthy {len(fh)}, anthracnose test {len(fa)}")

    log("step 2/4: healthy-anchored push-through and 2-D view")
    mA, CA = groups["healthy"].mean(axis=0), w17.shrunk_cov(groups["healthy"])
    mB, CB = fh.mean(axis=0), w17.shrunk_cov(fh)
    T = w17.affine_T(CA, CB)
    pushed = {k: mB + (groups[k] - mA) @ T.T for k in ("early", "septoria", "mold")}
    # 2-D view: axis 1 along the healthy shift (mB - mA), axis 2 the main spread of the tomato disease means
    e1 = (mB - mA) / np.linalg.norm(mB - mA)
    dm = np.array([groups[k].mean(axis=0) - mA for k in ("early", "septoria", "mold")])
    dm = dm - np.outer(dm @ e1, e1)
    e2 = np.linalg.svd(dm, full_matrices=False)[2][0]
    e2 = e2 - (e2 @ e1) * e1
    e2 /= np.linalg.norm(e2)
    V = np.stack([e1, e2], axis=1)
    proj = lambda Z: (Z - 0.5 * (mA + mB)) @ V

    def ell_pts(pts):
        m, C = pts.mean(axis=0), np.cov(pts.T)
        w, v = np.linalg.eigh(C)
        th = np.linspace(0, 2 * np.pi, 60)
        return m + (v @ (2 * np.sqrt(w)[:, None] * np.vstack([np.cos(th), np.sin(th)]))).T

    def limits(sets):
        pts = np.vstack(sets)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = 0.08 * (hi - lo)
        return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    def base(ax, lims):
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_facecolor("none")

    log("step 3/4: drawing the two feature-space panels")
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(3.0, 2.8), dpi=300)
    src_sets = [ell_pts(proj(g)) for g in groups.values()]
    base(ax, limits(src_sets))
    for k in ("mold", "septoria", "early", "healthy"):
        Z2 = proj(groups[k])
        sel = rng.choice(len(Z2), size=min(260, len(Z2)), replace=False)
        ax.scatter(Z2[sel, 0], Z2[sel, 1], s=3, color=TEAL[k], alpha=0.45, linewidths=0)
        ellipse(ax, Z2, fill=False, ec=NAVY if k == "healthy" else TEAL[k], lw=2.0 if k == "healthy" else 1.6)
    fig.savefig(A / "space_source.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.8), dpi=300)
    A2 = proj(fa)
    F2 = proj(fh)
    tgt_sets = [ell_pts(F2)] + [ell_pts(proj(p)) for p in pushed.values()] + [A2]
    base(ax, limits(tgt_sets))
    sel = rng.choice(len(F2), size=min(260, len(F2)), replace=False)
    ax.scatter(F2[sel, 0], F2[sel, 1], s=3, color=AMBER, alpha=0.5, linewidths=0)
    ellipse(ax, F2, fill=False, ec=NAVY, lw=2.0)
    for k in ("early", "septoria", "mold"):
        ellipse(ax, proj(pushed[k]), fill=False, ec=TEAL[k], lw=1.8, ls=(0, (4, 2.5)))
    ax.scatter(A2[:, 0], A2[:, 1], s=9, facecolor="none", edgecolor="#D55E00", linewidths=0.7)
    fig.savefig(A / "space_target.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    log("extra: mini transport panel (healthy source and target clouds with the map)")
    fig, ax = plt.subplots(figsize=(2.3, 1.5), dpi=300)
    S2, F2m = proj(groups["healthy"]), proj(fh)
    lims = limits([ell_pts(S2), ell_pts(F2m)])
    base(ax, lims)
    ax.set_aspect("equal", adjustable="datalim")
    for P2, col, ecol in ((S2, TEAL["healthy"], TEAL["early"]), (F2m, AMBER, AMBER)):
        sel = rng.choice(len(P2), size=min(200, len(P2)), replace=False)
        ax.scatter(P2[sel, 0], P2[sel, 1], s=2.5, color=col, alpha=0.5, linewidths=0)
        ellipse(ax, P2, fill=False, ec=ecol, lw=1.8)
    a, b = S2.mean(axis=0), F2m.mean(axis=0)
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=NAVY, lw=1.6, shrinkA=2, shrinkB=2))
    fig.savefig(A / "map_mini.png", transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log("step 4/4: feature strip and patch tokens")
    row = np.load(feats / "features_fig_sartak_sfld.npy").astype(np.float32)[fa_idx[3]]
    z = (row - row.mean()) / (row.std() + 1e-6)
    strip = z.reshape(24, 64)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("tw", [TEAL["early"], "#F7F5EF", AMBER])
    fig, ax = plt.subplots(figsize=(3.2, 1.2), dpi=300)
    ax.imshow(strip, cmap=cm, vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    ax.axis("off")
    fig.savefig(A / "feature_strip.png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    im = Image.open(A / "tgt_new.jpg").convert("RGB")
    n, s, gap = 4, im.size[0] // 4, 3
    tiles = [im.crop((j * s, i * s, (j + 1) * s, (i + 1) * s)).resize((60, 60), Image.LANCZOS) for i in range(n) for j in range(n)]
    pick = [tiles[i] for i in (1, 2, 5, 6, 9, 10)]
    canvas = Image.new("RGBA", (len(pick) * (60 + gap), 60), (0, 0, 0, 0))
    for k, t in enumerate(pick):
        canvas.paste(t, (k * (60 + gap), 0))
    canvas.save(A / "tokens.png")
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
