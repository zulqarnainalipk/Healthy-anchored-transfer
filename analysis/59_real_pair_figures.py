"""Module 59: figures of the real-pair test (R7 and R8) from results/53_real_pair_test.

R7  (a) mean balanced accuracy of the methods over the 80 tier-2 units for DINOv2 and BioCLIP features, with 90% two-way cluster
    bootstrap intervals of the mean; (b) per-unit difference HATC minus mean-centring (DINOv2), sorted, coloured by disease type,
    with the equivalence margin of +-2 points and the means for both encoders.
R8  (a) HATC minus MC against the one-class healthy detector minus MC per unit; (b) wrong-disease control: HATC with the right
    and with a wrong source disease on the same units.
Okabe-Ito palette, Arial. CPU module, seconds. Usage: python 59_real_pair_figures.py --results results --output figures
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
BLUE, ORANGE, GREEN, VERM, SKY, PINK, YEL, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#F0E442", "#999999"
METHODS = ["NA", "MC", "STD", "CORAL", "HATC", "OC", "probe5", "probe10", "probe25"]
LABELS = ["NA", "MC", "STD", "CORAL", "HATC", "one-class", "probe 5", "probe 10", "probe 25"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def units(path, tier="2"):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["tier"] == tier]


def val(r, m):
    v = r.get("ba_" + m, "")
    return float(v) if v not in ("", None) else np.nan


def boot_mean(rows, m, resamples=2000, seed=4):
    ok = [r for r in rows if not np.isnan(val(r, m))]
    tg = sorted({r["target"] for r in ok}); sr = sorted({r["source"] for r in ok})
    T = np.array([tg.index(r["target"]) for r in ok]); S = np.array([sr.index(r["source"]) for r in ok])
    v = np.array([val(r, m) for r in ok])
    rng = np.random.default_rng(seed)
    b = np.empty(resamples)
    for i in range(resamples):
        w = np.bincount(rng.integers(0, len(tg), len(tg)), minlength=len(tg))[T] * np.bincount(rng.integers(0, len(sr), len(sr)), minlength=len(sr))[S]
        b[i] = (w * v).sum() / w.sum()
    return v.mean(), np.percentile(b, 5), np.percentile(b, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="figures")
    args = ap.parse_args()
    R, out = Path(args.results) / "53_real_pair_test", Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    dino, bio = units(R / "pw_pca/units.csv"), units(R / "pw_bioclip_pca/units.csv")
    log("R7")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(METHODS))
    for k, (rows, col, lab) in enumerate(((dino, BLUE, "DINOv2"), (bio, ORANGE, "BioCLIP"))):
        stats = [boot_mean(rows, m) for m in METHODS]
        mean = np.array([s[0] for s in stats]); lo = np.array([s[1] for s in stats]); hi = np.array([s[2] for s in stats])
        ax[0].bar(x + (k - 0.5) * 0.38, mean, 0.36, yerr=[mean - lo, hi - mean], color=col, label=lab, error_kw={"lw": 0.8, "capsize": 1.5})
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(LABELS, rotation=40, ha="right")
    ax[0].set_ylim(0.5, 0.92)
    ax[0].set_ylabel("balanced accuracy (80 units)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[0].set_title("(a) mean over units", fontsize=9, loc="left")
    d = np.array([val(r, "HATC") - val(r, "MC") for r in dino])
    types = [r["disease"] for r in dino]
    order = np.argsort(d)
    uniq = ["rust", "mosaic virus", "downy mildew", "powdery mildew", "early blight", "black rot", "anthracnose", "sheath blight", "alternaria leaf spot"]
    cols = dict(zip(uniq, [VERM, GREEN, BLUE, ORANGE, PINK, SKY, "#000000", GREY, YEL]))
    ax[1].axvspan(-0.02, 0.02, color="#E6E6E6", zorder=0)
    ax[1].scatter(d[order], np.arange(len(d)), c=[cols[types[i]] for i in order], s=10, linewidths=0, zorder=3)
    ax[1].axvline(0, color="black", lw=0.6)
    ax[1].axvline(d.mean(), color=BLUE, lw=1.2, ls="-")
    ax[1].axvline(np.mean([val(r, "HATC") - val(r, "MC") for r in bio]), color=ORANGE, lw=1.2, ls="--")
    ax[1].set_xlabel("HATC minus mean-centring")
    ax[1].set_yticks([])
    ax[1].set_title("(b) per unit (DINOv2)", fontsize=9, loc="left")
    for t in uniq:
        ax[1].scatter([], [], c=cols[t], s=14, label=t)
    ax[1].legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"R7_real_pairs.{ext}", bbox_inches="tight")
    plt.close(fig)

    log("R8")
    right = {(r["source"], r["target"], r["disease"]): r for r in dino}
    wrong = {(r["source"], r["target"], r["disease"]): r for r in units(R / "pw_pca_wrongdisease/units.csv")}
    keys = sorted(set(right) & set(wrong))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    mc = np.array([val(right[k], "MC") for k in right])
    hd = np.array([val(right[k], "HATC") - val(right[k], "MC") for k in right])
    oc = np.array([val(right[k], "OC") - val(right[k], "MC") for k in right])
    ax[0].plot([-0.15, 0.3], [-0.15, 0.3], color=GREY, lw=0.8)
    ax[0].axhline(0, color="black", lw=0.5); ax[0].axvline(0, color="black", lw=0.5)
    ax[0].scatter(oc, hd, s=10, color=BLUE, linewidths=0)
    ax[0].set_xlabel("one-class healthy detector minus MC")
    ax[0].set_ylabel("HATC minus MC")
    ax[0].set_title("(a) per unit", fontsize=9, loc="left")
    names = ["NA", "MC", "HATC"]
    xr = np.arange(len(names))
    ax[1].bar(xr - 0.19, [np.mean([val(right[k], m) for k in keys]) for m in names], 0.36, color=BLUE, label="right disease")
    ax[1].bar(xr + 0.19, [np.mean([val(wrong[k], m) for k in keys]) for m in names], 0.36, color=VERM, label="wrong disease")
    ax[1].set_xticks(xr); ax[1].set_xticklabels(names)
    ax[1].set_ylim(0.5, 0.82)
    ax[1].set_ylabel(f"balanced accuracy ({len(keys)} units)")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("(b) wrong-disease control", fontsize=9, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"R8_real_pair_controls.{ext}", bbox_inches="tight")
    plt.close(fig)
    log("done")


if __name__ == "__main__":
    main()
