"""Module 47: black-and-white result figures from the stored results (png and pdf).

Figures written to figures/:
  R1_main_accuracy      accuracy per world for NA, MC, STD, CORAL, HATC at 200 healthy images (module 20)
  R2_healthy_sweep      accuracy against healthy n, world a, with the shrunk map S50 (module 43)
  R3_hatc_vs_best       HATC minus best simple method with 90% intervals, healthy n = 50 and 200 (module 36)
  R4_labelled_probe     probe accuracy against labels per class, with the HATC level (modules 26, 31)
  R5_agreement_auroc    AUROC of agreement bound, ensemble disagreement and low confidence (module 30)
  R6_select             HATC-S minus HATC on untouched worlds (module 50)

Okabe-Ito palette (Wong 2011) with hatching and line styles kept so the figures also read in greyscale; Arial. CPU module, runs locally in a few seconds.
Usage: python 47_figures.py --results results --output figures
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "figure.dpi": 120})
HATCH = ["", "///", "...", "xxx", "\\\\\\"]
GREY = ["white", "0.85", "0.65", "0.45", "0.2"]
MC = {"NA": "#999999", "MC": "#56B4E9", "STD": "#009E73", "CORAL": "#D55E00", "HATC": "#0072B2", "S50": "#E69F00"}   # Okabe-Ito (Wong 2011)
MARK = ["o", "s", "^", "D", "v"]
LINE = ["-", "--", ":", "-.", (0, (5, 1))]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="figures")
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/6: R1 main accuracy")
    rows = [r for r in read(R / "20_align_baselines/summary.csv") if r["healthy_n"] == "200"]
    worlds = [r["world"] for r in rows]
    methods = ["NA", "MC", "STD", "CORAL", "HATC"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.16
    for i, m in enumerate(methods):
        ax.bar(np.arange(len(worlds)) + (i - 2) * w, [float(r["acc_" + m]) for r in rows], w, label=m,
               color=MC[m], edgecolor="black", hatch=HATCH[i], linewidth=0.6)
    ax.set_xticks(range(len(worlds)))
    ax.set_xticklabels(worlds)
    ax.set_xlabel("synthetic world")
    ax.set_ylabel("accuracy")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, out, "R1_main_accuracy")

    log("step 2/6: R2 healthy sweep")
    rows = [r for r in read(R / "43_ablations/sweep_n_k16/summary.csv") if r["world"] == "a"]
    n = [int(r["healthy_n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, m in enumerate(["NA", "MC", "STD", "S50", "HATC"]):
        ax.plot(n, [float(r["acc_" + m]) for r in rows], color=MC[m], linestyle=LINE[i], marker=MARK[i], ms=4, lw=1, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("healthy target images")
    ax.set_ylabel("accuracy (world a)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "R2_healthy_sweep")

    log("step 3/6: R3 HATC minus best simple")
    iv = [r for r in read(R / "36_cluster_bootstrap/intervals.csv") if r["contrast"] == "HATC - best_simple"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, hn in zip(axes, ("50", "200")):
        sub = [r for r in iv if r["healthy_n"] == hn]
        y = np.arange(len(sub))
        d = np.array([float(r["mean_difference"]) for r in sub])
        lo = np.array([float(r["ci90_low"]) for r in sub])
        hi = np.array([float(r["ci90_high"]) for r in sub])
        ax.hlines(y, lo, hi, color="#0072B2", lw=1.2)
        ax.plot(d, y, "o", color="#0072B2", ms=4)
        ax.axvline(0, color="black", lw=0.6)
        ax.axvspan(-0.02, 0.02, color="#E6E6E6", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([r["world"] for r in sub])
        ax.set_title(f"{hn} healthy images", fontsize=9)
        ax.set_xlabel("HATC minus best simple method")
    save(fig, out, "R3_hatc_vs_best")

    log("step 4/6: R4 labelled probe")
    curve = read(R / "31_labelled_saving/labelled_saving.csv")
    ks = [0, 5, 10, 25, 50, 100, 200]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    r0 = [r for r in curve if r["world"] == "a" and r["healthy_n"] == "200"][0]
    xs = [k if k else 1 for k in ks]
    ax.plot(xs, [float(r0[f"probe_{k}"]) for k in ks], "-o", color="#009E73", ms=4, lw=1.4, label="probe")
    ax.axhline(float(r0["hatc_accuracy"]), color="#0072B2", linestyle="--", lw=1.4, label="HATC")
    ax.set_xscale("log")
    ax.set_xlabel("labelled target images per class (0 plotted at 1)")
    ax.set_ylabel("accuracy (world a)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "R4_labelled_probe")

    log("step 5/6: R5 agreement AUROC")
    rows = [r for r in read(R / "30_certificate_baselines/auroc.csv") if r["healthy_n"] == "200"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    keys = [("auroc_bound", "agreement bound"), ("auroc_ens_disagree", "ensemble disagreement"), ("auroc_low_conf", "low confidence")]
    for i, (k, lab) in enumerate(keys):
        ax.bar(np.arange(len(rows)) + (i - 1) * 0.25, [float(r[k]) for r in rows], 0.25, label=lab,
               color=["#0072B2", "#D55E00", "#999999"][i], edgecolor="black", hatch=HATCH[i], linewidth=0.6)
    ax.axhline(0.5, color="black", lw=0.6, linestyle=":")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["world"] for r in rows])
    ax.set_ylabel("AUROC for failure")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save(fig, out, "R5_agreement_auroc")

    log("step 6/6: R6 HATC-S")
    rows = [r for r in read(R / "50_hatc_select/seed9002/summary.csv") if r["healthy_n"] == "200"]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.bar(range(len(rows)), [float(r["select_minus_HATC"]) for r in rows], color="#E69F00", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["world"] for r in rows])
    ax.set_ylabel("HATC-S minus HATC")
    save(fig, out, "R6_select")
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
