"""Module 36b: coverage and power of the cluster bootstrap against the number of clusters (proposal 7.7, 8).

The real-data design needs 25 pairs; the synthetic worlds have 7 crops. This module asks, by simulation, what the
cluster bootstrap of modules 36 and 37 delivers at 7, 8, 12, 25 and 50 clusters.

Variance components come from the stored replicate accuracies of module 20 (paired difference HATC minus the best
simple method): the between-crop standard deviation and the within-crop standard deviation, per world, healthy n = 200.
Simulation: cluster effect = delta + N(0, between), observation = effect + N(0, within), M replicates per cluster.
  - COVERAGE: share of simulations whose 90 percent percentile bootstrap interval contains the true delta
  - POWER (different): share where the 90 percent interval excludes 0 when delta = DELTA_ALT (a 3-point difference)
  - POWER (equivalent): share where the 90 percent interval lies inside +-MARGIN when delta = 0 (TOST at alpha 0.05)

Outputs (results/36b_bootstrap_coverage_power/): coverage_power.csv, variance_components.csv, manifest.json
CPU module, runs locally in about 1 to 2 minutes.

Usage:
    python 36b_bootstrap_coverage_power.py --replicates results/20_align_baselines/replicate_results.csv
        --output results/36b_bootstrap_coverage_power
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

CLUSTERS = [7, 8, 12, 25, 50]
SIMS = 2000
RESAMPLES = 1000
MARGIN = 0.02
DELTA_ALT = 0.03
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def components(path):
    data = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["healthy_n"] != "200":
                continue
            data.setdefault(r["world"], {}).setdefault(r["method"], []).append((r["crop"], int(r["replicate"]), float(r["accuracy"])))
    rows = []
    for world, methods in sorted(data.items()):
        mean = {m: np.mean([a for _, _, a in v]) for m, v in methods.items()}
        best = max(["NA", "MC", "STD"], key=lambda m: mean[m])
        h = {(c, i): a for c, i, a in methods["HATC"]}
        b = {(c, i): a for c, i, a in methods[best]}
        keys = sorted(h)
        crops = np.array([k[0] for k in keys])
        d = np.array([h[k] - b[k] for k in keys])
        cm = np.array([d[crops == c].mean() for c in np.unique(crops)])
        within = float(np.sqrt(np.mean([d[crops == c].var(ddof=1) for c in np.unique(crops) if (crops == c).sum() > 1])))
        m_per = float(np.mean([(crops == c).sum() for c in np.unique(crops)]))
        between_var = max(cm.var(ddof=1) - within ** 2 / m_per, 0.0)
        rows.append({"world": world, "best_simple": best, "mean_diff": round(float(d.mean()), 4),
                     "between_sd": round(float(np.sqrt(between_var)), 4), "within_sd": round(within, 4),
                     "replicates_per_cluster": round(m_per, 1)})
    return rows


def simulate(rng, k, m, delta, sb, sw):
    eff = delta + sb * rng.standard_normal((SIMS, k))
    obs = eff[:, :, None] + sw * rng.standard_normal((SIMS, k, m))
    cm = obs.mean(axis=2)                                  # cluster means (equal cluster sizes)
    point = cm.mean(axis=1)
    idx = rng.integers(0, k, size=(SIMS, RESAMPLES, k))
    boots = np.take_along_axis(cm[:, None, :].repeat(RESAMPLES, 1), idx, axis=2).mean(axis=2)
    lo, hi = np.percentile(boots, 5, axis=1), np.percentile(boots, 95, axis=1)
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    log("step 1/3: variance components from the stored replicate accuracies")
    vc = components(args.replicates)
    sb = float(np.median([r["between_sd"] for r in vc]))
    sw = float(np.median([r["within_sd"] for r in vc]))
    m = int(round(np.median([r["replicates_per_cluster"] for r in vc])))
    log(f"  median between SD {sb:.4f}, within SD {sw:.4f}, {m} replicates per cluster")

    log("step 2/3: simulating coverage and power")
    rows = []
    for k in CLUSTERS:
        _, lo, hi = simulate(rng, k, m, 0.0, sb, sw)
        cover = float(np.mean((lo <= 0) & (0 <= hi)))
        equiv = float(np.mean((lo > -MARGIN) & (hi < MARGIN)))
        _, lo2, hi2 = simulate(rng, k, m, -DELTA_ALT, sb, sw)
        cover_alt = float(np.mean((lo2 <= -DELTA_ALT) & (-DELTA_ALT <= hi2)))
        diff = float(np.mean((hi2 < 0) | (lo2 > 0)))
        rows.append({"clusters": k, "coverage_90_delta0": round(cover, 3), "coverage_90_delta_minus3pts": round(cover_alt, 3),
                     "power_different_at_3pts": round(diff, 3), "power_equivalent_at_0_margin_2pts": round(equiv, 3)})
        log(f"  {k} clusters: coverage {cover:.3f} / {cover_alt:.3f}, power different {diff:.3f}, equivalent {equiv:.3f}")

    log("step 3/3: writing outputs")
    with open(out / "coverage_power.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "variance_components.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(vc[0].keys()))
        w.writeheader()
        w.writerows(vc)
    (out / "manifest.json").write_text(json.dumps({
        "module": "36b_bootstrap_coverage_power", "clusters": CLUSTERS, "sims": SIMS, "resamples": RESAMPLES,
        "margin": MARGIN, "delta_alt": DELTA_ALT, "between_sd": sb, "within_sd": sw, "replicates_per_cluster": m,
        "seed": SEED, "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
