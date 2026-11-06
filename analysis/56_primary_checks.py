"""Module 56: robustness checks of the primary real-pair outcome (module 53, tier 2).

Primary quantity: per unit, balanced accuracy of HATC minus balanced accuracy of the best simple method (chosen by the mean over
units among NA, MC, STD). Checks:
  1. the two-way cluster bootstrap with five different seeds and 20,000 resamples (stability of the interval)
  2. leave-one-target-crop-out and leave-one-source-crop-out (does one crop drive the result?)
  3. cluster permutation test: signs of the unit differences are flipped together for all units of one target crop
     (null: no systematic difference), 20,000 permutations
  4. per-target-crop and per-source-crop mean differences and the share that is positive
  5. paired contrasts between methods that tell how much of the gain is generic: HATC minus the one-class healthy detector (OC),
     HATC minus CORAL, probe 5 minus HATC
  6. agreement across split seeds (if several units.csv files are given): correlation of unit values and the primary difference per seed
Outputs: results/56_primary_checks/<tag>/checks.json and by_crop.csv. CPU module, about 3 minutes.
Usage: python 56_primary_checks.py --units results/53_real_pair_test/pw_pca/units.csv --tag pca
       --other-seeds results/53_real_pair_test/pw_pca_seed1/units.csv results/53_real_pair_test/pw_pca_seed2/units.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

SIMPLE = ["NA", "MC", "STD"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read(path, tier="2"):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["tier"] == tier]


def val(r, m):
    v = r.get("ba_" + m, "")
    return float(v) if v not in ("", None) else np.nan


def two_way(t_idx, s_idx, v, resamples, seed):
    rng = np.random.default_rng(seed)
    nt, ns = t_idx.max() + 1, s_idx.max() + 1
    out = np.empty(resamples)
    for b in range(resamples):
        wt = np.bincount(rng.integers(0, nt, nt), minlength=nt)
        ws = np.bincount(rng.integers(0, ns, ns), minlength=ns)
        w = wt[t_idx] * ws[s_idx]
        out[b] = (w * v).sum() / w.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--other-seeds", nargs="*", default=[])
    ap.add_argument("--tag", default="run")
    ap.add_argument("--output", default="results/56_primary_checks")
    args = ap.parse_args()
    out = Path(args.output) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = read(args.units)
    best = max(SIMPLE, key=lambda m: np.mean([val(r, m) for r in rows]))
    d = np.array([val(r, "HATC") - val(r, best) for r in rows])
    tg = sorted({r["target"] for r in rows})
    sr = sorted({r["source"] for r in rows})
    ti = np.array([tg.index(r["target"]) for r in rows])
    si = np.array([sr.index(r["source"]) for r in rows])
    res = {"units": len(rows), "target_crops": len(tg), "source_crops": len(sr), "best_simple": best,
           "primary_mean_diff": round(float(d.mean()), 4)}

    log("1/6 bootstrap with several seeds")
    bs = {}
    for seed in (1, 2, 3, 2026, 99):
        b = two_way(ti, si, d, 20000, seed)
        bs[str(seed)] = [round(float(np.percentile(b, 5)), 4), round(float(np.percentile(b, 95)), 4)]
    res["bootstrap_90ci_by_seed"] = bs

    log("2/6 leave-one-crop-out")
    lo_t = {t: round(float(d[ti != i].mean()), 4) for i, t in enumerate(tg)}
    lo_s = {s: round(float(d[si != i].mean()), 4) for i, s in enumerate(sr)}
    res["leave_one_target_out_range"] = [min(lo_t.values()), max(lo_t.values())]
    res["leave_one_source_out_range"] = [min(lo_s.values()), max(lo_s.values())]
    res["leave_one_target_out_most_influential"] = min(lo_t, key=lo_t.get)
    res["leave_one_source_out_most_influential"] = min(lo_s, key=lo_s.get)

    log("3/6 cluster sign-flip permutation test")
    rng = np.random.default_rng(11)
    obs = d.mean()
    null = np.empty(20000)
    for b in range(20000):
        flip = rng.choice([-1.0, 1.0], size=len(tg))
        null[b] = (flip[ti] * d).mean()
    res["permutation_p_two_sided"] = round(float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)), 5)

    log("4/6 per-crop summaries")
    with open(out / "by_crop.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "crop", "units", "mean_diff_hatc_minus_best_simple"])
        for i, t in enumerate(tg):
            w.writerow(["target", t, int((ti == i).sum()), round(float(d[ti == i].mean()), 4)])
        for i, s in enumerate(sr):
            w.writerow(["source", s, int((si == i).sum()), round(float(d[si == i].mean()), 4)])
    tm = np.array([d[ti == i].mean() for i in range(len(tg))])
    sm = np.array([d[si == i].mean() for i in range(len(sr))])
    res["target_crops_positive"] = int((tm > 0).sum())
    res["source_crops_positive"] = int((sm > 0).sum())

    log("5/6 paired contrasts")
    pair = {}
    for name, a, c in (("HATC_minus_OC", "HATC", "OC"), ("HATC_minus_CORAL", "HATC", "CORAL"), ("probe5_minus_HATC", "probe5", "HATC"),
                       ("probe10_minus_HATC", "probe10", "HATC"), ("OC_minus_best_simple", "OC", best)):
        x = np.array([val(r, a) - val(r, c) for r in rows])
        ok = ~np.isnan(x)
        b = two_way(ti[ok], si[ok], x[ok], 10000, 5)
        pair[name] = {"units": int(ok.sum()), "mean": round(float(x[ok].mean()), 4),
                      "ci90": [round(float(np.percentile(b, 5)), 4), round(float(np.percentile(b, 95)), 4)]}
    res["paired_contrasts"] = pair

    log("6/6 agreement across split seeds")
    seeds = {}
    base = {(r["source"], r["target"], r["disease"]): val(r, "HATC") - val(r, best) for r in rows}
    for path in args.other_seeds:
        other = read(path)
        o = {(r["source"], r["target"], r["disease"]): val(r, "HATC") - val(r, best) for r in other}
        keys = sorted(set(base) & set(o))
        a, c = np.array([base[k] for k in keys]), np.array([o[k] for k in keys])
        seeds[Path(path).parent.name] = {"units": len(keys), "mean_diff": round(float(c.mean()), 4),
                                         "correlation_with_base": round(float(np.corrcoef(a, c)[0, 1]), 4),
                                         "max_abs_unit_difference": round(float(np.abs(a - c).max()), 4)}
    res["other_seeds"] = seeds
    (out / "checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(json.dumps(res)[:1500])
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
