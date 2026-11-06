"""Module 46: summary tables from the stored results (markdown and csv).

Reads only the result files of earlier modules; recomputes nothing. Tables written to results/46_tables/:
  T1_datasets            dataset sizes and augmentation (local audit)
  T2_main_accuracy       per world, healthy n = 200: accuracy of NA, MC, STD, CORAL, HATC and the best method (module 20)
  T3_equivalence_counts  outcome counts of HATC against the best simple method (module 37)
  T4_labelled_saving     labels per class a probe needs to reach HATC (module 31)
  T5_device              D1 to D4 headline numbers (modules 39 to 42)
  T6_agreement           AUROC of the agreement bound, ensemble disagreement and low confidence (module 30)
  T7_healthy_sweep       accuracy against healthy n for world a (module 43)
  T8_select              HATC-S against fixed methods on untouched worlds (module 50)

CPU module, runs locally in a few seconds.
Usage: python 46_tables.py --results results --output results/46_tables
"""

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_table(out, name, header, rows, note=""):
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    text = "\n".join(lines) + ("\n\n" + note if note else "") + "\n"
    (out / f"{name}.md").write_text(text, encoding="utf-8")
    log(f"  wrote {name} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="results/46_tables")
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/8: T1 datasets")
    rows = [[r["dataset"], r["images"], r["gigabytes"], r["augmented_path_images"]] for r in read(R / "local_audit/dataset_summary.csv")]
    write_table(out, "T1_datasets", ["dataset", "images", "GB", "augmented-path images"], rows)

    log("step 2/8: T2 main accuracy")
    rows = []
    for r in read(R / "20_align_baselines/summary.csv"):
        if r["healthy_n"] == "200":
            rows.append([r["world"]] + [f"{float(r['acc_' + m]):.3f}" for m in ("NA", "MC", "STD", "CORAL", "HATC")] + [r["best_method"]])
    write_table(out, "T2_main_accuracy", ["world", "NA", "MC", "STD", "CORAL", "HATC", "best"], rows,
                "Mean accuracy at 200 healthy target images, synthetic worlds, operator strength 2 (module 20).")

    log("step 3/8: T3 equivalence counts")
    prim = read(R / "37_equivalence/primary_outcomes.csv")
    cnt = Counter((r["healthy_n"], r["outcome"]) for r in prim)
    rows = [[n, o, c] for (n, o), c in sorted(cnt.items())]
    write_table(out, "T3_equivalence_counts", ["healthy n", "outcome", "cells"], rows,
                "All exploratory (7 crop clusters, below the registered minimum of 8).")

    log("step 4/8: T4 labelled saving")
    rows = [[r["world"], r["healthy_n"], f"{float(r['hatc_accuracy']):.3f}", r["reaching_labels_per_class"], r["saving_labels_per_class"]]
            for r in read(R / "31_labelled_saving/labelled_saving.csv")]
    write_table(out, "T4_labelled_saving", ["world", "healthy n", "HATC accuracy", "labels/class to reach HATC", "saving"], rows)

    log("step 5/8: T5 device")
    d1 = read(R / "39_d1_enrolment/enrolment_rates.csv")
    d2 = read(R / "40_d2_routing/routing_rates.csv")
    d3 = read(R / "41_d3_drift/drift_results.csv")
    d4 = read(R / "42_d4_sequential/sequential_results.csv")
    rows = [["D1", "false enrolment with gate (range over n and mixed fraction)",
             f"{min(float(r['false_enrolment_rate_gate']) for r in d1):.2f} to {max(float(r['false_enrolment_rate_gate']) for r in d1):.2f}"],
            ["D1", "false enrolment without gate", f"{min(float(r['false_enrolment_rate_no_gate']) for r in d1):.2f}"]]
    for r in d2:
        rows.append(["D2", r["test_set"] + ": disease / wrong-crop / answer", f"{r['disease_route']} / {r['wrong_crop_route']} / {r['answer']}"])
    by = defaultdict(list)
    for r in d3:
        by[r["policy"]].append(float(r["mean_accuracy"]))
    for p, v in by.items():
        rows.append(["D3", f"mean accuracy, policy {p}", f"{sum(v) / len(v):.3f}"])
    rows.append(["D4", "refusal rate over worlds", f"{min(float(r['refusal_rate']) for r in d4):.2f} to {max(float(r['refusal_rate']) for r in d4):.2f}"])
    write_table(out, "T5_device", ["component", "measure", "value"], rows)

    log("step 6/8: T6 agreement")
    rows = [[r["world"], r["healthy_n"], r["failure_rate"], r["auroc_bound"], r["auroc_ens_disagree"], r["auroc_low_conf"]]
            for r in read(R / "30_certificate_baselines/auroc.csv")]
    write_table(out, "T6_agreement", ["world", "healthy n", "failure rate", "AUROC bound", "AUROC ensemble disagreement", "AUROC low confidence"], rows)

    log("step 7/8: T7 healthy sweep (world a)")
    rows = [[r["healthy_n"]] + [f"{float(r['acc_' + m]):.3f}" for m in ("NA", "MC", "STD", "S50", "CORAL", "HATC")]
            for r in read(R / "43_ablations/sweep_n_k16/summary.csv") if r["world"] == "a"]
    write_table(out, "T7_healthy_sweep", ["healthy n", "NA", "MC", "STD", "S50", "CORAL", "HATC"], rows)

    log("step 8/8: T8 HATC-S")
    rows = [[r["world"], r["healthy_n"], f"{float(r['acc_HATC']):.3f}", f"{float(r['acc_SELECT']):.3f}", r["best_fixed"], f"{float(r['select_minus_HATC']):+.3f}"]
            for r in read(R / "50_hatc_select/seed9002/summary.csv")]
    write_table(out, "T8_select", ["world", "healthy n", "HATC", "HATC-S", "best fixed", "HATC-S minus HATC"], rows,
                "Post-hoc, exploratory; untouched world seed.")
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
