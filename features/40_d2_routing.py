"""Module 40: D2 abstention routing with a disease branch (proposal section 6).

When the system is not sure, D2 routes a leaf by two tests, in the 16-dimensional projection:
  - envelope: is the leaf inside the healthy envelope of an enrolled crop? (Mahalanobis distance to the
    crop's healthy Gaussian at or below the 95th percentile of that crop's own training distances)
  - signature: does the leaf match a stored disease signature of an enrolled crop? (log-likelihood under
    the disease Gaussian at or above the 5th percentile of that disease's own training log-likelihoods)
Routes:
  answer          inside an envelope (answered as healthy for that crop)
  disease route   outside every envelope and matching a stored signature (returns the disease class)
  wrong-crop      outside every envelope and matching no signature (refusal, check the crop)
Guided recapture and referral (D2 routes iii and iv) need repeated captures and are not simulated here.

Test sets (validation split of PlantVillage), with thresholds set on the TRAINING split only:
  diseased leaves of enrolled crops     expected: disease route, with the correct disease
  healthy leaves of enrolled crops      expected: answer (inside the envelope)
  leaves of unenrolled crops (healthy and diseased)   expected: wrong-crop route
Baseline: a single abstention route: any leaf outside every envelope abstains, with no routing.

Outputs (results/40_d2_routing/): routing_rates.csv, manifest.json

CPU module, runs locally.

Usage:
    python 40_d2_routing.py --features ... --labelled ... --projection ... --output results/40_d2_routing
"""

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
s16 = importlib.util.spec_from_file_location("w16", HERE / "16_world_generator.py")
w16 = importlib.util.module_from_spec(s16)
s16.loader.exec_module(w16)
s17 = importlib.util.spec_from_file_location("w17", HERE / "17_benchmark_hatc.py")
w17 = importlib.util.module_from_spec(s17)
s17.loader.exec_module(w17)

MIN_HEALTHY = 250
MIN_DISEASE = 60
UNENROLLED = 3        # number of crops held out as unenrolled
SEED = 2026


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mahal(X, m, C):
    D = X - m
    inv = np.linalg.inv(C)
    return np.sqrt(np.einsum("ij,jk,ik->i", D, inv, D))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labelled", required=True)
    ap.add_argument("--projection", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    feats = Path(args.features)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("step 1/3: loading PlantVillage features by crop, disease and split")
    P = np.load(args.projection)
    mean, W = P["mean"].astype(np.float64), P["W"].astype(np.float64)
    with open(feats / "features_index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    crop_of = {}
    with open(args.labelled, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == "plantvillage_color" and r["role"] == "image":
                crop_of[r["path"]] = r["crop"].strip().lower()
    Z = (np.load(feats / "features_plantvillage_color.npy").astype(np.float64) - mean) @ W
    rows = [i for i, r in enumerate(index) if r["dataset"] == "plantvillage_color"]
    groups = {}  # (crop, disease or healthy, split) -> array
    tmp = {}
    for k, i in enumerate(rows):
        r = index[i]
        if r["path"] not in crop_of or r["split"] not in ("train", "val"):
            continue
        key = (crop_of[r["path"]], "healthy" if r["is_healthy"] == "1" else r["class"], r["split"])
        tmp.setdefault(key, []).append(Z[k])
    groups = {k: np.array(v) for k, v in tmp.items()}

    crops = sorted({c for (c, d, s) in groups if d == "healthy" and s == "train"
                    and len(groups[(c, d, s)]) >= MIN_HEALTHY})
    rng = np.random.default_rng(SEED)
    held = list(rng.choice(crops, size=UNENROLLED, replace=False))
    enrolled = [c for c in crops if c not in held]
    log(f"  enrolled: {enrolled}; unenrolled: {held}")

    log("step 2/3: profiles and thresholds from the training split")
    env = {}
    for c in enrolled:
        H = groups[(c, "healthy", "train")]
        m, C = H.mean(axis=0), w17.shrunk_cov(H)
        env[c] = (m, C, float(np.percentile(mahal(H, m, C), 95)))
    sig = {}
    for (c, d, s), arr in groups.items():
        if c in enrolled and d != "healthy" and s == "train" and len(arr) >= MIN_DISEASE:
            m, C = arr.mean(axis=0), w17.shrunk_cov(arr)
            ll = w17.gauss_loglik(arr, m, C)
            sig[(c, d)] = (m, C, float(np.percentile(ll, 5)))
    log(f"  {len(env)} envelopes, {len(sig)} disease signatures")

    def route(X):
        inside = np.zeros(len(X), dtype=bool)
        for c, (m, C, thr) in env.items():
            inside |= mahal(X, m, C) <= thr
        best_ll = np.full(len(X), -np.inf)
        best_sig = np.full(len(X), -1)
        keys = list(sig)
        match = np.zeros(len(X), dtype=bool)
        for si, k in enumerate(keys):
            m, C, thr = sig[k]
            ll = w17.gauss_loglik(X, m, C)
            hit = ll >= thr
            better = hit & (ll > best_ll)
            best_ll[better] = ll[better]
            best_sig[better] = si
            match |= hit
        r = np.where(inside, "answer", np.where(match, "disease", "wrong_crop"))
        return r, best_sig, keys

    log("step 3/3: routing validation leaves")
    results = []
    sets = {
        "diseased, enrolled crop": [(c, d) for (c, d) in sig],
    }
    # Diseased leaves of enrolled crops
    X, y_true = [], []
    for (c, d) in sig:
        arr = groups.get((c, d, "val"))
        if arr is not None:
            X.append(arr)
            y_true += [(c, d)] * len(arr)
    X = np.vstack(X)
    r, best, keys = route(X)
    correct_class = np.array([keys[b] == yt if b >= 0 else False for b, yt in zip(best, y_true)])
    n = len(X)
    results.append({"test_set": "diseased leaves, enrolled crops", "leaves": n,
                    "answer": round(float(np.mean(r == "answer")), 4),
                    "disease_route": round(float(np.mean(r == "disease")), 4),
                    "wrong_crop_route": round(float(np.mean(r == "wrong_crop")), 4),
                    "disease_class_correct_given_disease_route":
                        round(float(correct_class[r == "disease"].mean()), 4) if (r == "disease").any() else "",
                    "baseline_single_abstain_share": round(float(np.mean(r != "answer")), 4)})
    # Healthy leaves of enrolled crops
    Xh = np.vstack([groups[(c, "healthy", "val")] for c in enrolled if (c, "healthy", "val") in groups])
    r, _, _ = route(Xh)
    results.append({"test_set": "healthy leaves, enrolled crops", "leaves": len(Xh),
                    "answer": round(float(np.mean(r == "answer")), 4),
                    "disease_route": round(float(np.mean(r == "disease")), 4),
                    "wrong_crop_route": round(float(np.mean(r == "wrong_crop")), 4),
                    "disease_class_correct_given_disease_route": "",
                    "baseline_single_abstain_share": round(float(np.mean(r != "answer")), 4)})
    # Leaves of unenrolled crops (healthy and diseased)
    Xu = [arr for (c, d, s), arr in groups.items() if c in held and s == "val"]
    Xu = np.vstack(Xu)
    r, _, _ = route(Xu)
    results.append({"test_set": "leaves of unenrolled crops", "leaves": len(Xu),
                    "answer": round(float(np.mean(r == "answer")), 4),
                    "disease_route": round(float(np.mean(r == "disease")), 4),
                    "wrong_crop_route": round(float(np.mean(r == "wrong_crop")), 4),
                    "disease_class_correct_given_disease_route": "",
                    "baseline_single_abstain_share": round(float(np.mean(r != "answer")), 4)})
    with open(out / "routing_rates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (out / "manifest.json").write_text(json.dumps({
        "module": "40_d2_routing", "enrolled": enrolled, "unenrolled": held, "envelopes": len(env),
        "signatures": len(sig), "seed": SEED, "seconds": round(time.time() - t0, 1),
        "note": "recapture and referral routes are not simulated"}, indent=2), encoding="utf-8")
    for r in results:
        log(f"  {r['test_set']}: answer {r['answer']}, disease {r['disease_route']}, wrong-crop {r['wrong_crop_route']}")
    log("done")


if __name__ == "__main__":
    main()
