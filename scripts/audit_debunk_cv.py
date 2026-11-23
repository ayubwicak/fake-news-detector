#!/usr/bin/env python3
"""TAHAP 3.2 - Estimasi ambang kemiripan dengan cross-validation (hold-out v2
dipisahkan menjadi A/B, ambang dipilih di A, dievaluasi di B, dan sebaliknya).
Serta estimasi presisi pada berbagai proporsi prevalensi hoak."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

INDEX_PATH = os.path.join(MODELS_DIR, "hoax_index.npy")
META_PATH = os.path.join(MODELS_DIR, "hoax_index_meta.json")
HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_id_v2.csv")
OUT_PATH = os.path.join(LOGS_DIR, "debunk_cv.json")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
SEED = 42


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def evaluate_at_threshold(max_sim, labels, thr):
    flagged = max_sim >= thr
    n_flagged = int(flagged.sum())
    n_valid = int((labels == 0).sum())
    n_hoax = int((labels == 1).sum())
    false_match = int((flagged & (labels == 0)).sum())
    true_match = int((flagged & (labels == 1)).sum())
    fmr = false_match / n_valid if n_valid else 0.0
    recall = true_match / n_hoax if n_hoax else 0.0
    precision = true_match / n_flagged if n_flagged else 0.0
    return {
        "threshold": thr,
        "n_flagged": n_flagged,
        "false_match_valid": false_match,
        "fmr_valid": fmr,
        "true_match_hoax": true_match,
        "recall_hoax": recall,
        "precision": precision,
    }


def main():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("Index debunked belum ada.")
        return
    if not os.path.exists(HOLDOUT_PATH):
        print("Hold-out v2 tidak ditemukan.")
        return

    index = np.load(INDEX_PATH).astype(np.float32)
    ho = pd.read_csv(HOLDOUT_PATH)
    ho["label"] = ho["label"].astype(int)
    texts = ho["text"].astype(str).tolist()
    labels = ho["label"].values
    n_valid = int((labels == 0).sum())
    n_hoax = int((labels == 1).sum())
    print(f"Hold-out v2: n={len(ho)} | valid={n_valid} | hoak={n_hoax}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    q = model.encode(texts, batch_size=32, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
    sim = cosine_sim(q, index)
    max_sim = sim.max(axis=1)

    thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)

    # Split A/B stratified, 50/50
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    (a_idx, b_idx), = list(sss.split(texts, labels))
    a_idx = np.array(a_idx); b_idx = np.array(b_idx)

    def pick_and_eval(select_idx, eval_idx, tag):
        sel_sim = max_sim[select_idx]; sel_labels = labels[select_idx]
        ev_sim = max_sim[eval_idx]; ev_labels = labels[eval_idx]

        results = [evaluate_at_threshold(sel_sim, sel_labels, t) for t in thresholds]
        cand = [r for r in results if r["fmr_valid"] <= 0.05]
        if cand:
            chosen = max(cand, key=lambda r: r["recall_hoax"])
            rule = "fmr_valid <= 5%, recall tertinggi"
        else:
            chosen = min(results, key=lambda r: r["fmr_valid"])
            rule = "fmr_valid terendah"
        thr = chosen["threshold"]

        ev = evaluate_at_threshold(ev_sim, ev_labels, thr)
        print(f"\n=== {tag}: ambang dipilih di {tag.split(' -> ')[0]}, dievaluasi di {tag.split(' -> ')[1]} ===")
        print(f"  Ambang: {thr:.2f} ({rule})")
        print(f"  Eval  n_valid={int((ev_labels==0).sum())} n_hoak={int((ev_labels==1).sum())}")
        print(f"  False match valid: {ev['false_match_valid']}  FMR={ev['fmr_valid']:.2%}")
        print(f"  True match hoak: {ev['true_match_hoax']}  Recall={ev['recall_hoax']:.2%}")
        print(f"  Precision: {ev['precision']:.2%}")
        return thr, ev

    thr_ab, ev_ab = pick_and_eval(a_idx, b_idx, "A -> B")
    thr_ba, ev_ba = pick_and_eval(b_idx, a_idx, "B -> A")

    # Pooled: ambang dari gabungan selection (simulasi optimis), eval pada keseluruhan
    all_results = [evaluate_at_threshold(max_sim, labels, t) for t in thresholds]
    cand = [r for r in all_results if r["fmr_valid"] <= 0.05]
    if cand:
        chosen_all = max(cand, key=lambda r: r["recall_hoax"])
    else:
        chosen_all = min(all_results, key=lambda r: r["fmr_valid"])
    thr_all = chosen_all["threshold"]
    ev_all = evaluate_at_threshold(max_sim, labels, thr_all)
    print(f"\n=== Pooled (ambang dipilih dan dievaluasi di seluruh hold-out v2) ===")
    print(f"  Ambang: {thr_all:.2f}")
    print(f"  FMR valid: {ev_all['fmr_valid']:.2%} | Recall hoak: {ev_all['recall_hoax']:.2%} | Precision: {ev_all['precision']:.2%}")

    # Wilson CI for the more honest cross-validated estimates.
    # Use the eval-set sizes from each split (conservative = smaller n -> wider CI).
    print(f"\n=== WILSON CI 95% (estimasi jujur, cross-validated) ===")
    for tag, ev, sel_idx, eval_idx in [
        ("A -> B", ev_ab, a_idx, b_idx),
        ("B -> A", ev_ba, b_idx, a_idx),
    ]:
        nv = int((labels[eval_idx] == 0).sum())
        nh = int((labels[eval_idx] == 1).sum())
        ci_fmr = wilson_ci(ev["fmr_valid"], nv)
        ci_recall = wilson_ci(ev["recall_hoax"], nh)
        print(f"  {tag}: FMR valid 95% CI [{ci_fmr[0]:.2%}, {ci_fmr[1]:.2%}] (n_valid={nv}) | "
              f"Recall hoak 95% CI [{ci_recall[0]:.2%}, {ci_recall[1]:.2%}] (n_hoak={nh})")

    # Precision at various prevalences (Bayes)
    print(f"\n=== ESTIMASI PRECISI PADA BERBAGAI PREVALENSI HOXAK ===")
    # Use cross-validated recall and FMR as operating characteristics
    rec_cv = (ev_ab["recall_hoax"] + ev_ba["recall_hoax"]) / 2
    fmr_cv = (ev_ab["fmr_valid"] + ev_ba["fmr_valid"]) / 2
    print(f"  Recall hoak (CV rata-rata): {rec_cv:.2%}")
    print(f"  FMR valid (CV rata-rata): {fmr_cv:.2%}")
    for prev in (0.10, 0.30, 0.50):
        # precision = (recall * prev) / (recall * prev + fmr * (1 - prev))
        prec = (rec_cv * prev) / (rec_cv * prev + fmr_cv * (1 - prev)) if (rec_cv * prev + fmr_cv * (1 - prev)) > 0 else 0.0
        print(f"  Prevalensi hoak {prev:.0%}: precision estimasi = {prec:.2%}")

    out = {
        "embed_model": EMBED_MODEL,
        "n_holdout": int(len(ho)),
        "n_valid": n_valid,
        "n_hoax": n_hoax,
        "threshold_A_to_B": float(thr_ab),
        "eval_A_to_B": ev_ab,
        "threshold_B_to_A": float(thr_ba),
        "eval_B_to_A": ev_ba,
        "threshold_pooled": float(thr_all),
        "eval_pooled": ev_all,
        "recall_cv_avg": float(rec_cv),
        "fmr_cv_avg": float(fmr_cv),
        "wilson_ci_fmr_valid": [ci_fmr[0], ci_fmr[1]],
        "wilson_ci_recall_hoak": [ci_recall[0], ci_recall[1]],
        "precision_at_prevalence": {
            "0.10": float((rec_cv * 0.10) / (rec_cv * 0.10 + fmr_cv * 0.90)),
            "0.30": float((rec_cv * 0.30) / (rec_cv * 0.30 + fmr_cv * 0.70)),
            "0.50": float((rec_cv * 0.50) / (rec_cv * 0.50 + fmr_cv * 0.50)),
        },
        "note": "Ambang 0,66 asli dipilih dievaluasi pada hold-out v2 yang sama (selection bias). Estimasi CV di atas lebih jujur.",
    }
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()
