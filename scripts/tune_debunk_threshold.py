#!/usr/bin/env python3
"""TAHAP B.2 - Tuning ambang kemiripan (similarity threshold) untuk pencocokan
klaim dengan hoaks yang sudah dibantah.

Tujuan: pilih ambang yang memberi false match rendah pada berita valid di
 hold-out v2 (validasi), lalu laporkan precision dan recall pencocokan pada
 hold-out v2.

Metrik:
  - False match pada valid = persentase item valid yang similarity >= ambang.
  - Precision = di antara item yang ter-flag similar, berapa pct hoak sebenarnya.
  - Recall = di antara item hoak, berapa pct yang ter-flag similar.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

INDEX_PATH = os.path.join(MODELS_DIR, "hoax_index.npy")
META_PATH = os.path.join(MODELS_DIR, "hoax_index_meta.json")
HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_id_v2.csv")
OUT_PATH = os.path.join(LOGS_DIR, "debunk_threshold.json")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity antara setiap baris a dan setiap baris b."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def main():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        print("Index debunked belum ada. Jalankan scripts/build_hoax_index.py terlebih dahulu.")
        return

    if not os.path.exists(HOLDOUT_PATH):
        print("Hold-out v2 tidak ditemukan.")
        return

    print("Memuat indeks debunked ...")
    index = np.load(INDEX_PATH).astype(np.float32)
    meta = json.load(open(META_PATH, encoding="utf-8"))
    print(f"  {len(meta)} entri, {index.shape[1]} dimensi")

    print("Memuat model embeddings", EMBED_MODEL, " ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    print("Memuat hold-out v2 ...")
    ho = pd.read_csv(HOLDOUT_PATH)
    ho["label"] = ho["label"].astype(int)
    texts = ho["text"].astype(str).tolist()
    labels = ho["label"].values
    n = len(ho)
    n_valid = int((labels == 0).sum())
    n_hoax = int((labels == 1).sum())
    print(f"  n={n} | valid={n_valid} | hoak={n_hoax}")

    print("Encode teks hold-out ...")
    q = model.encode(texts, batch_size=32, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)

    print("Hitung kemiripan maksimum ke indeks ...")
    sim = cosine_similarity_matrix(q, index)  # (n, 12912)
    max_sim = sim.max(axis=1)
    nearest_idx = sim.argmax(axis=1)

    # Sweep ambang
    thresholds = np.round(np.arange(0.50, 0.96, 0.01), 2)
    rows = []
    for thr in thresholds:
        flagged = max_sim >= thr
        n_flagged = int(flagged.sum())
        false_match = int((flagged & (labels == 0)).sum())
        fmr = false_match / n_valid if n_valid else 0.0
        true_match = int((flagged & (labels == 1)).sum())
        tmr = true_match / n_hoax if n_hoax else 0.0  # recall
        precision = true_match / n_flagged if n_flagged else 0.0
        rows.append({
            "threshold": float(thr),
            "n_flagged": n_flagged,
            "false_match_valid": false_match,
            "false_match_rate_valid": fmr,
            "true_match_hoax": true_match,
            "recall_hoax": tmr,
            "precision": precision,
        })

    # Pilih ambang: false match rate valid <= 5%, lalu recall tertinggi.
    # Jika tidak ada yang <=5%, ambil yang false_match_rate paling rendah.
    cand = [r for r in rows if r["false_match_rate_valid"] <= 0.05]
    if cand:
        chosen = max(cand, key=lambda r: r["recall_hoax"])
        rule = "false_match_rate_valid <= 5%, recall tertinggi"
    else:
        chosen = min(rows, key=lambda r: r["false_match_rate_valid"])
        rule = "false_match_rate_valid terendah"

    thr = chosen["threshold"]
    print(f"\n=== HASIL SWEEP AMBANG (top_k={TOP_K}) ===")
    print(f"{'ambang':>7} {'n_flag':>7} {'false_match_valid':>17} {'FMR_valid':>9} "
          f"{'true_match_hoax':>15} {'recall_hoax':>10} {'precision':>9}")
    for r in rows:
        mark = "  <-- PILIHAN" if r["threshold"] == thr else ""
        print(f"{r['threshold']:>7.2f} {r['n_flagged']:>7} {r['false_match_valid']:>17} "
              f"{r['false_match_rate_valid']:>8.2%} {r['true_match_hoax']:>15} "
              f"{r['recall_hoax']:>9.2%} {r['precision']:>8.2%}{mark}")

    print(f"\nAmbang dipilih: {thr:.2f} ({rule})")
    print(f"  False match pada valid: {chosen['false_match_valid']}/{n_valid} "
          f"({chosen['false_match_rate_valid']:.2%})")
    print(f"  True match pada hoak: {chosen['true_match_hoax']}/{n_hoax} "
          f"(recall={chosen['recall_hoax']:.2%})")
    print(f"  Precision pencocokan: {chosen['precision']:.2%}")

    # Contoh item valid yang false match (bila ada) untuk transparansi
    flagged = max_sim >= thr
    false_valid = ho[flagged & (labels == 0)]
    if len(false_valid):
        print(f"\nContoh valid yang false-match (ambang {thr:.2f}):")
        for _, r in false_valid.head(5).iterrows():
            pos = list(ho.index).index(r.name)
            print(f"  - {str(r.text)[:110]}  (max_sim={max_sim[pos]:.3f})")

    out = {
        "embed_model": EMBED_MODEL,
        "n_index": int(len(meta)),
        "n_holdout": int(n),
        "n_valid": n_valid,
        "n_hoax": n_hoax,
        "top_k": TOP_K,
        "chosen_threshold": float(thr),
        "selection_rule": rule,
        "false_match_valid": int(chosen["false_match_valid"]),
        "false_match_rate_valid": float(chosen["false_match_rate_valid"]),
        "true_match_hoax": int(chosen["true_match_hoax"]),
        "recall_hoax": float(chosen["recall_hoax"]),
        "precision": float(chosen["precision"]),
        "sweep": rows,
    }
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()
