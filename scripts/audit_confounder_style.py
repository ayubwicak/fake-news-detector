#!/usr/bin/env python3
"""TAHAP A.2 - Uji konfounder sumber menggunakan fitur gaya (style features).

Hipotesis: jika model belajar gaya sumber (bukan isi), maka fitur gaya sederhana
akan mencapai akurasi tinggi pada data dalam distribusi (in-distribution) namun
turun drastis pada hold-out (out-of-source / independen).

Fitur gaya:
  - panjang teks (karakter)
  - jumlah kata
  - jumlah kalimat
  - rasio huruf kapital
  - jumlah tanda seru / tanya
  - awalan: 3 kata pertama (one-hot untuk top-N awalan)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def style_features(texts):
    """Ekstrak fitur gaya dari daftar teks (string mentah)."""
    rows = []
    for t in texts:
        t = "" if not isinstance(t, str) else t
        chars = len(t)
        words = len(t.split())
        sentences = max(1, len(re.findall(r"[.!?]+", t)))
        letters = [c for c in t if c.isalpha()]
        cap_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        excl = t.count("!")
        quest = t.count("?")
        first3 = " ".join(t.split()[:3]).lower()
        rows.append(
            {
                "chars": chars,
                "words": words,
                "sentences": sentences,
                "words_per_sentence": words / sentences,
                "cap_ratio": cap_ratio,
                "excl": excl,
                "quest": quest,
                "first3": first3,
            }
        )
    return rows


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return (float(lo), float(hi))


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_id_oos.csv"))
    holdout = pd.read_csv(os.path.join(DATA_DIR, "holdout_id_v2.csv"))

    for df in (train, holdout):
        df["label"] = df["label"].astype(int)

    Xtr_rows = style_features(train["text"].tolist())
    Xho_rows = style_features(holdout["text"].tolist())

    # One-hot encode first-3-words prefix (top 20 prefixes in train)
    prefix_counter = Counter(r["first3"] for r in Xtr_rows)
    top_prefixes = [p for p, _ in prefix_counter.most_common(20)]

    def prefix_vector(rows):
        vec = []
        for r in rows:
            vec.append([1 if r["first3"] == p else 0 for p in top_prefixes])
        return np.array(vec, dtype=float)

    numeric_cols = ["chars", "words", "sentences", "words_per_sentence", "cap_ratio", "excl", "quest"]

    def numeric_matrix(rows):
        return np.array([[r[c] for c in numeric_cols] for r in rows], dtype=float)

    Xtr_num = numeric_matrix(Xtr_rows)
    Xho_num = numeric_matrix(Xho_rows)
    Xtr_pre = prefix_vector(Xtr_rows)
    Xho_pre = prefix_vector(Xho_rows)

    Xtr = np.hstack([Xtr_num, Xtr_pre])
    Xho = np.hstack([Xho_num, Xho_pre])

    ytr = train["label"].values
    yho = holdout["label"].values

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(Xtr, ytr)

    pred_tr = clf.predict(Xtr)
    proba_tr = clf.predict_proba(Xtr)[:, 1]
    acc_tr = accuracy_score(ytr, pred_tr)
    auc_tr = roc_auc_score(ytr, proba_tr)

    pred_ho = clf.predict(Xho)
    proba_ho = clf.predict_proba(Xho)[:, 1]
    acc_ho = accuracy_score(yho, pred_ho)
    auc_ho = roc_auc_score(yho, proba_ho)

    print("=== UJI KONFOUNDER SUMBER (fitur gaya saja) ===")
    print(f"Data latih : {len(train)} baris  | Hold-out v2 : {len(holdout)} baris")
    print(f"\nIn-distribution (train):")
    print(f"  Akurasi  = {acc_tr:.2%}")
    print(f"  ROC-AUC  = {auc_tr:.4f}")
    print(f"\nHold-out v2 (out-of-source / independen):")
    print(f"  Akurasi  = {acc_ho:.2%}")
    print(f"  ROC-AUC  = {auc_ho:.4f}")
    drop = acc_tr - acc_ho
    print(f"\nPenurunan akurasi train -> hold-out: {drop:.2%}")

    if acc_tr >= 0.70 and acc_ho < acc_tr - 0.15:
        print("\nBUkti KONFOUNDER SUMBER: fitur gaya berhasil dalam distribusi ")
        print("namun gagal di luar distribusi -> model belajar gaya sumber, bukan isi berita.")
    else:
        print("\nTidak cukup bukti konfounder sumber dari fitur gaya saja.")

    # CI Wilson for hold-out accuracy
    ci = wilson_ci(acc_ho, len(holdout))
    print(f"\nInterval kepercayaan 95% (Wilson) akurasi hold-out (gaya saja): [{ci[0]:.2%}, {ci[1]:.2%}]")

    # Save results
    out = {
        "train_acc": acc_tr,
        "train_auc": auc_tr,
        "holdout_acc": acc_ho,
        "holdout_auc": auc_ho,
        "drop": drop,
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "wilson_ci_95": [ci[0], ci[1]],
    }
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, "confounder_style.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan: {os.path.join(LOGS_DIR, 'confounder_style.json')}")


if __name__ == "__main__":
    main()
