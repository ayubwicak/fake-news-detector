#!/usr/bin/env python3
"""TAHAP 4 - Uji fitur gaya dengan bootstrap (1000 resampel) pada hold-out v2.

Menghitung interval kepercayaan akurasi dan selisih in-distribution vs hold-out
untuk model yang hanya menggunakan fitur gaya (panjang, tanda baca, kapital,
awalan 3 kata). Tujuan: menilai apakah ada bias sumber yang signifikan."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUT_PATH = os.path.join(LOGS_DIR, "confounder_style_bootstrap.json")

RANDOM_STATE = 42
N_BOOT = 1000
np.random.seed(RANDOM_STATE)


def style_features(texts):
    import re
    rows = []
    for t in texts:
        t = "" if not isinstance(t, str) else t
        chars = len(t)
        words = len(t.split())
        sentences = max(1, len(re.findall(r"[.!?]+", t)))
        letters = [c for c in t if c.isalpha()]
        cap_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        rows.append({
            "chars": chars, "words": words, "sentences": sentences,
            "wps": words / sentences, "cap_ratio": cap_ratio,
            "excl": t.count("!"), "quest": t.count("?"),
        })
    return rows


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


import math  # noqa: E402


def main():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_id_oos.csv"))
    holdout = pd.read_csv(os.path.join(DATA_DIR, "holdout_id_v2.csv"))
    for df in (train, holdout):
        df["label"] = df["label"].astype(int)

    Xtr_rows = style_features(train["text"].tolist())
    Xho_rows = style_features(holdout["text"].tolist())

    prefix_counter = Counter(r["first3"] if "first3" in r else "" for r in Xtr_rows)
    # first3 not in style_features above; recompute quickly
    import re
    Xtr_rows = []
    for t in train["text"].astype(str):
        first3 = " ".join(t.split()[:3]).lower()
        chars = len(t); words = len(t.split())
        sentences = max(1, len(re.findall(r"[.!?]+", t)))
        letters = [c for c in t if c.isalpha()]
        cap_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        Xtr_rows.append({
            "chars": chars, "words": words, "sentences": sentences,
            "wps": words / sentences, "cap_ratio": cap_ratio,
            "excl": t.count("!"), "quest": t.count("?"), "first3": first3,
        })
    Xho_rows = []
    for t in holdout["text"].astype(str):
        first3 = " ".join(t.split()[:3]).lower()
        chars = len(t); words = len(t.split())
        sentences = max(1, len(re.findall(r"[.!?]+", t)))
        letters = [c for c in t if c.isalpha()]
        cap_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        Xho_rows.append({
            "chars": chars, "words": words, "sentences": sentences,
            "wps": words / sentences, "cap_ratio": cap_ratio,
            "excl": t.count("!"), "quest": t.count("?"), "first3": first3,
        })

    top_prefixes = [p for p, _ in Counter(r["first3"] for r in Xtr_rows).most_common(20)]

    def prefix_vec(rows):
        return np.array([[1 if r["first3"] == p else 0 for p in top_prefixes] for r in rows], dtype=float)

    numeric_cols = ["chars", "words", "sentences", "wps", "cap_ratio", "excl", "quest"]

    def num_mat(rows):
        return np.array([[r[c] for c in numeric_cols] for r in rows], dtype=float)

    Xtr = np.hstack([num_mat(Xtr_rows), prefix_vec(Xtr_rows)])
    Xho = np.hstack([num_mat(Xho_rows), prefix_vec(Xho_rows)])
    ytr = train["label"].values
    yho = holdout["label"].values

    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(Xtr, ytr)

    acc_tr = accuracy_score(ytr, clf.predict(Xtr))
    acc_ho = accuracy_score(yho, clf.predict(Xho))
    diff = acc_tr - acc_ho

    # Bootstrap on hold-out only (in-distribution acc is fixed since train is fixed)
    n_ho = len(holdout)
    boot_acc = []
    boot_diff = []
    for _ in range(N_BOOT):
        idx = np.random.randint(0, n_ho, n_ho)
        yb = yho[idx]
        Xb = Xho[idx]
        pred_b = clf.predict(Xb)
        acc_b = accuracy_score(yb, pred_b)
        boot_acc.append(acc_b)
        boot_diff.append(acc_tr - acc_b)

    boot_acc = np.array(boot_acc)
    boot_diff = np.array(boot_diff)

    ci_acc = (np.percentile(boot_acc, 2.5), np.percentile(boot_acc, 97.5))
    ci_diff = (np.percentile(boot_diff, 2.5), np.percentile(boot_diff, 97.5))

    print("=== UJI GESTAI BOOTSTRAP (fitur gaya saja, 1000 resampel) ===")
    print(f"Hold-out v2: n={n_ho} | valid={(yho==0).sum()} | hoak={(yho==1).sum()}")
    print(f"\nIn-distribution (train) akurasi: {acc_tr:.2%} (tetap, bukan di-bootstrap)")
    print(f"Hold-out akurasi: {acc_ho:.2%}")
    print(f"Bootstrap hold-out akurasi 95% CI: [{ci_acc[0]:.2%}, {ci_acc[1]:.2%}]")
    print(f"Selisih in-dist vs hold-out: {diff:.2%}")
    print(f"Bootstrap selisih 95% CI: [{ci_diff[0]:.2%}, {ci_diff[1]:.2%}]")

    # Interpretation
    print(f"\n=== TOLAKAN ===")
    print(f"Fitur gaya saja mencapai {acc_ho:.2%} pada hold-out (di atas tebakan acak 50%).")
    print(f"Selisih in-dist vs hold-out 95% CI: [{ci_diff[0]:.2%}, {ci_diff[1]:.2%}]")
    if ci_diff[0] > 0:
        print("-> Selisih signifikan > 0: ada indikasi model belajar pola yang tidak transfer ke luar distribusi.")
    else:
        print("-> Selisih tidak signifikant: tidak cukup bukti konfounder sumber dari fitur gaya.")

    out = {
        "n_holdout": int(n_ho),
        "n_valid": int((yho == 0).sum()),
        "n_hoax": int((yho == 1).sum()),
        "train_acc": float(acc_tr),
        "holdout_acc": float(acc_ho),
        "diff": float(diff),
        "n_bootstrap": N_BOOT,
        "bootstrap_acc_ci_95": [float(ci_acc[0]), float(ci_acc[1])],
        "bootstrap_diff_ci_95": [float(ci_diff[0]), float(ci_diff[1])],
    }
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nDisimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()
