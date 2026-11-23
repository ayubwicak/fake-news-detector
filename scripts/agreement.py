#!/usr/bin/env python3
"""Kesepakatan dua anotator pada hold-out v3 (Cohen's kappa).

Membaca kolom `label_anotator1` dan `label_anotator2` (opsional) dari CSV,
menghitung Cohen's kappa, persentase kesepakatan, serta mencantumkan baris yang
tidak sepakat. Tidak mengubah data dan tidak menulis label final.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    """Interval kepercayaan Wilson 95% untuk proporsi."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [centre - half, centre + half]


def _as_label(value) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if text in ("0", "1") else None


def compute_agreement(rows: list[dict]) -> dict:
    """Hitung kappa dan daftar baris yang tidak sepakat."""
    if not rows or "label_anotator1" not in rows[0] or "label_anotator2" not in rows[0]:
        return {
            "status": "fail",
            "violations": ["Kolom label_anotator1 dan label_anotator2 harus ada."],
            "n_pairs": 0,
        }

    pairs = []
    invalid = []
    for i, row in enumerate(rows, start=2):
        a, b = _as_label(row.get("label_anotator1")), _as_label(row.get("label_anotator2"))
        if a is None and b is None:
            continue  # baris belum diberi dua label
        if a is None or b is None:
            invalid.append(i)
            continue
        pairs.append((i, row.get("id", ""), a, b))

    if invalid:
        return {
            "status": "fail",
            "violations": [f"Satu anotator kosong pada baris: {invalid[:20]}"],
            "n_pairs": len(pairs),
        }
    if len(pairs) < 2:
        return {
            "status": "fail",
            "violations": ["Minimal dua baris berlabel ganda dibutuhkan untuk kappa."],
            "n_pairs": len(pairs),
        }

    from sklearn.metrics import cohen_kappa_score

    y1 = [p[2] for p in pairs]
    y2 = [p[3] for p in pairs]
    agreed = sum(1 for a, b in zip(y1, y2) if a == b)
    kappa = float(cohen_kappa_score(y1, y2, labels=[0, 1]))
    ci = wilson_interval(agreed, len(pairs))
    disagreements = [
        {"csv_line": line, "id": row_id, "anotator1": a, "anotator2": b}
        for line, row_id, a, b in pairs if a != b
    ]
    return {
        "status": "pass",
        "n_pairs": len(pairs),
        "agreement_count": agreed,
        "agreement_rate": agreed / len(pairs),
        "agreement_rate_wilson_95": ci,
        "cohens_kappa": kappa,
        "disagreements": disagreements,
        "note": (
            "Kappa dihitung hanya pada baris yang diberi label oleh kedua anotator. "
            "Baris tidak sepakat harus direkonsiliasi sebelum kolom `label` diisi final."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "data", "holdout_id_v3.csv"))
    parser.add_argument("--report", default=os.path.join(BASE_DIR, "logs", "holdout_v3_agreement.json"))
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        report = {"status": "fail", "violations": [f"berkas tidak ditemukan: {args.csv}"],
                  "n_pairs": 0}
    else:
        with open(args.csv, newline="", encoding="utf-8-sig") as fh:
            report = compute_agreement(list(csv.DictReader(fh)))
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["csv_path"] = os.path.relpath(args.csv, BASE_DIR)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    if report["status"] == "pass":
        print(f"n = {report['n_pairs']} pasangan | Cohen's kappa = {report['cohens_kappa']:.3f} | "
              f"kesepakatan = {report['agreement_rate']:.1%} "
              f"(Wilson 95% [{report['agreement_rate_wilson_95'][0]:.1%}, "
              f"{report['agreement_rate_wilson_95'][1]:.1%}])")
        print(f"Baris tidak sepakat: {len(report['disagreements'])}")
    else:
        print("Gagal:", "; ".join(report["violations"]))
    print(f"Laporan: {args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
