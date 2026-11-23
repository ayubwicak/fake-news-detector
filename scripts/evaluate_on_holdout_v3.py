#!/usr/bin/env python3
"""Evaluasi SATU PINTU pada hold-out v3.

Syarat menjalankan:
  1. `scripts/validate_holdout_v3.py` harus lolos (status=pass) dan hash CSV
     yang tersimpan di laporan validator harus sama dengan hash CSV sekarang.
  2. Skrip ini menolak dijalankan kedua kalinya dengan hash yang sama, kecuali
     memakai flag --rerun-acknowledged. Setiap percobaan dicatat di
     logs/holdout_v3_run_attempts.jsonl.

Skrip ini TIDAK melatih, tidak menyetel ambang, dan tidak memilih model.
Model yang tersedia dimuat dari artefak yang sudah ada; model tanpa artefak
dilaporkan sebagai tidak tersedia (bukan dilatih di sini).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text  # noqa: E402

GRAY_LOW, GRAY_HIGH = 0.30, 0.70
PREVALENCES = [0.10, 0.30, 0.50]

MODEL_ID_DIR = os.path.join(BASE_DIR, "models", "id")
INDOBERT_DIR = os.path.join(BASE_DIR, "models", "indobert")
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "holdout_id_v3.csv")
DEFAULT_VALIDATION = os.path.join(BASE_DIR, "logs", "holdout_v3_validation.json")
DEFAULT_RESULT = os.path.join(BASE_DIR, "logs", "holdout_v3_result.json")
DEFAULT_ATTEMPTS = os.path.join(BASE_DIR, "logs", "holdout_v3_run_attempts.jsonl")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    """Interval kepercayaan Wilson 95% untuk proporsi."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [centre - half, centre + half]


def _rate(k: int, n: int) -> float:
    return k / n if n else 0.0


def compute_classification_metrics(y_true: list[int], prob_fake: list[float],
                                   y_pred: list[int]) -> dict:
    """Metrik standar. Tidak melakukan tuning apa pun."""
    from sklearn.metrics import (
        accuracy_score, brier_score_loss, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    n = len(y_true)
    if n == 0:
        return {"n": 0, "error": "tidak ada sampel"}
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = float(roc_auc_score(y_true, prob_fake))
    except Exception:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    gray = sum(1 for p in prob_fake if GRAY_LOW <= p <= GRAY_HIGH)

    result = {
        "n": n,
        "n_valid": int(sum(1 for t in y_true if t == 0)),
        "n_hoax": int(sum(1 for t in y_true if t == 1)),
        "accuracy": float(acc),
        "accuracy_wilson_95": wilson_interval(correct, n),
        "roc_auc": auc,
        "precision_valid": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_hoax": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_valid": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_hoax": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_valid": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_hoax": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate_valid": float(_rate(int(fp), int(fp + tn))),
        "false_positive_rate_valid_wilson_95": wilson_interval(int(fp), int(fp + tn)),
        "gray_zone_pct": float(gray / n),
        "brier_score": float(brier_score_loss(y_true, prob_fake)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    return result


def per_group_metrics(y_true: list[int], y_pred: list[int], prob_fake: list[float],
                      groups: dict[str, list[str]]) -> dict:
    """Ringkasan per sumber / topik / bentuk."""
    out: dict[str, dict] = {}
    for name, values in groups.items():
        buckets: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
        for t, p, pr, value in zip(y_true, y_pred, prob_fake, values):
            buckets[str(value or "(kosong)")].append((t, p, pr))
        entries = {}
        for key, items in sorted(buckets.items()):
            yt = [x[0] for x in items]
            yp = [x[1] for x in items]
            pr = [x[2] for x in items]
            ok = sum(1 for a, b in zip(yt, yp) if a == b)
            entries[key] = {
                "n": len(items),
                "accuracy": ok / len(items),
                "accuracy_wilson_95": wilson_interval(ok, len(items)),
                "predicted_hoax_rate": _rate(sum(1 for p in yp if p == 1), len(items)),
                "mean_prob_fake": sum(pr) / len(pr),
                "n_valid": sum(1 for t in yt if t == 0),
                "n_hoax": sum(1 for t in yt if t == 1),
            }
        out[name] = entries
    return out


def compute_claim_metrics(hits: list[int], labels: list[int]) -> dict:
    """Recall hoak, false-match valid, presisi pada prevalensi 10/30/50%."""
    n_hoax = sum(1 for t in labels if t == 1)
    n_valid = sum(1 for t in labels if t == 0)
    tp = sum(1 for h, t in zip(hits, labels) if h and t == 1)
    fp = sum(1 for h, t in zip(hits, labels) if h and t == 0)
    recall = _rate(tp, n_hoax)
    fpr = _rate(fp, n_valid)
    precision_at = {}
    for prev in PREVALENCES:
        expected_tp = prev * recall
        expected_fp = (1 - prev) * fpr
        denom = expected_tp + expected_fp
        precision_at[f"{int(prev*100)}%"] = expected_tp / denom if denom else 0.0
    return {
        "n_hoax": n_hoax,
        "n_valid": n_valid,
        "matches_on_hoax": tp,
        "matches_on_valid": fp,
        "recall_hoax": recall,
        "recall_hoax_wilson_95": wilson_interval(tp, n_hoax),
        "false_match_on_valid": fpr,
        "false_match_on_valid_wilson_95": wilson_interval(fp, n_valid),
        "precision_at_prevalence": precision_at,
        "note": (
            "Presisi bergantung prevalensi dunia nyata; dihitung dari recall hoak "
            "dan false-match pada valid. Ambang 0,66 dipilih pada hold-out v2, "
            "jadi angka ini mengandung selection bias dan tidak boleh dipakai "
            "sebagai klaim produksi."
        ),
    }


def _indobert_checksums_ok() -> tuple[bool, str]:
    """IndoBERT hanya bisa dipakai bila semua berkasnya cocok dengan checksum."""
    if not os.path.isdir(INDOBERT_DIR):
        return False, "folder models/indobert/ tidak ada"
    checksums = os.path.join(INDOBERT_DIR, "checksums.txt")
    if not os.path.exists(checksums):
        return False, "models/indobert/checksums.txt tidak ada"
    verified = 0
    with open(checksums, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            expected, name = parts[0].lower(), parts[1].strip().lstrip("./")
            path = os.path.join(INDOBERT_DIR, name)
            if not os.path.exists(path):
                return False, f"berkas {name} tidak ada"
            if sha256_file(path) != expected:
                return False, f"checksum {name} tidak cocok"
            verified += 1
    if verified < 2:
        return False, "checksum terlalu sedikit"
    return True, f"{verified} berkas terverifikasi"


def check_guard(csv_path: str, validation_path: str, result_path: str,
                rerun_acknowledged: bool = False) -> tuple[bool, str, dict]:
    """Cek syarat satu pintu. Mengembalikan (lolos, alasan, info)."""
    info: dict = {}
    if not os.path.exists(csv_path):
        return False, f"hold-out v3 tidak ditemukan: {csv_path}", info
    current_hash = sha256_file(csv_path)
    info["holdout_sha256"] = current_hash
    if not os.path.exists(validation_path):
        return False, "laporan validator belum ada; jalankan validate_holdout_v3.py dulu", info
    try:
        with open(validation_path, encoding="utf-8") as fh:
            validation = json.load(fh)
    except Exception as exc:
        return False, f"laporan validator tidak terbaca: {exc}", info
    if validation.get("status") != "pass":
        return False, "validator belum lolos (status != pass)", info
    if validation.get("csv_sha256") != current_hash:
        return False, "hash CSV berubah sejak validasi; jalankan validator ulang", info
    info["validation_generated_at"] = validation.get("generated_at")

    if os.path.exists(result_path):
        try:
            with open(result_path, encoding="utf-8") as fh:
                previous = json.load(fh)
        except Exception:
            previous = {}
        if previous.get("holdout_sha256") == current_hash and not rerun_acknowledged:
            return False, (
                "evaluasi dengan hash yang sama sudah pernah dijalankan. "
                "Gunakan --rerun-acknowledged hanya bila Anda menerima risiko tuning diam-diam."
            ), info
        info["previous_result_sha256"] = previous.get("holdout_sha256")
    return True, "syarat satu pintu terpenuhi", info


def _append_attempt(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _load_tfidf_candidate(rows: list[dict]) -> dict:
    import joblib

    model_path = os.path.join(MODEL_ID_DIR, "model.pkl")
    vec_path = os.path.join(MODEL_ID_DIR, "vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return {"available": False, "reason": "artefak models/id/model.pkl atau vectorizer.pkl tidak ada"}
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    threshold_path = os.path.join(MODEL_ID_DIR, "threshold.json")
    with open(threshold_path, encoding="utf-8") as fh:
        threshold = float(json.load(fh).get("threshold", 0.5))
    meta_path = os.path.join(MODEL_ID_DIR, "metadata.json")
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    cleaned = [clean_text(r["text"]) for r in rows]
    proba = model.predict_proba(vectorizer.transform(cleaned))[:, 1].astype(float)
    pred = (proba >= threshold).astype(int)
    y_true = [int(r["label"]) for r in rows]
    groups = {
        "sumber": [r.get("nama_sumber", "") for r in rows],
        "topik": [r.get("topik", "") for r in rows],
        "bentuk": [r.get("bentuk", "") for r in rows],
    }
    metrics = compute_classification_metrics(y_true, proba.tolist(), pred.tolist())
    metrics["per_source_topic_form"] = per_group_metrics(
        y_true, pred.tolist(), proba.tolist(), groups)
    metrics["decision_threshold"] = threshold
    metrics["threshold_origin"] = "dari models/id/threshold.json (TIDAK disetel di sini)"
    return {
        "available": True,
        "actual_model": meta.get("model", "unknown"),
        "n_features": meta.get("n_features"),
        "metrics": metrics,
    }


def _load_sent_trans_candidate(rows: list[dict]) -> dict:
    model_path = os.path.join(MODEL_ID_DIR, "sent_trans_lr.pkl")
    vec_path = os.path.join(MODEL_ID_DIR, "sent_trans_vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return {
            "available": False,
            "reason": (
                "artefak Sentence-Transformer + LogisticRegression tidak disimpan "
                "oleh pipeline saat ini; skrip ini dilarang melatih model baru"
            ),
        }
    import joblib
    from sentence_transformers import SentenceTransformer

    model = joblib.load(model_path)
    vec = joblib.load(vec_path)
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = encoder.encode(
        [r["text"] for r in rows], batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    if hasattr(vec, "transform"):
        features = vec.transform(embeddings)
    else:
        features = embeddings
    proba = model.predict_proba(features)[:, 1].astype(float)
    y_true = [int(r["label"]) for r in rows]
    pred = (proba >= 0.5).astype(int)
    metrics = compute_classification_metrics(y_true, proba.tolist(), pred.tolist())
    metrics["decision_threshold"] = 0.5
    metrics["threshold_origin"] = "default (tidak ada ambang tersimpan; TIDAK disetel di sini)"
    return {"available": True, "actual_model": "SentenceTransformer + LogisticRegression",
            "metrics": metrics}


def _load_indobert_candidate(rows: list[dict]) -> dict:
    ok, reason = _indobert_checksums_ok()
    if not ok:
        return {"available": False, "reason": f"IndoBERT dilewati: {reason}"}
    import app as app_module

    y_true = [int(r["label"]) for r in rows]
    proba = []
    for row in rows:
        value = app_module.indobert_predict_proba(row["text"])
        proba.append(0.5 if value is None else float(value))
    pred = [1 if p >= 0.5 else 0 for p in proba]
    metrics = compute_classification_metrics(y_true, proba, pred)
    metrics["decision_threshold"] = 0.5
    metrics["threshold_origin"] = "default (TIDAK disetel di sini)"
    return {"available": True, "actual_model": "fine-tuned IndoBERT (checksum terverifikasi)",
            "metrics": metrics}


def _claim_matching(rows: list[dict]) -> dict:
    import app as app_module

    app_module._load_debunk_index()
    hits = []
    for row in rows:
        try:
            hits.append(1 if app_module.find_similar_debunked(row["text"]) else 0)
        except Exception:
            hits.append(0)
    labels = [int(r["label"]) for r in rows]
    result = compute_claim_metrics(hits, labels)
    result["threshold_used"] = app_module.DEBUNK_THRESHOLD
    result["index_entries"] = (
        len(app_module.HOAX_META) if app_module.HOAX_META is not None else 0
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--validation-report", default=DEFAULT_VALIDATION)
    parser.add_argument("--result", default=DEFAULT_RESULT)
    parser.add_argument("--attempts", default=DEFAULT_ATTEMPTS)
    parser.add_argument("--rerun-acknowledged", action="store_true",
                        help="Izinkan evaluasi ulang pada hash yang sama (dicatat di log)")
    args = parser.parse_args(argv)

    passed, reason, info = check_guard(
        args.csv, args.validation_report, args.result, args.rerun_acknowledged)
    attempt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "csv": os.path.relpath(args.csv, BASE_DIR),
        "hash": info.get("holdout_sha256"),
        "rerun_acknowledged": args.rerun_acknowledged,
        "outcome": "completed" if passed else "blocked",
        "reason": reason,
    }
    _append_attempt(args.attempts, attempt)
    if not passed:
        print(f"EVALUASI DITOLAK: {reason}")
        print(f"Percobaan dicatat di {args.attempts}")
        return 3 if "sudah pernah dijalankan" in reason else 2

    rows = _read_rows(args.csv)
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        source = (row.get("nama_sumber") or row.get("sumber_url") or "").strip().lower()
        source_counts[source or "(kosong)"] += 1
    holdout_profile = {
        "n": len(rows),
        "n_valid": sum(1 for r in rows if int(r["label"]) == 0),
        "n_hoax": sum(1 for r in rows if int(r["label"]) == 1),
        "n_sources": len([s for s in source_counts if s != "(kosong)"]),
        "max_source_share": (max(source_counts.values()) / len(rows)) if rows else 0.0,
        "n_topics": len({(r.get("topik") or "").strip() for r in rows}),
        "n_forms": len({(r.get("bentuk") or "").strip() for r in rows}),
    }
    candidates = {
        "tfidf_lr_id": _load_tfidf_candidate(rows),
        "sent_trans_lr": _load_sent_trans_candidate(rows),
        "indobert": _load_indobert_candidate(rows),
    }
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_csv": os.path.relpath(args.csv, BASE_DIR),
        "holdout_sha256": info["holdout_sha256"],
        "validator": {
            "status": "pass",
            "generated_at": info.get("validation_generated_at"),
        },
        "rerun_acknowledged": args.rerun_acknowledged,
        "previous_result_sha256": info.get("previous_result_sha256"),
        "holdout_profile": holdout_profile,
        "candidates": candidates,
        "claim_matching": _claim_matching(rows),
        "notes": [
            "Skrip ini tidak melatih/menyetel model atau ambang; hanya memuat artefak yang ada.",
            "Hold-out v3 dipakai sekali untuk laporan akhir; tidak untuk tuning.",
            "Beberapa kandidat tidak tersedia karena artefak tidak disimpan — lihat field reason.",
        ],
    }
    os.makedirs(os.path.dirname(args.result), exist_ok=True)
    with open(args.result, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"Evaluasi selesai: {len(rows)} baris -> {args.result}")
    for name, candidate in candidates.items():
        status = "tersedia" if candidate["available"] else "TIDAK tersedia"
        print(f"  - {name}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
