#!/usr/bin/env python3
"""Validator untuk data/holdout_id_v3.csv — TIDAK mengubah data.

Memeriksa:
- kolom wajib, label 0/1, panjang teks 10-600 karakter, ID unik;
- duplikat dan near-duplicate (cosine >= 0.90) terhadap data latih, hold-out v2,
  indeks judul hoak, dan di dalam set itu sendiri;
- sebaran sumber (minimum 8 sumber per kelas, maksimal 20% per sumber),
  topik, bentuk, rentang tanggal (>= 12 bulan), dan keseimbangan kelas;
- bukti_url wajib untuk label hoaks.

Keluaran: laporan JSON (default logs/holdout_v3_validation.json) dan exit code
0 = lolos, 1 = ada pelanggaran. Skrip ini tidak pernah menulis ke CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text  # noqa: E402

REQUIRED_COLUMNS = [
    "id", "text", "label", "sumber_url", "nama_sumber", "tanggal_terbit",
    "topik", "bentuk", "bukti_url", "catatan",
]
OPTIONAL_COLUMNS = ["label_anotator1", "label_anotator2"]

MIN_ROWS = 100
MIN_SOURCES_PER_CLASS = 8
MAX_SOURCE_SHARE = 0.20
MIN_DATE_SPAN_DAYS = 365
MIN_TEXT_CHARS = 10
MAX_TEXT_CHARS = 600
NEAR_DUP_COSINE = 0.90

REQUIRED_TOPICS = {"politik", "kesehatan", "ekonomi", "bencana", "penipuan_daring"}
REQUIRED_FORMS = {"judul", "paragraf", "artikel", "pesan_berantai"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _valid_url(value: str) -> bool:
    value = (value or "").strip()
    return bool(URL_RE.match(value) and urlparse(value).scheme in ("http", "https"))


def _label_of(value) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    if text in ("0", "1"):
        return int(text)
    return None


def _source_of(row: dict) -> str:
    name = (row.get("nama_sumber") or "").strip().lower()
    if name:
        return name
    url = (row.get("sumber_url") or "").strip()
    return (urlparse(url).netloc or "tanpa_sumber").lower()


def _load_reference_texts(base_dir: str) -> list[tuple[str, str]]:
    """(label referensi, teks) dari data latih, hold-out v2, indeks hoak."""
    refs: list[tuple[str, str]] = []
    for name, column in (
        ("train_id.csv", "text"),
        ("holdout_id_v2.csv", "text"),
    ):
        path = os.path.join(base_dir, "data", name)
        if os.path.exists(path):
            try:
                for row in read_csv(path):
                    text = clean_text(row.get(column, "") or row.get("clean_text", ""))
                    if len(text) >= 10:
                        refs.append((name, text))
            except Exception:
                pass
    meta_path = os.path.join(base_dir, "models", "hoax_index_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            for entry in meta:
                title = clean_text(entry.get("judul", ""))
                if len(title) >= 10:
                    refs.append(("hoax_index_meta.json", title))
        except Exception:
            pass
    return refs


def _cosine_near_duplicates(texts: list[str], refs: list[tuple[str, str]]):
    """Kembalikan (indeks_baris, label_ref, similarity) untuk cosine >= 0.90."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ref_texts = [text for _, text in refs]
    if not texts or not ref_texts:
        return []
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=200000,
    )
    matrix = vectorizer.fit_transform(texts + ref_texts)
    query = matrix[: len(texts)]
    reference = matrix[len(texts) :]
    found = []
    for i in range(len(texts)):
        sims = (query[i] @ reference.T).toarray().ravel()
        j = int(sims.argmax())
        if sims[j] >= NEAR_DUP_COSINE:
            found.append((i, refs[j][0], round(float(sims[j]), 4)))
    return found


def validate(rows: list[dict], base_dir: str = BASE_DIR,
             check_external: bool = True) -> dict:
    """Validasi baris; mengembalikan laporan dict. Tidak mengubah `rows`."""
    checks: dict[str, dict] = {}

    def add(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    if not rows:
        add("rows", False, "Tidak ada baris.")
        return {"status": "fail", "checks": checks,
                "violations": ["rows: tidak ada baris"], "n_rows": 0}

    available = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLUMNS if c not in available]
    add("columns", not missing,
        "semua kolom wajib ada" if not missing else f"kolom hilang: {missing}")
    if missing:
        return {"status": "fail", "checks": checks,
                "violations": [f"columns: {missing}"], "n_rows": len(rows)}

    ids = [(r.get("id") or "").strip() for r in rows]
    add("id_unique", len(set(ids)) == len(ids),
        f"{len(set(ids))} ID unik dari {len(ids)} baris")

    labels = [_label_of(r.get("label")) for r in rows]
    bad_labels = [i + 2 for i, lab in enumerate(labels) if lab is None]
    add("labels", not bad_labels,
        "semua label 0/1" if not bad_labels else f"baris CSV dengan label invalid: {bad_labels[:10]}")

    texts_clean = [clean_text(r.get("text", "")) for r in rows]
    raw_lengths = [len((r.get("text") or "")) for r in rows]
    bad_len = [i + 2 for i, n in enumerate(raw_lengths)
               if n < MIN_TEXT_CHARS or n > MAX_TEXT_CHARS]
    add("text_length", not bad_len,
        f"panjang {min(raw_lengths)}–{max(raw_lengths)} karakter (batas {MIN_TEXT_CHARS}–{MAX_TEXT_CHARS})"
        if not bad_len else f"baris dengan panjang di luar batas: {bad_len[:10]}")

    n0 = sum(1 for lab in labels if lab == 0)
    n1 = sum(1 for lab in labels if lab == 1)
    balanced = len(rows) >= MIN_ROWS and abs(n0 - n1) <= 0.2 * len(rows)
    add("class_balance", balanced,
        f"n={len(rows)}, valid={n0}, hoaks={n1}, selisih={abs(n0-n1)} "
        f"(maks {0.2*len(rows):.0f}; minimal {MIN_ROWS} baris)")

    sources = [_source_of(r) for r in rows]
    src0 = {s for s, lab in zip(sources, labels) if lab == 0}
    src1 = {s for s, lab in zip(sources, labels) if lab == 1}
    src_ok = len(src0) >= MIN_SOURCES_PER_CLASS and len(src1) >= MIN_SOURCES_PER_CLASS
    add("sources_per_class", src_ok,
        f"sumber Valid={len(src0)}, Hoaks={len(src1)} (minimal {MIN_SOURCES_PER_CLASS} tiap kelas)")

    counts = Counter(sources)
    top_source, top_n = counts.most_common(1)[0] if counts else ("", 0)
    share = top_n / len(rows)
    add("source_share", share <= MAX_SOURCE_SHARE,
        f"sumber terbesar '{top_source}' = {top_n}/{len(rows)} = {share:.1%} "
        f"(maks {MAX_SOURCE_SHARE:.0%})")

    dates = []
    bad_dates = []
    for i, r in enumerate(rows):
        value = (r.get("tanggal_terbit") or "").strip()
        if not DATE_RE.match(value):
            bad_dates.append(i + 2)
            continue
        try:
            dates.append(datetime.strptime(value, "%Y-%m-%d"))
        except ValueError:
            bad_dates.append(i + 2)
    span = (max(dates) - min(dates)).days if len(dates) >= 2 else 0
    add("date_span", span >= MIN_DATE_SPAN_DAYS,
        f"{len(dates)} tanggal valid, rentang {span} hari (minimal {MIN_DATE_SPAN_DAYS} hari); "
        f"tanggal invalid: {len(bad_dates)}")

    topics = {(r.get("topik") or "").strip().lower().replace(" ", "_") for r in rows}
    missing_topics = sorted(REQUIRED_TOPICS - topics)
    add("topics", not missing_topics,
        "kelima topik hadir" if not missing_topics else f"topik kurang: {missing_topics}")

    forms = {(r.get("bentuk") or "").strip().lower() for r in rows}
    missing_forms = sorted(REQUIRED_FORMS - forms)
    add("forms", not missing_forms,
        "keempat bentuk hadir" if not missing_forms else f"bentuk kurang: {missing_forms}")

    bad_bukti = [i + 2 for i, (lab, r) in enumerate(zip(labels, rows))
                 if lab == 1 and not _valid_url(r.get("bukti_url", ""))]
    add("bukti_for_hoax", not bad_bukti,
        "semua hoaks punya bukti_url http/https" if not bad_bukti
        else f"hoaks tanpa bukti_url valid: {bad_bukti[:10]}")

    if "label_anotator1" in available and "label_anotator2" in available:
        bad_ann = [
            i + 2 for i, r in enumerate(rows)
            if ((_label_of(r.get("label_anotator1")) is None
                 and (r.get("label_anotator1") or "").strip())
                or (_label_of(r.get("label_anotator2")) is None
                    and (r.get("label_anotator2") or "").strip()))
        ]
        add("annotator_columns", not bad_ann,
            "kolom anotator berisi 0/1 atau kosong" if not bad_ann
            else f"nilai anotator invalid di baris: {bad_ann[:10]}")

    exact = Counter(t for t in texts_clean if t)
    duplicates = [t[:60] for t, c in exact.items() if c > 1]
    add("exact_duplicates", not duplicates,
        "tidak ada duplikat teks" if not duplicates
        else f"{len(duplicates)} teks duplikat, contoh: {duplicates[:3]}")

    if check_external:
        refs = _load_reference_texts(base_dir)
        found = _cosine_near_duplicates(texts_clean, refs)
    else:
        refs, found = [], []
    leakage = [{"row_id": ids[i], "reference": ref, "cosine": sim}
               for i, ref, sim in found]
    add("near_duplicate_external", not leakage,
        f"{len(refs)} teks referensi diperiksa; 0 kecocokan cosine >= {NEAR_DUP_COSINE}"
        if not leakage else f"{len(leakage)} baris mirip referensi: {leakage[:5]}")

    violations = [f"{name}: {info['detail']}" for name, info in checks.items()
                  if not info["ok"]]
    return {
        "status": "pass" if not violations else "fail",
        "n_rows": len(rows),
        "class_counts": {"valid": n0, "hoax": n1},
        "source_counts": dict(counts),
        "checks": checks,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(BASE_DIR, "data", "holdout_id_v3.csv"))
    parser.add_argument("--report", default=os.path.join(BASE_DIR, "logs", "holdout_v3_validation.json"))
    parser.add_argument("--no-external", action="store_true",
                        help="Lewati pengecekan kemiripan ke data latih/indeks hoak")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        report = {
            "status": "fail",
            "violations": [f"csv: berkas tidak ditemukan: {args.csv}"],
            "n_rows": 0,
            "checks": {},
        }
    else:
        rows = read_csv(args.csv)
        report = validate(rows, base_dir=BASE_DIR, check_external=not args.no_external)
        report["csv_path"] = os.path.relpath(args.csv, BASE_DIR)
        report["csv_sha256"] = sha256_file(args.csv)

    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"Hold-out v3: {report['status'].upper()} — "
          f"{report.get('n_rows', 0)} baris; {len(report.get('violations', []))} pelanggaran")
    for violation in report.get("violations", []):
        print(f"  - {violation[:300]}")
    print(f"Laporan: {args.report}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
