#!/usr/bin/env python3
"""Prepare the Indonesian fake-news dataset.

Source
------
Pratiwi, Asmara & Rahutomo (2017), "Study of hoax news detection using naive
Bayes classifier in Indonesian language", ICTS 2017.
Mendeley Data: https://data.mendeley.com/datasets/p3hfgr5j3m/1
Licence: CC BY 4.0.

The archive contains:
  - 250 news with valid hoax label.csv
  - 600 news with valid hoax label.csv   <- superset used here
Labels were assigned by three annotators and resolved by voting.

Output
------
data/train_id.csv          text, clean_text, label, source, language
data/sample_news_id.csv    small stratified smoke-test set
logs/dataset_id_audit.json audit of labels, leakage signals, duplicates

Label convention: 0 = Valid/Real, 1 = Hoax/Fake.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections import Counter

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text, normalize_label  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/p3hfgr5j3m/files/"
    "38bfcff2-8a32-4920-9c26-4f63b5b2dad8/file_downloaded"
)
ARCHIVE_PATH = os.path.join(RAW_DIR, "mendeley_id_hoax.zip")
EXTRACT_DIR = os.path.join(RAW_DIR, "mendeley_id_hoax")
DATASET_NAME = "600 news with valid hoax label.csv"

RANDOM_STATE = 42

# Signals that can shortcut the label instead of learning content.
AUDIT_PATTERNS = {
    "url": re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE),
    "meta_hoax": re.compile(r"\b(?:hoax|hoaks|hoak)\b", re.IGNORECASE),
    "meta_debunk": re.compile(r"\b(?:debunk|cek fakta|fact check)\b", re.IGNORECASE),
    "klarifikasi": re.compile(r"\bklarifikasi\b", re.IGNORECASE),
    "baca_juga": re.compile(r"baca juga", re.IGNORECASE),
    "turnbackhoax": re.compile(r"turnbackhoax", re.IGNORECASE),
    "detik": re.compile(r"\bdetik(?:com|\.com)?\b", re.IGNORECASE),
    "kompas": re.compile(r"\bkompas(?:\.com)?\b", re.IGNORECASE),
    "tribun": re.compile(r"\btribun(?:news)?\b", re.IGNORECASE),
}


def download_archive() -> None:
    """Download the Mendeley archive if it is not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(ARCHIVE_PATH) and os.path.getsize(ARCHIVE_PATH) > 100_000:
        print(f"Using cached archive: {ARCHIVE_PATH}")
        return

    print("Downloading Mendeley archive (~578 KB)...")
    request = urllib.request.Request(
        MENDELEY_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; fake-news-detector/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with open(ARCHIVE_PATH, "wb") as handle:
        handle.write(payload)
    print(f"Saved archive: {ARCHIVE_PATH} ({len(payload):,} bytes)")


def extract_archive() -> None:
    """Extract the archive (idempotent)."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    target = os.path.join(EXTRACT_DIR, DATASET_NAME)
    if os.path.exists(target):
        print(f"Using cached CSV: {target}")
        return
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        archive.extractall(EXTRACT_DIR)
    if not os.path.exists(target):
        raise FileNotFoundError(f"{DATASET_NAME} not found in archive")
    print(f"Extracted to: {EXTRACT_DIR}")


def parse_mendeley_csv(path: str) -> list[dict]:
    """Parse the semicolon-separated, latin-1, multi-line-quoted CSV."""
    rows: list[dict] = []
    with open(path, encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        header = next(reader)
        if len(header) < 2:
            raise ValueError(f"Unexpected header: {header!r}")
        for raw in reader:
            if len(raw) < 2 or not raw[0].strip():
                continue
            text = raw[0].strip()
            label = normalize_label(raw[1])
            rows.append({"text": text, "label": label})
    return rows


def audit(rows: list[dict], label: str, text_key: str = "text") -> dict:
    """Compute label balance, leakage-signal prevalence, and length stats."""
    total = len(rows)
    by_label = Counter(row["label"] for row in rows)
    result = {
        "label": label,
        "total_rows": total,
        "label_counts": {
            "valid_real_0": by_label.get(0, 0),
            "hoax_fake_1": by_label.get(1, 0),
        },
    }

    prevalence = {}
    for name, pattern in AUDIT_PATTERNS.items():
        entry = {}
        for code, key in ((0, "valid_real"), (1, "hoax_fake")):
            subset = [r for r in rows if r["label"] == code]
            hits = sum(1 for r in subset if pattern.search(r[text_key]))
            entry[key] = {
                "count": hits,
                "pct": round(100 * hits / len(subset), 2) if subset else 0.0,
            }
        prevalence[name] = entry
    result["signal_prevalence_pct"] = prevalence

    for code, key in ((0, "valid_real"), (1, "hoax_fake")):
        lengths = sorted(len(r[text_key]) for r in rows if r["label"] == code)
        if lengths:
            result[key] = {
                "length_median": lengths[len(lengths) // 2],
                "length_min": lengths[0],
                "length_max": lengths[-1],
            }

    exact = Counter(r[text_key] for r in rows)
    result["exact_duplicate_texts"] = sum(1 for count in exact.values() if count > 1)
    result["unique_texts"] = len(exact)

    conflicts = {}
    for row in rows:
        conflicts.setdefault(row[text_key], set()).add(row["label"])
    result["conflicting_label_texts"] = sum(
        1 for labels in conflicts.values() if len(labels) > 1
    )
    return result


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    download_archive()
    extract_archive()

    csv_path = os.path.join(EXTRACT_DIR, DATASET_NAME)
    print(f"Parsing: {csv_path}")
    rows = parse_mendeley_csv(csv_path)
    print(f"Parsed {len(rows):,} rows")

    before = audit(rows, "raw")

    # Clean + deduplicate.
    cleaned_rows = []
    for row in rows:
        cleaned = clean_text(row["text"])
        if len(cleaned) < 50:
            continue
        cleaned_rows.append(
            {
                "text": row["text"],
                "clean_text": cleaned,
                "label": row["label"],
                "source": "Mendeley/ICTS-2017",
                "language": "id",
            }
        )

    deduped = []
    seen = set()
    for row in cleaned_rows:
        if row["clean_text"] in seen:
            continue
        seen.add(row["clean_text"])
        deduped.append(row)

    after = audit(deduped, "clean_deduped", text_key="clean_text")

    df = pd.DataFrame(deduped)
    train_path = os.path.join(DATA_DIR, "train_id.csv")
    df.to_csv(train_path, index=False, encoding="utf-8")
    print(f"Saved: {train_path} ({len(df):,} rows)")

    # Small stratified smoke-test sample (30 real + 30 fake, if available).
    sample_parts = []
    for label, size in ((0, 30), (1, 30)):
        subset = df[df["label"] == label]
        if len(subset):
            sample_parts.append(subset.sample(n=min(size, len(subset)), random_state=RANDOM_STATE))
    sample_df = pd.concat(sample_parts).sample(frac=1, random_state=RANDOM_STATE)
    sample_path = os.path.join(DATA_DIR, "sample_news_id.csv")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    print(f"Saved: {sample_path} ({len(sample_df):,} rows)")

    audit_data = {
        "source": "Mendeley p3hfgr5j3m / ICTS 2017 (Pratiwi et al.)",
        "licence": "CC BY 4.0",
        "raw": before,
        "clean_deduped": after,
        "cleaning": [
            "remove URLs, HTML, e-mail addresses",
            "remove portal boilerplate (baca juga, editor, penulis, tags, sumber)",
            "remove agency markers (detikcom, kompas, tribun, ...)",
            "remove label-leaking meta words (hoax, hoaks, fake news, debunk, ...)",
            "lowercase, keep a-z only, collapse whitespace",
            "deduplicate on cleaned text",
        ],
        "label_mapping": {"Valid": 0, "Hoax": 1},
        "random_state": RANDOM_STATE,
    }
    audit_path = os.path.join(LOGS_DIR, "dataset_id_audit.json")
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit_data, handle, indent=2, ensure_ascii=False)
    print(f"Saved audit: {audit_path}")

    print("\nSummary")
    print(f"  Total clean rows : {len(df):,}")
    print(f"  Valid/Real (0)   : {(df['label'] == 0).sum():,}")
    print(f"  Hoax/Fake (1)    : {(df['label'] == 1).sum():,}")
    print("  Signal prevalence (raw -> clean):")
    for name in ("meta_hoax", "meta_debunk", "turnbackhoax", "url", "baca_juga"):
        raw_pct = before["signal_prevalence_pct"][name]
        clean_pct = after["signal_prevalence_pct"][name]
        print(
            f"    {name:<14} fake {raw_pct['hoax_fake']['pct']:5.1f}% -> "
            f"{clean_pct['hoax_fake']['pct']:4.1f}% | "
            f"real {raw_pct['valid_real']['pct']:5.1f}% -> "
            f"{clean_pct['valid_real']['pct']:4.1f}%"
        )


if __name__ == "__main__":
    main()
