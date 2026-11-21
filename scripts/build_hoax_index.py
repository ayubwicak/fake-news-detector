#!/usr/bin/env python3
"""TAHAP B.1 - Bangun indeks embedding klaim hoaks yang sudah dibantah.

Sumber: data/raw/rifky_indonesian_hoax.parquet (Hugging Face Rifky/indonesian-hoax-news,
17.806 judul dari turnbackhoax.id, setiap judul berawalan [SALAH] atau [BENAR]).

Hanya baris berlabel hoaks (awalan [SALAH]) yang diindeks. Tidak menyimpan teks penuh
artikel; hanya judul, tautan sumber, dan ringkasan pendek (judul difilter awalan).

Output:
  - models/hoax_index.npy        : embedding (float32) dari judul yang telah dibersihkan
  - models/hoax_index_meta.json  : daftar {judul, url, tanggal, summary}
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

PARQUET_PATH = os.path.join(DATA_DIR, "raw", "rifky_indonesian_hoax.parquet")
INDEX_PATH = os.path.join(MODELS_DIR, "hoax_index.npy")
META_PATH = os.path.join(MODELS_DIR, "hoax_index_meta.json")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.65  # akan disesuaikan pada TAHAP B.2 (validation)

# Awalan yang menandai klaim hoaks di Rifky/turnbackhoax.id
HOAX_PREFIX_RE = re.compile(r"^\s*\[(?:SALAH|HOAX|FALSE)\b", re.IGNORECASE)


def strip_prefix(title: str) -> str:
    """Hapus awalan [SALAH]/[HOAX]/[BENAR] dll. dari judul untuk ringkasan pendek."""
    t = re.sub(r"^\s*\[[^\]]*\]\s*–?\s*", "", title).strip()
    t = re.sub(r"\s*–\s*TurnBackHoax\.ID\s*$", "", t, flags=re.IGNORECASE)
    return t


def extract_date(url: str) -> str:
    """Coba ambil tanggal dari URL (format .../YYYY/MM/DD/...) atau kosong."""
    m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", url)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return ""


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"Parquet tidak ditemukan: {PARQUET_PATH}")
        print("Jalankan dulu scripts/prepare_dataset_id.py untuk mengunduh data Rifky.")
        return

    print(f"Memuat {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Total baris: {len(df):,}")

    # Filter hanya hoak (awalan [SALAH] / [HOAX] / [FALSE])
    hoax = df[df["title"].astype(str).str.match(HOAX_PREFIX_RE)].copy()
    print(f"Baris berlabel hoak: {len(hoax):,}")

    titles = hoax["title"].astype(str).tolist()
    urls = hoax["url"].astype(str).tolist()

    summaries = [strip_prefix(t) for t in titles]
    dates = [extract_date(u) for u in urls]

    # Embedding judul (bukan teks penuh artikel)
    print(f"Memuat model embeddings {EMBED_MODEL} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        titles, batch_size=64, convert_to_numpy=True, show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Embedding shape: {embeddings.shape}")

    # Simpan indeks
    np.save(INDEX_PATH, embeddings.astype(np.float32))

    meta = []
    for t, u, s, d in zip(titles, urls, summaries, dates):
        meta.append(
            {
                "judul": t,
                "url": u,
                "tanggal": d,
                "summary": s,
            }
        )
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDisimpan:")
    print(f"  {INDEX_PATH}  ({embeddings.shape[0]:,} vektor x {embeddings.shape[1]} dimensi)")
    print(f"  {META_PATH}  ({len(meta)} entri)")
    print(f"\nCatatan: hanya judul + tautan disimpan (tidak ada teks penuh artikel).")


if __name__ == "__main__":
    main()
