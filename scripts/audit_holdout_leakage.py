#!/usr/bin/env python3
"""TAHAP 3.1 - Deteksi kebocoran (leakage) hold-out v2 ke indeks hoak / train data."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text as id_clean_text  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

HOLDOUT_PATH = os.path.join(DATA_DIR, "holdout_id_v2.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train_id_oos.csv")
INDEX_PATH = os.path.join(MODELS_DIR, "hoax_index.npy")
META_PATH = os.path.join(MODELS_DIR, "hoax_index_meta.json")


def norm(t):
    return id_clean_text("" if not isinstance(t, str) else t)


def main():
    ho = pd.read_csv(HOLDOUT_PATH)
    ho["label"] = ho["label"].astype(int)
    ho_texts = [norm(t) for t in ho["text"].astype(str)]

    # 1. Exact overlap with train
    tr = pd.read_csv(TRAIN_PATH)
    tr["label"] = tr["label"].astype(int)
    tr_texts = set(norm(t) for t in tr["text"].astype(str))

    exact_in_train = [t for t in ho_texts if t and t in tr_texts]
    print(f"=== KEBOCORAN EXAK KE TRAIN ===")
    print(f"Hold-out v2 baris: {len(ho)}")
    print(f"Baris hold-out yang muncul EXAK di train: {len(exact_in_train)}")
    if exact_in_train:
        print("  (seharusnya 0; bila ada = kebocoran label/data)")

    # 2. Exact overlap with hoax_index_meta (judul)
    meta = json.load(open(META_PATH, encoding="utf-8"))
    meta_titles = set(norm(m.get("judul", "")) for m in meta)
    exact_in_index = [t for t in ho_texts if t and t in meta_titles]
    print(f"\n=== KEBOCORAN EXAK KE HOAX INDEX ===")
    print(f"Judul hoak di indeks: {len(meta_titles)}")
    print(f"Baris hold-out yang muncul EXAK di indeks hoak: {len(exact_in_index)}")

    # 3. Near-duplicate (cosine >= 0.9) vs train using SentenceTransformer
    print(f"\n=== NEAR-DUPLICATE (cosine >= 0.9) vs TRAIN ===")
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        # Encode hold-out + sample of train (train is 7940; encode all for safety but batch)
        h_emb = model.encode(ho_texts, batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        tr_sample = tr_texts if len(tr_texts) <= 3000 else set(list(tr_texts)[:3000])
        tr_list = list(tr_sample)
        t_emb = model.encode(tr_list, batch_size=64, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        sim = cosine_similarity(h_emb, t_emb)
        max_sim = sim.max(axis=1)
        near = int((max_sim >= 0.9).sum())
        print(f"Train yang dibandingkan: {len(tr_list)}")
        print(f"Baris hold-out dengan near-dup (cos>=0.9) di train: {near}")
        if near:
            idxs = np.where(max_sim >= 0.9)[0]
            for i in idxs[:5]:
                print(f"  hold-out #{i} label={ho.label.iloc[i]} max_sim={max_sim[i]:.3f}: {str(ho.text.iloc[i])[:90]}")
    except Exception as e:
        print(f"Tidak dapat mengecek near-dup vs train (sentence-transformers): {e}")

    # 4. Near-duplicate vs hoax index
    print(f"\n=== NEAR-DUPLICATE (cos >= 0.9) vs HOAX INDEX ===")
    try:
        index = np.load(INDEX_PATH).astype(np.float32)
        h_emb2 = model.encode(ho_texts, batch_size=32, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        sim2 = cosine_similarity(h_emb2, index)
        max_sim2 = sim2.max(axis=1)
        near2 = int((max_sim2 >= 0.9).sum())
        print(f"Hoak di indeks: {len(index)}")
        print(f"Baris hold-out dengan near-dup (cos>=0.9) di indeks hoak: {near2}")
        if near2:
            idxs = np.where(max_sim2 >= 0.9)[0]
            for i in idxs[:5]:
                print(f"  hold-out #{i} label={ho.label.iloc[i]} max_sim={max_sim2[i]:.3f}: {str(ho.text.iloc[i])[:90]}")
    except Exception as e:
        print(f"Tidak dapat mengecek near-dup vs indeks: {e}")


if __name__ == "__main__":
    main()
