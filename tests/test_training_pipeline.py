"""Tes pipeline training & split dengan data SINTETIS (Butir 5).

Tidak ada tes yang menyentuh data latih/hold-out asli dan tidak ada model
yang disimpan ke disk. Semua data dibuat di dalam tes ini.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
for path in (BASE_DIR, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import prepare_dataset_id as prep  # noqa: E402
import train_model_id as trainer  # noqa: E402


# ---------------------------------------------------------------------------
# Data sintetis kecil & deterministik
# ---------------------------------------------------------------------------
SOURCES = ["portal_a", "portal_b", "portal_c", "portal_d", "portal_e", "portal_f"]
VALID_TOKENS = "pemerintah resmi mengumumkan kebijakan kesehatan ekonomi nasional"
HOAX_TOKENS = "konspirasi menyesatkan hoaks palsu bohong kabar dusta"


def _synthetic_rows(n_per_source=10):
    rows = []
    for src_idx, source in enumerate(SOURCES):
        for i in range(n_per_source):
            label = i % 2
            text = (VALID_TOKENS if label == 0 else HOAX_TOKENS) + f" sumber {source}"
            rows.append({
                "text": text,
                "label": label,
                "source": source,
            })
    return rows


# ---------------------------------------------------------------------------
# GroupKFold by-source (default)
# ---------------------------------------------------------------------------
def test_group_kfold_tidak_ada_sumber_yang_bocor(capsys):
    rows = _synthetic_rows(n_per_source=10)
    train, test, info = prep.group_kfold_split(rows, test_size=0.2)

    train_sources = {r["source"] for r in train}
    test_sources = {r["source"] for r in test}
    assert train_sources.isdisjoint(test_sources)
    assert info["method"] == "GroupKFold"
    assert info["n_train"] == len(train)
    assert info["n_test"] == len(test)
    assert len(train) + len(test) == len(rows)
    assert "[GroupKFold split]" in capsys.readouterr().out


def test_group_kfold_deterministik(capsys):
    rows = _synthetic_rows(n_per_source=10)
    a, _, _ = prep.group_kfold_split(rows, test_size=0.2)
    b, _, _ = prep.group_kfold_split(rows, test_size=0.2)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_group_kfold_menolak_data_tanpa_kolom_source():
    rows = [{"text": x, "label": 0} for x in VALID_TOKENS.split()]
    with pytest.raises(ValueError, match="source"):
        prep.group_kfold_split(rows)


def test_random_split_memberi_peringatan(capsys):
    rows = _synthetic_rows(n_per_source=10)
    train, test, info = prep.group_kfold_split(rows, test_size=0.2, random_split=True)
    assert info["method"] == "random"
    assert info["warning"] == "source leakage possible"
    assert "PERINGATAN" in capsys.readouterr().out
    assert len(train) + len(test) == len(rows)


# ---------------------------------------------------------------------------
# Pipeline TF-IDF + model kecil
# ---------------------------------------------------------------------------
def _fit_small_model():
    rows = _synthetic_rows(n_per_source=10)
    texts = pd.Series([r["text"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    vectorizer = trainer.fit_vectorizer(texts)
    X = vectorizer.transform(texts)
    model = trainer.build_model()
    model.fit(X, labels)
    probabilities = model.predict_proba(X)[:, 1]
    return vectorizer, model, labels, probabilities


def test_fit_vectorizer_menghasilkan_vocabulary():
    rows = _synthetic_rows(n_per_source=10)
    vectorizer = trainer.fit_vectorizer(pd.Series([r["text"] for r in rows]))
    assert len(vectorizer.get_feature_names_out()) > 0
    # Kata negasi/umum dari Sastrawi tidak boleh masuk vocabulary.
    names = set(vectorizer.get_feature_names_out())
    assert "yang" not in names
    assert "dan" not in names


def test_optimize_threshold_dalam_rentang():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.3, 0.45, 0.55, 0.7, 0.9])
    threshold, f1 = trainer.optimize_threshold(y_true, proba)
    assert 0.05 <= threshold <= 0.95
    assert 0.0 <= f1 <= 1.0


def test_extract_feature_weights_menghasilkan_weight_signifikan():
    vectorizer, model, _, _ = _fit_small_model()
    weights = trainer.extract_feature_weights(model, vectorizer)
    assert isinstance(weights, dict)
    assert len(weights) > 0
    assert all(abs(v) >= 0.01 for v in weights.values())


def test_bootstrap_ci_mengandung_estimasi_titik():
    y_true = np.array([0, 1] * 30)
    y_pred = np.array([0, 1] * 30)
    from sklearn.metrics import accuracy_score

    lo, hi = trainer.bootstrap_ci(y_true, y_pred, accuracy_score, n_boot=100)
    assert lo <= 1.0 <= hi
    assert 0.0 <= lo <= hi <= 1.0
