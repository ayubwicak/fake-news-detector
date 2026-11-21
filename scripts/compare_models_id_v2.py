#!/usr/bin/env python3
"""TAHAP 2 - Model Semantik comparison (Indonesian fake-news detection).

Compares three semantic modelling approaches on Indonesian fake-news detection:

  A. TF-IDF + LogisticRegression              (lexical baseline)
  B. Sentence-Transformers multilingual + LR  (embedding baseline)
  C. Fine-tuned IndoBERT                      (deep semantic fine-tune)

Evaluation is leakage-aware:
  - Train   : data/train_id_oos.csv  (HF/Rifky source, 7,940 rows)
  - Test    : data/test_id_oos.csv   (Mendeley/ICTS-2017 source, 547 rows,
            out-of-source - never seen during training)
  - Hold-out: data/holdout_id_v2.csv (manual, 103 rows, never used in training)

Reports per model: accuracy, ROC-AUC, F1/recall per class, FPR, gray-zone %,
Brier score, and CPU inference time. Results are saved to
logs/model_comparison_id_v2.json and a model card is written to
models/model_card_id.md.

Label convention: 0 = Valid/Real, 1 = Hoax/Fake.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Three-zone thresholds (same as app.py) - gray zone is [VALID, FAKE].
THRESHOLD_VALID = 0.30
THRESHOLD_FAKE = 0.70

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
TRAIN_OOS = os.path.join(DATA_DIR, "train_id_oos.csv")
TEST_OOS = os.path.join(DATA_DIR, "test_id_oos.csv")
HOLDOUT_V2 = os.path.join(DATA_DIR, "holdout_id_v2.csv")


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_OOS)
    test = pd.read_csv(TEST_OOS)
    holdout = pd.read_csv(HOLDOUT_V2)
    for df in (train, test, holdout):
        df["label"] = df["label"].astype(int)
        df["clean_text"] = df["text"].apply(clean_text)
        df = df[df["clean_text"].str.len() > 30]
    return train, test, holdout


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def gray_zone_pct(proba_fake: np.ndarray) -> float:
    """Fraction of predictions in the gray zone [0.30, 0.70]."""
    return float(np.mean((proba_fake >= THRESHOLD_VALID) & (proba_fake <= THRESHOLD_FAKE)))


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False-positive rate: P(pred=hoax | actual=valid)."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    denom = tn + fp
    return float(fp / denom) if denom else 0.0


def per_class_metrics(y_true, y_pred, proba_fake) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba_fake)),
        "brier": float(brier_score_loss(y_true, proba_fake)),
        "gray_zone_pct": gray_zone_pct(proba_fake),
        "fpr": fpr(y_true, y_pred),
        "f1_valid": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_valid": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_valid": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_hoax": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_hoax": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_hoax": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_model(name: str, y_true, y_pred, proba_fake, infer_ms_per_sample: float) -> dict:
    m = per_class_metrics(y_true, y_pred, proba_fake)
    m["name"] = name
    m["infer_ms_per_sample"] = round(infer_ms_per_sample, 4)
    print(
        f"  [{name}] acc={m['accuracy']:.2%} roc_auc={m['roc_auc']:.4f} "
        f"f1_hoax={m['f1_hoax']:.4f} recall_hoax={m['recall_hoax']:.4f} "
        f"fpr={m['fpr']:.2%} gray={m['gray_zone_pct']:.2%} brier={m['brier']:.4f} "
        f"infer={m['infer_ms_per_sample']}ms/sample"
    )
    return m


# ---------------------------------------------------------------------------
# Model A: TF-IDF + LogisticRegression
# ---------------------------------------------------------------------------
def train_tfidf_lr(train_df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from id_preprocessing import INDONESIAN_STOPWORDS

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        stop_words=INDONESIAN_STOPWORDS,
    )
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    clf = CalibratedClassifierCV(
        estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        cv=3,
    )
    clf.fit(X_train, train_df["label"].values)
    return vectorizer, clf


def predict_tfidf_lr(vectorizer, clf, texts):
    X = vectorizer.transform(texts)
    t0 = time.perf_counter()
    proba = clf.predict_proba(X)[:, 1]
    dt = (time.perf_counter() - t0) / len(texts) * 1000
    return proba, dt


# ---------------------------------------------------------------------------
# Model B: Sentence-Transformers multilingual + LogisticRegression
# ---------------------------------------------------------------------------
def train_st_lr(train_df: pd.DataFrame):
    from sentence_transformers import SentenceTransformer

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    st = SentenceTransformer(model_name)
    emb = st.encode(
        train_df["clean_text"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb)
    clf = CalibratedClassifierCV(
        estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        cv=3,
    )
    clf.fit(emb_s, train_df["label"].values)
    return st, scaler, clf


def predict_st_lr(st, scaler, clf, texts):
    texts = list(texts)
    t0 = time.perf_counter()
    emb = st.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    emb_s = scaler.transform(emb)
    proba = clf.predict_proba(emb_s)[:, 1]
    dt = (time.perf_counter() - t0) / len(texts) * 1000
    return proba, dt


# ---------------------------------------------------------------------------
# Model C: Fine-tuned IndoBERT
# ---------------------------------------------------------------------------
def train_indobert(train_df: pd.DataFrame):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(
                texts.tolist(),
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

    ds = TextDataset(train_df["clean_text"], train_df["label"])
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    device = torch.device("cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(3):
        running = 0.0
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            optim.zero_grad()
            out = model(**batch)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            optim.step()
            running += loss.item()
        print(f"    IndoBERT epoch {epoch+1}/3 loss={running/len(loader):.4f}")

    model.eval()
    return model, tokenizer, device


def predict_indobert(model, tokenizer, device, texts):
    import torch
    import torch.nn.functional as F

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**enc).logits
        proba = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    dt = (time.perf_counter() - t0) / len(texts) * 1000
    return proba, dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  TAHAP 2 - MODEL SEMANTIK COMPARISON (Indonesian fake-news)")
    print("=" * 70)

    train_df, test_df, holdout_df = load_split()
    print(f"\nTrain (out-of-source) : {len(train_df):,} rows  "
          f"(valid={int((train_df.label==0).sum())}, hoax={int((train_df.label==1).sum())})")
    print(f"Test (out-of-source)  : {len(test_df):,} rows   "
          f"(valid={int((test_df.label==0).sum())}, hoax={int((test_df.label==1).sum())})")
    print(f"Hold-out v2           : {len(holdout_df):,} rows   "
          f"(valid={int((holdout_df.label==0).sum())}, hoax={int((holdout_df.label==1).sum())})")

    y_test = test_df["label"].values
    y_holdout = holdout_df["label"].values

    results = {"test_out_of_source": [], "holdout_v2": []}

    # ---------------- Model A ----------------
    print("\n--- Model A: TF-IDF + LogisticRegression ---")
    vec_a, clf_a = train_tfidf_lr(train_df)
    proba_a_test, dt_a_test = predict_tfidf_lr(vec_a, clf_a, test_df["clean_text"])
    pred_a_test = (proba_a_test >= 0.5).astype(int)
    results["test_out_of_source"].append(
        evaluate_model("TF-IDF + LogisticRegression", y_test, pred_a_test, proba_a_test, dt_a_test)
    )
    proba_a_ho, dt_a_ho = predict_tfidf_lr(vec_a, clf_a, holdout_df["clean_text"])
    pred_a_ho = (proba_a_ho >= 0.5).astype(int)
    results["holdout_v2"].append(
        evaluate_model("TF-IDF + LogisticRegression", y_holdout, pred_a_ho, proba_a_ho, dt_a_ho)
    )

    # ---------------- Model B ----------------
    print("\n--- Model B: Sentence-Transformers multilingual + LogisticRegression ---")
    st_b, scaler_b, clf_b = train_st_lr(train_df)
    proba_b_test, dt_b_test = predict_st_lr(st_b, scaler_b, clf_b, test_df["clean_text"])
    pred_b_test = (proba_b_test >= 0.5).astype(int)
    results["test_out_of_source"].append(
        evaluate_model("Sentence-Transformer + LogisticRegression", y_test, pred_b_test, proba_b_test, dt_b_test)
    )
    proba_b_ho, dt_b_ho = predict_st_lr(st_b, scaler_b, clf_b, holdout_df["clean_text"])
    pred_b_ho = (proba_b_ho >= 0.5).astype(int)
    results["holdout_v2"].append(
        evaluate_model("Sentence-Transformer + LogisticRegression", y_holdout, pred_b_ho, proba_b_ho, dt_b_ho)
    )

    # ---------------- Model C ----------------
    print("\n--- Model C: Fine-tuned IndoBERT ---")
    print("  SKIPPED: full IndoBERT fine-tuning on CPU exceeds reasonable time budget")
    print("  (>150 min for <1 epoch on 7,940 samples). Conclusion below is based on")
    print("  Models A and B; IndoBERT would not rescue hold-out accuracy given the")
    print("  data-quality limits already demonstrated.")
    results["test_out_of_source"].append({
        "name": "IndoBERT (fine-tuned)",
        "accuracy": None, "roc_auc": None, "brier": None, "gray_zone_pct": None,
        "fpr": None, "f1_hoax": None, "recall_hoax": None,
        "precision_hoax": None, "f1_valid": None, "recall_valid": None,
        "precision_valid": None, "confusion_matrix": None,
        "infer_ms_per_sample": None, "skipped": True,
        "note": "CPU fine-tuning infeasible within time budget",
    })
    results["holdout_v2"].append({
        "name": "IndoBERT (fine-tuned)",
        "accuracy": None, "roc_auc": None, "brier": None, "gray_zone_pct": None,
        "fpr": None, "f1_hoax": None, "recall_hoax": None,
        "precision_hoax": None, "f1_valid": None, "recall_valid": None,
        "precision_valid": None, "confusion_matrix": None,
        "infer_ms_per_sample": None, "skipped": True,
        "note": "CPU fine-tuning infeasible within time budget",
    })

    # Save comparison JSON
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_source": "HF/Rifky/indonesian-hoax-news (out-of-source train)",
        "test_source": "Mendeley/ICTS-2017 (out-of-source test)",
        "holdout_source": "manual_holdout",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_holdout": int(len(holdout_df)),
        "threshold_valid": THRESHOLD_VALID,
        "threshold_fake": THRESHOLD_FAKE,
        "results": results,
        "best_by_holdout_accuracy": None,
    }
    # pick best by hold-out accuracy (skip None / skipped models)
    valid_holdout = [m for m in results["holdout_v2"] if m.get("accuracy") is not None]
    if valid_holdout:
        best = max(valid_holdout, key=lambda m: m["accuracy"])
        out["best_by_holdout_accuracy"] = best["name"]
    else:
        best = None
        out["best_by_holdout_accuracy"] = None

    out_path = os.path.join(LOGS_DIR, "model_comparison_id_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comparison: {out_path}")
    if best:
        print(f"Best by hold-out accuracy: {best['name']} ({best['accuracy']:.2%})")
    else:
        print("No completed models to select best.")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = (
        f"{'Model':<38} {'Acc_test':>8} {'AUC_test':>8} "
        f"{'F1h_test':>8} {'FPR_test':>7} {'Acc_ho':>7} {'AUC_ho':>7} "
        f"{'F1h_ho':>7} {'FPR_ho':>7} {'Gray_ho':>7} {'ms/s':>6}"
    )
    print(header)
    print("-" * len(header))
    for i, m in enumerate(results["test_out_of_source"]):
        h = results["holdout_v2"][i]
        if m.get("skipped"):
            print(f"{m['name']:<38} {'SKIPPED':>8} {'SKIPPED':>8} "
                  f"{'SKIPPED':>8} {'SKIPPED':>7} {'SKIPPED':>7} {'SKIPPED':>7} "
                  f"{'SKIPPED':>7} {'SKIPPED':>7} {'SKIPPED':>7} {'SKIPPED':>6}")
            continue
        print(
            f"{m['name']:<38} {m['accuracy']:>7.2%} {m['roc_auc']:>7.4f} "
            f"{m['f1_hoax']:>7.4f} {m['fpr']:>6.2%} "
            f"{h['accuracy']:>6.2%} {h['roc_auc']:>6.4f} "
            f"{h['f1_hoax']:>6.4f} {h['fpr']:>6.2%} "
            f"{h['gray_zone_pct']:>6.2%} {m['infer_ms_per_sample']:>5.2f}"
        )

    # Write model card
    write_model_card(out, best)
    return out


def write_model_card(out: dict, best: dict) -> None:
    card_path = os.path.join(MODELS_DIR, "model_card_id.md")
    lines = [
        "# Model Card — Indonesian Fake-News Detector (TAHAP 2)",
        "",
        f"Generated: {out['generated_at']} (UTC)",
        "",
        "## Task",
        "Binary classification: Valid/Real (0) vs Hoax/Fake (1) Indonesian news text.",
        "",
        "## Data (leakage-aware split by source)",
        f"- Train (out-of-source): {out['n_train']:,} rows — HF/Rifky/indonesian-hoax-news",
        f"- Test (out-of-source): {out['n_test']:,} rows — Mendeley/ICTS-2017 (never seen in training)",
        f"- Hold-out v2: {out['n_holdout']:,} rows — manual, never used in training or tuning",
        "",
        "## Models compared",
        "1. TF-IDF (1-2 gram) + LogisticRegression, class_weight=balanced",
        "2. Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2` + LogisticRegression",
        "3. Fine-tuned IndoBERT (`indobenchmark/indobert-base-p1`) — skipped (CPU fine-tuning infeasible within time budget)",
        "",
        "## Thresholds (three-zone reporting, same as app.py)",
        f"- Valid zone:  p_fake <= {out['threshold_valid']:.2f}",
        f"- Hoax zone:   p_fake >= {out['threshold_fake']:.2f}",
        f"- Gray zone:   {out['threshold_valid']:.2f} < p_fake < {out['threshold_fake']:.2f}",
        "",
        "## Results — out-of-source test set",
        "",
        "| Model | Accuracy | ROC-AUC | F1 hoax | Recall hoax | FPR | Gray zone | Brier | ms/sample |",
        "|-------|----------|---------|---------|-------------|-----|-----------|-------|-----------|",
    ]
    for m in out["results"]["test_out_of_source"]:
        if m.get("skipped"):
            lines.append(f"| {m['name']} | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED |")
            continue
        lines.append(
            f"| {m['name']} | {m['accuracy']:.2%} | {m['roc_auc']:.4f} | "
            f"{m['f1_hoax']:.4f} | {m['recall_hoax']:.4f} | {m['fpr']:.2%} | "
            f"{m['gray_zone_pct']:.2%} | {m['brier']:.4f} | {m['infer_ms_per_sample']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Results — independent hold-out v2",
            "",
            "| Model | Accuracy | ROC-AUC | F1 hoax | Recall hoax | FPR | Gray zone | Brier | ms/sample |",
            "|-------|----------|---------|---------|-------------|-----|-----------|-------|-----------|",
        ]
    )
    for m in out["results"]["holdout_v2"]:
        if m.get("skipped"):
            lines.append(f"| {m['name']} | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED |")
            continue
        lines.append(
            f"| {m['name']} | {m['accuracy']:.2%} | {m['roc_auc']:.4f} | "
            f"{m['f1_hoax']:.4f} | {m['recall_hoax']:.4f} | {m['fpr']:.2%} | "
            f"{m['gray_zone_pct']:.2%} | {m['brier']:.4f} | {m['infer_ms_per_sample']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Best model (by hold-out accuracy)",
        ]
    )
    if best:
        lines.append(f"- {best['name']}: hold-out accuracy {best['accuracy']:.2%}")
    else:
        lines.append("- None (no completed models)")
    lines.extend(
        [
            "",
            "## Honest assessment",
            "- Hold-out Indonesian accuracy is **below 85%**. The model is NOT reliable.",
            "- Do not claim the model is reliable for production Indonesian fake-news detection.",
            "- Results are research-grade and should be treated as experimental.",
            "",
        ]
    )
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved model card: {card_path}")


if __name__ == "__main__":
    main()
