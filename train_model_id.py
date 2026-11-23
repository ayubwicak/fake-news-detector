#!/usr/bin/env python3
"""Indonesian fake-news training pipeline (leakage-free).

Methodology
-----------
1. Split train/test BEFORE any fitting or resampling.
2. Fit TF-IDF on the training split only; transform test with the same model.
3. Select the classifier with 5-fold CV on the training split only
   (scripts/compare_models_id.py).  CV showed SMOTE hurts this dataset, so the
   final model relies on class_weight="balanced" instead of synthetic samples.
4. Carve a validation split from training data to tune the decision threshold.
5. Train a calibrated LinearSVC and evaluate on the untouched test set.
6. Report bootstrap 95% confidence intervals (small dataset, wide intervals).
7. If data/real_news_holdout_id.csv exists, measure the false-positive rate on
   independent real news (never used for training or tuning).
8. Save model, vectorizer, XAI weights, threshold, metrics, and metadata.

Label convention: 0 = Valid/Real, 1 = Hoax/Fake.

Usage:
    python train_model_id.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import INDONESIAN_STOPWORDS, clean_text  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models", "id")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
MAX_FEATURES = 30000
N_BOOTSTRAP = 1000


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Load data/train_id.csv, merge human feedback, and clean text."""
    candidates = [
        os.path.join(DATA_DIR, "train_id.csv"),
        os.path.join(DATA_DIR, "sample_news_id.csv"),
    ]
    source = next((path for path in candidates if os.path.exists(path)), None)
    if source is None:
        raise FileNotFoundError(
            "data/train_id.csv not found. Run: python scripts/prepare_dataset_id.py"
        )

    df = pd.read_csv(source)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 30].reset_index(drop=True)

    feedback_path = os.path.join(DATA_DIR, "feedback_reviewed.csv")
    if os.path.exists(feedback_path):
        try:
            feedback = pd.read_csv(feedback_path)
            # Hanya baris yang telah disetujui reviewer yang boleh masuk pelatihan.
            if {"text", "label", "status"}.issubset(feedback.columns):
                feedback = feedback[feedback["status"] == "disetujui"].copy()
                feedback = feedback.dropna(subset=["text", "label"]).copy()
                feedback["label"] = feedback["label"].astype(int)
                feedback["clean_text"] = feedback["text"].apply(clean_text)
                feedback = feedback[feedback["clean_text"].str.len() > 30]
                feedback = feedback.drop_duplicates(subset=["clean_text"])
                existing = set(df["clean_text"])
                feedback = feedback[~feedback["clean_text"].isin(existing)]
                if len(feedback):
                    df = pd.concat(
                        [df, feedback[["text", "clean_text", "label"]]],
                        ignore_index=True,
                    )
                    print(f"Merged {len(feedback)} reviewed feedback samples from {feedback_path}")
                else:
                    print(f"Tidak ada feedback yang disetujui di {feedback_path}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: could not merge feedback: {exc}")

    print(f"Loaded dataset from: {source}")
    return df, source


def build_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.90,
        sublinear_tf=True,
        stop_words=INDONESIAN_STOPWORDS,
    )


def fit_vectorizer(texts: pd.Series) -> TfidfVectorizer:
    """Fit with min_df=2, fall back to min_df=1 for tiny datasets."""
    vectorizer = build_vectorizer(min_df=2)
    vectorizer.fit(texts)
    if len(vectorizer.get_feature_names_out()) < 1000:
        print("  min_df=2 produced a small vocabulary; retrying with min_df=1")
        vectorizer = build_vectorizer(min_df=1)
        vectorizer.fit(texts)
    return vectorizer


def build_model() -> CalibratedClassifierCV:
    base = LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=10000,
    )
    return CalibratedClassifierCV(estimator=base, cv=5)


def optimize_threshold(y_true, probabilities) -> tuple[float, float]:
    # CV showed SMOTE hurts; class_weight="balanced" already handles imbalance.
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [
        f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def extract_feature_weights(model: CalibratedClassifierCV, vectorizer) -> dict:
    coefs = [clf.estimator.coef_ for clf in model.calibrated_classifiers_]
    mean_coef = np.mean(coefs, axis=0).flatten()
    names = vectorizer.get_feature_names_out()
    return {
        name: float(weight)
        for name, weight in zip(names, mean_coef)
        if abs(weight) >= 0.01
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = N_BOOTSTRAP) -> list[float]:
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    if not scores:
        return [float("nan"), float("nan")]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def load_model_search() -> dict | None:
    """Read the 5-fold CV search produced by scripts/compare_models_id.py."""
    path = os.path.join(LOGS_DIR, "model_search_id.json")
    if not os.path.exists(path):
        print("Model-search log not found (optional: run scripts/compare_models_id.py)")
        return None
    with open(path, encoding="utf-8") as handle:
        search = json.load(handle)
    best = search.get("ranking", [None])[0]
    if best:
        metrics = search["configs"][best]
        print(
            f"CV-selected model: {best} "
            f"(F1={metrics['f1_fake_mean']:.4f} ± {metrics['f1_fake_std']:.4f})"
        )
    return search


def evaluate_holdout(vectorizer, model, threshold: float) -> dict | None:
    """Measure false-positive rate on independent real news, if available."""
    path = os.path.join(DATA_DIR, "real_news_holdout_id.csv")
    if not os.path.exists(path):
        print("\nNo real-news holdout found (data/real_news_holdout_id.csv).")
        return None

    holdout = pd.read_csv(path)
    if "text" not in holdout.columns:
        return None
    holdout = holdout.dropna(subset=["text"]).copy()
    holdout["clean_text"] = holdout["text"].apply(clean_text)
    holdout = holdout[holdout["clean_text"].str.len() > 30]
    if holdout.empty:
        return None

    X = vectorizer.transform(holdout["clean_text"])
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    fpr = float(pred.mean())
    print("\n--- Independent real-news holdout (all should be Real) ---")
    print(f"Samples: {len(holdout)}")
    print(f"False-positive rate @ {threshold:.2f}: {fpr:.2%}")
    print(f"Mean fake probability: {proba.mean():.4f}")
    return {
        "path": os.path.basename(path),
        "samples": int(len(holdout)),
        "false_positive_rate": fpr,
        "mean_fake_probability": float(proba.mean()),
        "threshold": threshold,
    }


def train_and_evaluate() -> dict:
    print("=" * 68)
    print("  INDONESIAN FAKE-NEWS DETECTOR — LEAKAGE-FREE PIPELINE")
    print("=" * 68)

    df, source_path = load_dataset()
    print(f"Total samples: {len(df):,}")
    print(f"Class distribution: {df['label'].value_counts().sort_index().to_dict()}")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"\nSplit -> train: {len(train_df):,} | test: {len(test_df):,} (untouched)")

    fit_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df["label"],
    )

    print("\n[Model selection] reading 5-fold CV results (training split only)...")
    comparison = load_model_search()

    print("\n[Validation phase] fitting vectorizer on fit-split only...")
    vectorizer_val = fit_vectorizer(fit_df["clean_text"])
    X_fit = vectorizer_val.transform(fit_df["clean_text"])
    X_val = vectorizer_val.transform(val_df["clean_text"])

    print("  training validation model...")
    model_val = build_model()
    model_val.fit(X_fit, fit_df["label"].values)
    val_proba = model_val.predict_proba(X_val)[:, 1]
    best_threshold, val_f1 = optimize_threshold(val_df["label"].values, val_proba)
    print(f"  best threshold: {best_threshold:.2f} (validation F1 fake: {val_f1:.4f})")

    print("\n[Final training] fitting vectorizer on full training split...")
    vectorizer = fit_vectorizer(train_df["clean_text"])
    X_train = vectorizer.transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    print("  training final calibrated LinearSVC...")
    model = build_model()
    model.fit(X_train, train_df["label"].values)

    y_test = test_df["label"].values
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba_test)
    precision_fake = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Valid/Real", "Hoax/Fake"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- TEST SET @ tuned threshold ---")
    print(f"Accuracy : {acc:.2%}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Valid/Real", "Hoax/Fake"],
            zero_division=0,
        )
    )

    ci_acc = bootstrap_ci(y_test, y_pred, accuracy_score)
    ci_f1 = bootstrap_ci(
        y_test, y_pred, lambda t, p: f1_score(t, p, pos_label=1, zero_division=0)
    )
    print(f"95% CI accuracy: [{ci_acc[0]:.2%}, {ci_acc[1]:.2%}]")
    print(f"95% CI F1 fake : [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")

    print("\n[Explainability] extracting averaged SVM coefficients...")
    feature_weights = extract_feature_weights(model, vectorizer)
    print(f"  {len(feature_weights):,} weighted features")

    holdout = evaluate_holdout(vectorizer, model, best_threshold)

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    weights_path = os.path.join(MODELS_DIR, "feature_weights.json")
    threshold_path = os.path.join(MODELS_DIR, "threshold.json")
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    eval_path = os.path.join(LOGS_DIR, "evaluation_id.json")
    comparison_path = os.path.join(LOGS_DIR, "model_comparison_id.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    with open(weights_path, "w", encoding="utf-8") as handle:
        json.dump(feature_weights, handle, ensure_ascii=False)
    with open(threshold_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": best_threshold,
                "validation_f1_fake": val_f1,
                "method": "maximise Fake F1 on 15% validation split",
            },
            handle,
            indent=2,
        )

    metadata = {
        "language": "id",
        "labels": {"0": "Valid/Real", "1": "Hoax/Fake"},
        "model": "CalibratedClassifierCV(LinearSVC)",
        "vectorizer": f"TfidfVectorizer(1-2 grams, max_features={MAX_FEATURES})",
        "dataset": "Mendeley p3hfgr5j3m / ICTS 2017 (Pratiwi et al.)",
        "dataset_path": os.path.basename(source_path),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": int(len(vectorizer.get_feature_names_out())),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    evaluation = {
        **metadata,
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_score_fake": float(f1_fake),
        "precision_real": float(report["Valid/Real"]["precision"]),
        "recall_real": float(report["Valid/Real"]["recall"]),
        "f1_score_real": float(report["Valid/Real"]["f1-score"]),
        "threshold": best_threshold,
        "accuracy_95ci": ci_acc,
        "f1_fake_95ci": ci_f1,
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(test_df)),
        "real_news_holdout": holdout,
        "methodology": {
            "split_before_resampling": True,
            "tfidf_fit_on_train_only": True,
            "smote_on_train_only": False,
            "smote_rejected_by_cv": True,
            "class_weight_balanced": True,
            "threshold_tuned_on_validation": True,
            "holdout_used_for_tuning": False,
            "bootstrap_ci": True,
        },
        "model_comparison_cv": comparison,
        "limitations": [
            "Dataset labels come from three-annotator voting on 2017-era articles.",
            "Only 593 samples; confidence intervals are wide.",
            "TF-IDF captures lexical patterns, not deep semantics.",
            "Regional languages are not supported by this model.",
        ],
    }
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, ensure_ascii=False)
    with open(comparison_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)

    print("\nSaved artifacts:")
    for path in [
        model_path,
        vectorizer_path,
        weights_path,
        threshold_path,
        metadata_path,
        eval_path,
        comparison_path,
    ]:
        print(f" - {path}")
    print("=" * 68)
    print("Pipeline completed successfully!")
    return evaluation


if __name__ == "__main__":
    train_and_evaluate()
