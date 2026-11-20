#!/usr/bin/env python3
"""Model search for the Indonesian detector — training split only.

Compares word TF-IDF vs word+character TF-IDF, with and without SMOTETomek,
using 5-fold stratified CV.  The untouched test set is NOT used here; model
selection must not peek at it.

Usage:
    python scripts/compare_models_id.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import INDONESIAN_STOPWORDS  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def word_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        stop_words=INDONESIAN_STOPWORDS,
    )


def char_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=3,
        sublinear_tf=True,
    )


def word_char_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", word_features()),
            ("char", char_features()),
        ]
    )


def make_pipeline(features, clf, smote: bool):
    steps = [("features", features)]
    if smote:
        steps.append(
            (
                "smote",
                SMOTETomek(
                    smote=SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
                    random_state=RANDOM_STATE,
                ),
            )
        )
    steps.append(("clf", clf))
    return (ImbPipeline if smote else SkPipeline)(steps)


def main() -> None:
    df = pd.read_csv(os.path.join(DATA_DIR, "train_id.csv"))
    df = df.dropna(subset=["clean_text", "label"])
    train_df, _ = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    configs = {
        "word+LinearSVC": (word_features(), LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000), False),
        "word+LogReg": (word_features(), LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE), False),
        "word+ComplementNB": (word_features(), ComplementNB(alpha=0.5), False),
        "wordchar+LinearSVC": (word_char_features(), LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000), False),
        "wordchar+LogReg": (word_char_features(), LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE), False),
        "wordchar+ComplementNB": (word_char_features(), ComplementNB(alpha=0.5), False),
        "word+LinearSVC+SMOTE": (word_features(), LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000), True),
        "word+LogReg+SMOTE": (word_features(), LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE), True),
        "word+MultinomialNB": (word_features(), MultinomialNB(alpha=0.5), False),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, (features, clf, smote) in configs.items():
        print(f"Running {name} ...", flush=True)
        pipeline = make_pipeline(features, clf, smote)
        scores = cross_validate(
            pipeline,
            train_df["clean_text"],
            train_df["label"],
            cv=cv,
            scoring={
                "f1": "f1",
                "accuracy": "accuracy",
                "roc_auc": "roc_auc",
            },
            n_jobs=1,
            error_score="raise",
        )
        results[name] = {
            "f1_fake_mean": float(np.mean(scores["test_f1"])),
            "f1_fake_std": float(np.std(scores["test_f1"])),
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std": float(np.std(scores["test_accuracy"])),
            "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        }
        print(
            f"  F1={results[name]['f1_fake_mean']:.4f} ± {results[name]['f1_fake_std']:.4f} | "
            f"Acc={results[name]['accuracy_mean']:.4f} | "
            f"AUC={results[name]['roc_auc_mean']:.4f}"
        )

    ranked = sorted(results.items(), key=lambda kv: kv[1]["f1_fake_mean"], reverse=True)
    print("\nRanking by CV F1 (fake):")
    for name, metrics in ranked:
        print(f"  {name:<24} {metrics['f1_fake_mean']:.4f}")

    output = {
        "validation": "5-fold stratified CV on training split only (test untouched)",
        "n_train": int(len(train_df)),
        "configs": results,
        "ranking": [name for name, _ in ranked],
    }
    os.makedirs(LOGS_DIR, exist_ok=True)
    path = os.path.join(LOGS_DIR, "model_search_id.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
