"""Fake News Detector — training pipeline (leakage-free methodology).

Methodology (serious ML):
  1. Split train/test BEFORE any fitting or resampling.
  2. Fit TF-IDF on training data only; transform test with the same vectorizer.
  3. Apply SMOTETomek on training data only.
  4. Carve a validation split from training data to tune the decision threshold.
  5. Train calibrated LinearSVC, evaluate on the untouched test set.
  6. Save artifacts: model, vectorizer, XAI weights, threshold, metrics.

Label convention: 0 = Real, 1 = Fake.

Usage:
    python train_model.py
"""
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.15
MAX_FEATURES = 20000


def clean_text(text: str) -> str:
    """Preprocess text: remove URLs, punctuation, digits; lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def load_dataset():
    """Load prepared dataset (data/train.csv) or fallback sample, plus feedback."""
    candidate_paths = [
        os.path.join(DATA_DIR, 'train.csv'),
        os.path.join(BASE_DIR, 'train.csv'),
        os.path.join(DATA_DIR, 'sample_news.csv'),
    ]

    df = None
    source_used = None

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                temp_df = pd.read_csv(path)
                if 'text' in temp_df.columns and 'label' in temp_df.columns:
                    df = temp_df
                    source_used = path
                    break
                elif 'title' in temp_df.columns and 'label' in temp_df.columns:
                    temp_df['text'] = temp_df['title'].fillna('') + " " + temp_df.get('text', '').fillna('')
                    df = temp_df
                    source_used = path
                    break
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")

    if df is None:
        raise FileNotFoundError(
            "No valid dataset found. Run 'python scripts/prepare_dataset.py' or place data/train.csv."
        )

    print(f"Loaded dataset from: {source_used}")
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 10].reset_index(drop=True)

    # Merge human feedback collected via the web UI (/feedback endpoint)
    feedback_path = os.path.join(DATA_DIR, 'feedback.csv')
    if os.path.exists(feedback_path):
        try:
            fb = pd.read_csv(feedback_path)
            if {'text', 'label'}.issubset(fb.columns):
                fb = fb.dropna(subset=['text', 'label'])
                fb['label'] = fb['label'].astype(int)
                fb['clean_text'] = fb['text'].apply(clean_text)
                fb = fb[fb['clean_text'].str.len() > 10]
                fb = fb.drop_duplicates(subset=['clean_text'])
                existing = set(df['clean_text'])
                fb = fb[~fb['clean_text'].isin(existing)]
                if len(fb) > 0:
                    df = pd.concat([df, fb[['text', 'label', 'clean_text']]], ignore_index=True)
                    print(f"Merged {len(fb)} human-feedback samples from: {feedback_path}")
                else:
                    print("Human feedback found, but all samples already exist in dataset.")
        except Exception as e:
            print(f"Warning: Could not merge feedback data: {e}")

    return df, source_used


def balance_training_data(X, y):
    """Apply SMOTETomek to TRAINING data only (never to test data)."""
    class_counts = np.bincount(y)
    min_class_count = min(class_counts)

    if min_class_count >= 6:
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            k_neighbors = min(5, min_class_count - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
            sampler = SMOTETomek(smote=smote, random_state=RANDOM_STATE)
            X_res, y_res = sampler.fit_resample(X, y)
            print(f"  SMOTETomek (train only): {np.bincount(y)} -> {np.bincount(y_res)}")
            return X_res, y_res
        except Exception as e:
            print(f"  SMOTETomek skipped ({e}).")
    else:
        print("  Skipping SMOTETomek (class too small).")
    return X, y


def build_model():
    base_svc = LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000)
    return CalibratedClassifierCV(base_svc, cv=3)


def optimize_threshold(y_true, proba_fake):
    """Choose the decision threshold that maximizes Fake-class F1 on validation."""
    thresholds = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_true, (proba_fake >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def extract_feature_weights(calibrated_model, tfidf_vectorizer, min_abs=0.01):
    """Average SVM coefficients across calibration folds for XAI attribution."""
    coefs = [clf.estimator.coef_ for clf in calibrated_model.calibrated_classifiers_]
    avg_coef = np.mean(coefs, axis=0).flatten()
    feature_names = tfidf_vectorizer.get_feature_names_out()

    weights = {
        name: float(w)
        for name, w in zip(feature_names, avg_coef)
        if abs(w) >= min_abs
    }
    return weights


def train_and_evaluate():
    print("=" * 64)
    print("  FAKE NEWS DETECTOR — LEAKAGE-FREE TRAINING PIPELINE")
    print("=" * 64)

    # ---------- 1. Data ----------
    df, source_name = load_dataset()
    print(f"Total samples: {len(df):,}")
    print(f"Class distribution: {dict(pd.Series(df['label']).value_counts().sort_index())}")

    # ---------- 2. Split BEFORE fitting/resampling ----------
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['label']
    )
    print(f"\nSplit -> train: {len(train_df):,} | test: {len(test_df):,} (untouched)")

    # ---------- 3. Validation split (from train) for threshold tuning ----------
    fit_df, val_df = train_test_split(
        train_df,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df['label']
    )

    print("\n[Validation phase] fitting vectorizer on fit-split only...")
    tfidf_val = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )
    X_fit = tfidf_val.fit_transform(fit_df['clean_text'])
    X_val = tfidf_val.transform(val_df['clean_text'])

    X_fit_bal, y_fit_bal = balance_training_data(X_fit, fit_df['label'].values)

    print("  training validation model...")
    model_val = build_model()
    model_val.fit(X_fit_bal, y_fit_bal)

    val_proba = model_val.predict_proba(X_val)[:, 1]
    best_threshold, val_f1 = optimize_threshold(val_df['label'].values, val_proba)
    print(f"  best threshold: {best_threshold:.2f} (validation Fake-F1: {val_f1:.4f})")

    # ---------- 4. Final training on the full training split ----------
    print("\n[Final training] fitting vectorizer on full training split...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2
    )
    X_train = tfidf.fit_transform(train_df['clean_text'])
    X_test = tfidf.transform(test_df['clean_text'])

    X_train_bal, y_train_bal = balance_training_data(X_train, train_df['label'].values)

    print("  training final calibrated LinearSVC...")
    model = build_model()
    model.fit(X_train_bal, y_train_bal)

    # ---------- 5. Evaluation on untouched test set ----------
    proba_test = model.predict_proba(X_test)[:, 1]
    y_test = test_df['label'].values

    for name, threshold in [('0.50 (default)', 0.50), (f'{best_threshold:.2f} (tuned)', best_threshold)]:
        y_pred = (proba_test >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba_test)
        print(f"\n--- TEST @ threshold {name} ---")
        print(f"Accuracy : {acc:.4%}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake'], zero_division=0))

    # Final reported metrics use the tuned threshold
    y_pred = (proba_test >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba_test)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    prec_fake = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_fake = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # ---------- 6. XAI weights ----------
    feature_weights = extract_feature_weights(model, tfidf)
    print(f"XAI: {len(feature_weights):,} weighted features extracted")

    # ---------- 7. Save artifacts ----------
    model_path = os.path.join(MODELS_DIR, 'model.pkl')
    vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    weights_path = os.path.join(MODELS_DIR, 'feature_weights.json')
    threshold_path = os.path.join(MODELS_DIR, 'threshold.json')
    eval_path = os.path.join(LOGS_DIR, 'evaluation.json')

    joblib.dump(model, model_path)
    joblib.dump(tfidf, vec_path)
    with open(weights_path, 'w') as f:
        json.dump(feature_weights, f)
    with open(threshold_path, 'w') as f:
        json.dump({
            'threshold': best_threshold,
            'validation_f1_fake': val_f1,
            'method': 'maximize Fake F1 on 15% validation split'
        }, f, indent=2)

    eval_data = {
        'source_dataset': os.path.basename(source_name),
        'total_samples': int(len(df)),
        'train_samples': int(len(train_df)),
        'test_samples': int(len(test_df)),
        'n_features': int(len(tfidf.get_feature_names_out())),
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'precision_fake': float(prec_fake),
        'recall_fake': float(rec_fake),
        'f1_score_fake': float(f1_fake),
        'precision_real': float(rep['0']['precision']),
        'recall_real': float(rep['0']['recall']),
        'f1_score_real': float(rep['0']['f1-score']),
        'threshold': best_threshold,
        'confusion_matrix': cm.tolist(),
        'methodology': {
            'split_before_resampling': True,
            'tfidf_fit_on_train_only': True,
            'smote_on_train_only': True,
            'threshold_tuned_on_validation': True
        }
    }

    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print("\nSaved artifacts:")
    for p in [model_path, vec_path, weights_path, threshold_path, eval_path]:
        print(f" - {p}")
    print("=" * 64)
    print("Pipeline completed successfully!")

    return model, tfidf, eval_data


if __name__ == '__main__':
    train_and_evaluate()
