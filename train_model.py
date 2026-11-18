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
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """Preprocess text: remove URLs, punctuation, and lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def load_dataset():
    """Load dataset from Kaggle train.csv or bundled sample."""
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
            "No valid dataset found. Please place 'train.csv' inside 'data/' or 'sample_news.csv'."
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
                existing = set(df['clean_text'])
                fb = fb[~fb['clean_text'].isin(existing)]
                if len(fb) > 0:
                    df = pd.concat(
                        [df, fb[['text', 'label', 'clean_text']]],
                        ignore_index=True
                    )
                    print(f"Merged {len(fb)} human-feedback samples from: {feedback_path}")
                else:
                    print("Human feedback found, but all samples already exist in dataset.")
        except Exception as e:
            print(f"Warning: Could not merge feedback data: {e}")

    return df, source_used


def handle_imbalance(X, y):
    """Safely apply SMOTETomek if samples per class allow it."""
    class_counts = np.bincount(y)
    min_class_count = min(class_counts)

    if min_class_count >= 6:
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
            k_neighbors = min(3, min_class_count - 1)
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            smote_tomek = SMOTETomek(smote=smote, random_state=42)
            X_res, y_res = smote_tomek.fit_resample(X, y)
            print(f"Applied SMOTETomek. Balanced distribution: {np.bincount(y_res)}")
            return X_res, y_res
        except Exception as e:
            print(f"SMOTETomek skipped ({e}). Proceeding with original distribution.")
    else:
        print("Dataset size per class is small; skipping SMOTETomek resampling.")

    return X, y


def extract_feature_weights(calibrated_model, tfidf_vectorizer):
    """Extract averaged feature weights for XAI (Explainable AI)."""
    coefs = [clf.estimator.coef_ for clf in calibrated_model.calibrated_classifiers_]
    avg_coef = np.mean(coefs, axis=0).flatten()
    feature_names = tfidf_vectorizer.get_feature_names_out()

    weight_dict = {
        name: float(weight)
        for name, weight in zip(feature_names, avg_coef)
        if abs(weight) > 0.001
    }
    return weight_dict


def train_and_evaluate():
    print("=" * 60)
    print("  FAKE NEWS DETECTOR - MACHINE LEARNING PIPELINE")
    print("=" * 60)

    # 1. Load data
    df, source_name = load_dataset()
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {dict(pd.Series(df['label']).value_counts())}")

    # 2. Vectorization
    print("\nFitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1
    )
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label'].values

    # 3. Handle Imbalance
    X_balanced, y_balanced = handle_imbalance(X, y)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced
    )

    # 5. Train Model
    print("\nTraining Linear Support Vector Classifier with Calibration...")
    base_svc = LinearSVC(class_weight='balanced', random_state=42)
    model = CalibratedClassifierCV(base_svc)
    model.fit(X_train, y_train)

    # 6. Evaluation
    print("\nEvaluating on Test Split...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.2%}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 7. Extract Feature Weights (XAI)
    feature_weights = extract_feature_weights(model, tfidf)

    # 8. Save Artifacts
    model_path = os.path.join(MODELS_DIR, 'model.pkl')
    vec_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    weights_path = os.path.join(MODELS_DIR, 'feature_weights.json')
    eval_path = os.path.join(LOGS_DIR, 'evaluation.json')

    joblib.dump(model, model_path)
    joblib.dump(tfidf, vec_path)

    with open(weights_path, 'w') as f:
        json.dump(feature_weights, f, indent=2)

    fake_metrics = rep.get('1', rep.get(1, {'precision': 0, 'recall': 0, 'f1-score': 0}))
    real_metrics = rep.get('0', rep.get(0, {'precision': 0, 'recall': 0, 'f1-score': 0}))

    eval_data = {
        'source_dataset': os.path.basename(source_name),
        'total_samples': int(len(df)),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'precision_fake': float(fake_metrics['precision']),
        'recall_fake': float(fake_metrics['recall']),
        'f1_score_fake': float(fake_metrics['f1-score']),
        'precision_real': float(real_metrics['precision']),
        'recall_real': float(real_metrics['recall']),
        'f1_score_real': float(real_metrics['f1-score']),
        'confusion_matrix': cm.tolist()
    }

    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2)

    print("\nSaved artifacts:")
    print(f" - Model: {model_path}")
    print(f" - Vectorizer: {vec_path}")
    print(f" - Feature Weights (XAI): {weights_path}")
    print(f" - Evaluation Log: {eval_path}")
    print("=" * 60)
    print("Pipeline completed successfully!")

    return model, tfidf, eval_data


if __name__ == '__main__':
    train_and_evaluate()
