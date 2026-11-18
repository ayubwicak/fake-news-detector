"""Batch prediction tester for the Fake News Detector.

Usage:
    python scripts/batch_test.py data/batch_examples.csv
    python scripts/batch_test.py data/batch_examples.csv --output results.csv

CSV requirements:
    - 'text' column (required)
    - 'label' column (optional: 0 = Real, 1 = Fake). If present, accuracy is computed.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train_model import clean_text  # noqa: E402


def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, 'models', 'model.pkl'))
    tfidf = joblib.load(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'))
    return model, tfidf


def main():
    parser = argparse.ArgumentParser(description='Batch-test the fake news model on a CSV file.')
    parser.add_argument('input_csv', help='Path to CSV with a "text" column (optional "label").')
    parser.add_argument('--output', default=None, help='Optional path for detailed results CSV.')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        sys.exit(f'Error: file not found: {args.input_csv}')

    df = pd.read_csv(args.input_csv)
    if 'text' not in df.columns:
        sys.exit('Error: input CSV must contain a "text" column.')

    model, tfidf = load_artifacts()
    texts = df['text'].astype(str).map(clean_text)
    vecs = tfidf.transform(texts)

    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)
    confidences = np.max(probs, axis=1) * 100

    df['prediction'] = np.where(preds == 1, 'Fake', 'Real')
    df['confidence'] = np.round(confidences, 2)
    df['probability_real'] = np.round(probs[:, 0] * 100, 2)
    df['probability_fake'] = np.round(probs[:, 1] * 100, 2)

    print('=' * 70)
    print(f'Processed {len(df)} samples')
    print('=' * 70)

    if 'label' in df.columns:
        y_true = df['label'].astype(int).values
        acc = accuracy_score(y_true, preds)
        print(f'Accuracy: {acc:.2%}\n')
        print(classification_report(y_true, preds, target_names=['Real', 'Fake'], zero_division=0))
        df['is_correct'] = (y_true == preds)
    else:
        print('No "label" column found — showing predictions only.\n')

    preview_cols = [c for c in ['text', 'prediction', 'confidence', 'is_correct'] if c in df.columns]
    print(df[preview_cols].to_string(index=True, max_colwidth=70))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f'\nDetailed results saved to: {args.output}')


if __name__ == '__main__':
    main()
