"""Compare baseline classifiers with cross-validation.

Uses a leakage-free imblearn Pipeline: TF-IDF is fitted inside each CV fold,
SMOTETomek is applied to training folds only.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --max-rows 12000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train_model import load_dataset  # noqa: E402

RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser(description='Compare baseline models with cross-validation.')
    parser.add_argument('--max-rows', type=int, default=12000,
                        help='Stratified subsample size for faster comparison (default 12000).')
    args = parser.parse_args()

    df, source = load_dataset()
    print(f'Dataset: {source} ({len(df):,} samples)')

    if len(df) > args.max_rows:
        parts = []
        for label, group in df.groupby('label'):
            n = max(1, round(args.max_rows * len(group) / len(df)))
            parts.append(group.sample(n=min(len(group), n), random_state=RANDOM_STATE))
        df = pd.concat(parts).reset_index(drop=True)
        print(f'Stratified comparison sample: {len(df):,} rows')

    k_neighbors = max(1, min(3, int(df['label'].value_counts().min()) - 1))

    models = {
        'LinearSVC (calibrated in final model)': LinearSVC(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE),
        'MultinomialNB': MultinomialNB(),
        'ComplementNB': ComplementNB(),
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_fake': make_scorer(f1_score, pos_label=1),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in models.items():
        print(f'\nEvaluating: {name}')
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                      stop_words='english', min_df=2)),
            ('smote', SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE),
                                 random_state=RANDOM_STATE)),
            ('clf', clf),
        ])

        start = time.time()
        scores = cross_validate(pipeline, df['clean_text'], df['label'],
                                cv=cv, scoring=scoring, n_jobs=1)
        elapsed = time.time() - start

        results[name] = {
            'accuracy_mean': float(np.mean(scores['test_accuracy'])),
            'accuracy_std': float(np.std(scores['test_accuracy'])),
            'f1_fake_mean': float(np.mean(scores['test_f1_fake'])),
            'f1_fake_std': float(np.std(scores['test_f1_fake'])),
            'fit_time_sec': round(elapsed, 2),
        }
        print(f"  Accuracy: {results[name]['accuracy_mean']:.4f} ± {results[name]['accuracy_std']:.4f}")
        print(f"  Fake F1 : {results[name]['f1_fake_mean']:.4f} ± {results[name]['f1_fake_std']:.4f}")
        print(f"  Time    : {elapsed:.1f}s")

    best = max(results, key=lambda k: results[k]['f1_fake_mean'])

    print('\n' + '=' * 68)
    print('  MODEL COMPARISON SUMMARY (3-fold CV, Fake F1)')
    print('=' * 68)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['f1_fake_mean'], reverse=True):
        marker = ' <-- best' if name == best else ''
        print(f"  {name:42s} F1={r['f1_fake_mean']:.4f} (±{r['f1_fake_std']:.4f}){marker}")
    print('=' * 68)
    print(f'  Winner: {best}')

    os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
    out_path = os.path.join(BASE_DIR, 'logs', 'model_comparison.json')
    with open(out_path, 'w') as f:
        json.dump({
            'dataset': os.path.basename(source),
            'sample_rows': int(len(df)),
            'cv_folds': 3,
            'results': results,
            'winner': best,
        }, f, indent=2)
    print(f'  Saved to: {out_path}')


if __name__ == '__main__':
    main()
