"""Hyperparameter tuning for the Fake News Detector.

Runs GridSearchCV over the TF-IDF + SMOTETomek + LinearSVC pipeline and prints
the best configuration. Optionally saves the best params to JSON.

Usage:
    python scripts/tune_model.py
    python scripts/tune_model.py --save models/best_params.json
"""
import argparse
import json
import os
import sys

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer, f1_score

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train_model import load_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Tune the fake news classification pipeline.')
    parser.add_argument('--save', default=None, help='Optional path to save best params as JSON.')
    args = parser.parse_args()

    df, source = load_dataset()
    print(f'Dataset: {source} ({len(df)} samples)')

    min_class = df['label'].value_counts().min()
    k_neighbors = max(1, min(3, min_class - 1))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('smote', SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=42), random_state=42)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': [5000, 10000],
        'clf__C': [0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, pos_label=1)

    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    search.fit(df['clean_text'], df['label'])

    print('\n' + '=' * 60)
    print(f'Best CV F1 (Fake): {search.best_score_:.4f}')
    print(f'Best params: {json.dumps(search.best_params_, indent=2)}')
    print('=' * 60)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        with open(args.save, 'w') as f:
            json.dump({
                'best_score_f1_fake': float(search.best_score_),
                'best_params': search.best_params_,
                'dataset': os.path.basename(source),
                'samples': int(len(df))
            }, f, indent=2)
        print(f'Best params saved to: {args.save}')


if __name__ == '__main__':
    main()
