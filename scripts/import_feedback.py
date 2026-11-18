"""Import a labeled CSV into the human-feedback dataset.

Usage:
    python scripts/import_feedback.py data/batch_examples.csv

The CSV must contain 'text' and 'label' columns (0 = Real, 1 = Fake).
Rows are appended to data/feedback.csv; duplicates are skipped.
Run `python train_model.py` afterwards to incorporate the feedback.
"""
import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from train_model import clean_text  # noqa: E402

FEEDBACK_PATH = os.path.join(BASE_DIR, 'data', 'feedback.csv')


def main():
    parser = argparse.ArgumentParser(description='Import labeled rows into the feedback dataset.')
    parser.add_argument('input_csv', help='CSV with "text" and "label" columns.')
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        sys.exit(f'Error: file not found: {args.input_csv}')

    df = pd.read_csv(args.input_csv)
    if not {'text', 'label'}.issubset(df.columns):
        sys.exit('Error: CSV must contain "text" and "label" columns.')

    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]
    df = df[df['text'].astype(str).str.strip().str.len() >= 10]

    existing = set()
    if os.path.exists(FEEDBACK_PATH):
        old = pd.read_csv(FEEDBACK_PATH)
        if 'text' in old.columns:
            existing = set(old['text'].astype(str).str.strip())

    new_rows = df[~df['text'].astype(str).str.strip().isin(existing)]
    skipped = len(df) - len(new_rows)

    if len(new_rows) > 0:
        header_needed = not os.path.exists(FEEDBACK_PATH)
        new_rows[['text', 'label']].to_csv(
            FEEDBACK_PATH, mode='a', header=header_needed, index=False
        )

    total = 0
    if os.path.exists(FEEDBACK_PATH):
        total = len(pd.read_csv(FEEDBACK_PATH))

    print(f'Imported : {len(new_rows)} new feedback samples')
    print(f'Skipped  : {skipped} duplicates/invalid rows')
    print(f'Total    : {total} feedback samples in {FEEDBACK_PATH}')
    print('\nNext step: run "python train_model.py" to retrain with this feedback.')


if __name__ == '__main__':
    main()
