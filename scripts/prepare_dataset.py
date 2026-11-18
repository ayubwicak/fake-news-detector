"""Prepare the full fake-news dataset for training.

Downloads the public GonzaloA/fake_news dataset (40,587 English articles from
Hugging Face), strips agency shortcuts (e.g. "Reuters" datelines), removes
duplicates, and writes a clean `data/train.csv` with columns: text,label.

Label convention used by this project: 0 = Real, 1 = Fake.
(The source dataset uses the inverse: 1 = true, 0 = false.)

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --max-rows 20000
"""
import argparse
import os
import re
import sys
import urllib.request

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

SOURCES = {
    'raw_train.csv': 'https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/train.csv',
    'raw_test.csv': 'https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/test.csv',
    'raw_eval.csv': 'https://huggingface.co/datasets/GonzaloA/fake_news/resolve/main/evaluation.csv',
}

AGENCY_PATTERNS = [
    r'\(Reuters\)\s*-?',
    r'\bReuters\b',
    r'\(AP\)\s*-?',
    r'\(AFP\)\s*-?',
    r'\(Bloomberg\)\s*-?',
]


def download_sources(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in SOURCES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000 and not force:
            print(f'  [skip] {filename} already present ({os.path.getsize(path) / 1e6:.1f} MB)')
            continue
        print(f'  [download] {filename} ...')
        urllib.request.urlretrieve(url, path)
        print(f'  [done] {filename} ({os.path.getsize(path) / 1e6:.1f} MB)')


def strip_agency_markers(text: str) -> str:
    """Remove wire-service markers that cause label leakage."""
    for pattern in AGENCY_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    # Collapse leftover dateline artifacts, e.g. "WASHINGTON  - "
    text = re.sub(r'\b([A-Z][A-Z .]+)\s+-\s+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_key(text: str, length: int = 400) -> str:
    key = re.sub(r'[^a-z0-9 ]', '', text.lower())
    return re.sub(r'\s+', ' ', key).strip()[:length]


def main():
    parser = argparse.ArgumentParser(description='Download and prepare the full fake-news dataset.')
    parser.add_argument('--max-rows', type=int, default=None, help='Optional cap on dataset size.')
    parser.add_argument('--force-download', action='store_true', help='Re-download raw files.')
    args = parser.parse_args()

    print('=' * 60)
    print('  DATASET PREPARATION')
    print('=' * 60)
    download_sources(force=args.force_download)

    frames = []
    for filename in SOURCES:
        path = os.path.join(DATA_DIR, filename)
        df = pd.read_csv(path, sep=';')
        df = df.loc[:, [c for c in df.columns if not str(c).startswith('Unnamed')]]
        frames.append(df)
        print(f'  loaded {filename}: {len(df):,} rows')

    df = pd.concat(frames, ignore_index=True)
    print(f'\n  combined rows      : {len(df):,}')
    print(f'  source label counts: {dict(df["label"].value_counts())}')

    # Invert labels: source 1=true, 0=false  ->  project 1=fake, 0=real
    df['label'] = 1 - df['label'].astype(int)

    df['title'] = df['title'].fillna('').astype(str)
    df['text'] = df['text'].fillna('').astype(str)

    # Build full article text and strip agency shortcuts
    df['text'] = (df['title'].str.strip() + '. ' + df['text'].str.strip()).str.strip()
    df['text'] = df['text'].map(strip_agency_markers)

    # Drop rows that are too short to be meaningful
    df = df[df['text'].str.len() >= 50]

    # Exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f'  exact duplicates removed: {before - len(df):,}')

    # Near-duplicates (same normalized first 400 chars + same label)
    df['_key'] = df['text'].map(normalize_key)
    before = len(df)
    df = df.drop_duplicates(subset=['_key', 'label'])
    print(f'  near-duplicates removed : {before - len(df):,}')

    df = df.drop(columns=['_key', 'title'])

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)
        print(f'  downsampled to          : {len(df):,} rows')

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, 'train.csv')
    df[['text', 'label']].to_csv(out_path, index=False)

    reuters_left = df['text'].str.contains('reuters', case=False, regex=False).mean() * 100
    print('\n' + '=' * 60)
    print('  PREPARED DATASET SUMMARY')
    print('=' * 60)
    print(f'  rows            : {len(df):,}')
    print(f'  label counts    : {dict(df["label"].value_counts())}')
    print(f'  "reuters" left  : {reuters_left:.2f}% (was ~32% before stripping)')
    print(f'  avg text length : {df["text"].str.len().mean():.0f} chars')
    print(f'  saved to        : {out_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
