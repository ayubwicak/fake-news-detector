# Dataset Guide

## Primary: full dataset (recommended)

Download, clean, and deduplicate the 40k-article dataset automatically:

```bash
cd fake-news-detector
python scripts/prepare_dataset.py
```

This script:

1. Downloads three split files (~101 MB total) from `GonzaloA/fake_news` on Hugging Face
2. Combines them into 40,587 articles
3. Inverts labels to project convention (0 = Real, 1 = Fake)
4. Strips agency shortcuts (e.g. `(Reuters) -`) that cause label leakage
5. Removes exact and near-duplicates
6. Writes `data/train.csv` (`text,label`)

Then train:

```bash
python train_model.py
```

## Fallback: bundled sample

If `data/train.csv` is absent, training automatically falls back to `data/sample_news.csv` (60 hand-written examples). This is only a smoke test — accuracy will be much lower.

## Alternative: Kaggle Fake News Competition

1. Download `train.csv` from https://www.kaggle.com/c/fake-news/data
2. Place it at `data/train.csv`
3. Run `python train_model.py`

The trainer prioritizes `data/train.csv` automatically.

## Files

| File | Purpose |
|------|---------|
| `sample_news.csv` | 60-article seed dataset (committed) |
| `batch_examples.csv` | 10 labeled rows for batch testing (committed) |
| `batch_holdout.csv` | 10 independent rows for honest generalization checks (committed) |
| `train.csv` | Prepared full dataset (generated, git-ignored) |
| `raw_*.csv` | Raw downloads (generated, git-ignored) |
| `feedback.csv` | Human corrections from the UI/API (generated, git-ignored) |
