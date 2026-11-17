# Downloads and prepares the Kaggle Fake News dataset

This project auto-detects `data/train.csv`. To train on the full dataset (~20k articles, expected accuracy 92–95%), follow these steps:

## Option A — Kaggle CLI (recommended)

1. Create a Kaggle API token: https://www.kaggle.com/settings → *API* → *Create New Token*
2. Save `kaggle.json` to `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows)
3. Run:

```bash
pip install kaggle
kaggle competitions download -c fake-news -p data/
cd data && unzip fake-news.zip && cd ..
python train_model.py
```

## Option B — Manual download

1. Open https://www.kaggle.com/c/fake-news/data
2. Download `train.csv`
3. Place it at `data/train.csv`
4. Re-run:

```bash
python train_model.py
```

The training script prioritizes `data/train.csv` over the bundled sample automatically.
