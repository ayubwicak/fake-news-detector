# 📰 Fake News Detector — End-to-End NLP Machine Learning System

A production-grade Machine Learning web application that classifies news articles as **Real** or **Fake**, trained on **40.449 English news articles**, with **leakage-free methodology**, **Explainable AI (XAI)**, and a **human feedback loop**.

**Test set performance (8.091 unseen articles):**

| Metric | Score |
|--------|-------|
| Accuracy | **97.52%** |
| ROC-AUC | **0.9948** |
| Fake Precision | **0.97** |
| Fake Recall | **0.98** |
| Fake F1-Score | **0.97** |

---

## 🌟 Key Features

- **Real-scale dataset** — 40,449 deduplicated articles (GonzaloA/fake_news, Hugging Face).
- **Leakage-free pipeline** — split first, TF-IDF fit on train only, SMOTETomek on train only.
- **Shortcut removal** — strips "Reuters" agency markers (96.5% → ~0% leakage signal).
- **Model selection by evidence** — 3-fold CV comparison of LinearSVC, Logistic Regression, Naive Bayes.
- **Threshold tuning** — decision threshold 0.44 optimized on a validation split, not on test.
- **Explainable AI (XAI)** — 18,982 weighted n-gram features surfaced per prediction.
- **REST API** — `/predict`, `/metrics`, `/feedback`, `/health`.
- **Modern Web UI** — dark glassmorphism dashboard with probability bars, XAI pills, and feedback buttons.
- **Human feedback loop** — corrections saved to `feedback.csv` and merged on retrain.
- **19 automated tests** — API edge cases, pipeline logic, feedback validation.
- **Deployment ready** — Dockerfile, Procfile (Gunicorn), requirements split (runtime/dev).
- **EDA notebook** — executed with leakage audit, word clouds, confusion matrix, ROC curve.

---

## 🏗️ Architecture

```
[ User Input (Text/Headline) ]
               │
               ▼
   [ Text Cleaning & Regex ]
               │
               ▼
 [ TF-IDF Vectorizer ] (20,000 features, 1–2 grams)
               │
               ▼
[ Calibrated LinearSVC ]  ← tuned decision threshold 0.44
         │            │
         ▼            ▼
   [ Class 0/1 ]   [ Calibrated Probabilities ]
         │            │
         └─────┬──────┘
               ▼
   [ XAI: Top Influential Tokens ]
               │
               ▼
   [ JSON API & Dashboard UI ]
               │
               ▼
   [ Human Feedback → feedback.csv → Retrain ]
```

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                               # Flask server — prediction API, XAI, feedback
├── train_model.py                       # Leakage-free training pipeline + threshold tuning
├── requirements.txt                     # Runtime dependencies
├── requirements-dev.txt                 # Dev dependencies (pytest, matplotlib, jupyter, ...)
├── Dockerfile                           # Container image
├── Procfile                             # Gunicorn production entry
├── .gitignore
├── README.md
├── data/
│   ├── sample_news.csv                  # Small seed dataset (60 articles, for smoke tests)
│   ├── batch_examples.csv               # Example batch-test set (10 labeled rows)
│   ├── batch_holdout.csv                # Independent holdout set (generalization checks)
│   ├── train.csv                        # Prepared full dataset (generated, git-ignored)
│   ├── raw_*.csv                        # Raw downloaded files (git-ignored)
│   └── feedback.csv                     # Human feedback (auto-created, git-ignored)
├── notebooks/
│   └── exploratory_data_analysis.ipynb  # Executed EDA with plots
├── scripts/
│   ├── prepare_dataset.py               # Download + clean + dedupe the full dataset
│   ├── compare_models.py                # CV comparison of baseline classifiers
│   ├── tune_model.py                    # GridSearchCV hyperparameter tuning
│   ├── batch_test.py                    # Batch-test the model on a labeled CSV
│   └── import_feedback.py               # Bulk-import labeled corrections
├── models/
│   ├── model.pkl                        # Trained calibrated model
│   ├── vectorizer.pkl                   # Fitted TF-IDF vectorizer
│   ├── feature_weights.json             # XAI feature attributions
│   └── threshold.json                   # Tuned decision threshold
├── logs/
│   ├── evaluation.json                  # Final metrics + methodology flags
│   └── model_comparison.json            # CV results for all baseline models
├── templates/
│   └── index.html                       # Web dashboard
└── tests/
    ├── test_api.py                      # 14 API tests
    └── test_pipeline.py                 # 5 pipeline tests
```

---

## 🚀 Quick Start

### 1. Setup
```bash
cd fake-news-detector
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: tests & notebook
```

### 2. Prepare the full dataset (one-time, ~101 MB download)
```bash
python scripts/prepare_dataset.py
```
Output: `data/train.csv` — 40,449 cleaned articles, agency shortcuts stripped, duplicates removed.

> Offline? Skip this step — the trainer falls back to the bundled 60-article sample for a smoke test.

### 3. Train
```bash
python train_model.py
```
Prints validation threshold, final test metrics, and saves all artifacts.

### 4. Run the web app
```bash
python app.py
```
Open **http://localhost:5000**.

### 5. Run tests
```bash
python -m pytest tests/ -v     # 19 passed
```

### 6. Docker
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

---

## 🔬 Methodology (Why the Numbers Are Trustworthy)

The previous pipeline had two classic mistakes. Both are fixed:

| Issue | Before | Now |
|-------|--------|-----|
| **Vectorizer leakage** | TF-IDF fitted on all data before splitting | Fit on training split only; test transformed with same vectorizer |
| **Resampling leakage** | SMOTETomek applied before split (synthetic rows in test) | Split first; SMOTETomek applied to training data only |
| **Shortcut learning** | "Reuters" present in 96.5% of Real articles | Agency markers stripped during dataset preparation |
| **Threshold bias** | Fixed 0.5 | Tuned on a 15% validation split (0.44), evaluated on untouched test |

### Model Comparison (3-fold CV, Fake F1)

| Model | Fake F1 | Accuracy |
|-------|---------|----------|
| **Calibrated LinearSVC (selected)** | **0.9614 ± 0.0027** | **~96.3%** |
| Logistic Regression | 0.9550 ± 0.0050 | 95.9% |
| MultinomialNB | 0.9202 ± 0.0047 | 92.5% |
| ComplementNB | 0.9202 ± 0.0047 | 92.5% |

Reproduce with:
```bash
python scripts/compare_models.py
python scripts/tune_model.py --save models/best_params.json
```

---

## 🔌 API Reference

### `POST /predict`
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Anonymous whistleblowers claim secret alien bases exist beneath polar ice."}'
```

```json
{
  "prediction": "Fake",
  "label_code": 1,
  "confidence": 83.73,
  "probability_real": 16.27,
  "probability_fake": 83.73,
  "decision_threshold": 0.44,
  "word_count": 12,
  "key_influences": [
    {"token": "claim",   "weight": 0.5501, "indicator": "fake"},
    {"token": "secret",  "weight": 0.4350, "indicator": "fake"}
  ],
  "key_influences_note": null,
  "language_detection": {
    "code": "en", "name": "English", "supported": true, "warning": null
  }
}
```

When no known keyword can be attributed, `key_influences` is empty and
`key_influences_note` explains why. Full-width input (e.g. `Ｂｒｅａｋｉｎｇ`) is
normalised with Unicode NFKC before cleaning and language detection.

Language detection uses a lightweight lexicon vote. Supported languages are
English (`en`) and Indonesian (`id`); Javanese (`jv`), Sundanese (`su`), and
Batak (`btk`) are detected but flagged as unsupported (`warning` set) because
the primary model's vocabulary is English. Unknown input returns `code: "unknown"`.

### `POST /feedback`
```bash
curl -X POST http://127.0.0.1:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "The central bank kept interest rates unchanged.", "label": 0}'
```

### Other endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/metrics` | GET | Evaluation metrics + methodology flags |
| `/health` | GET | Health, threshold, pending feedback count |

**Rate limiting** — `/predict`, `/feedback`, and `/metrics` allow 30 requests
per minute per IP by default (`429` with a JSON error when exceeded). Override
with `RATE_LIMIT_PER_MINUTE` (e.g. `10 per minute`) and set
`RATE_LIMIT_STORAGE_URI` to a shared backend (e.g. Redis) for multi-worker
deployments. `/` and `/health` are deliberately unlimited for UI and uptime
checks.

`POST /feedback` only accepts labels `0`/`1` (numbers or strings); booleans and
other values are rejected with `400`.

---

## 🔁 Continuous Improvement (Feedback Loop)

```
Predict → Human verifies → feedback.csv → retrain → evaluate on holdout
```

1. **Collect** — UI buttons (✅ Benar / ❌ Sebenarnya Real / ❌ Sebenarnya Fake) or bulk:
   ```bash
   python scripts/import_feedback.py data/koreksi.csv
   ```
2. **Retrain** — `python train_model.py` (auto-merges feedback, deduplicated).
3. **Measure honestly** — `python scripts/batch_test.py data/batch_holdout.csv`
4. **Tune** — `python scripts/tune_model.py`

> Always evaluate on samples the model has never seen. Re-testing on feedback rows only proves memorization.

### What actually improves the model

| Action | Impact |
|--------|--------|
| Diverse labeled feedback (50–100+) | Better generalization |
| Larger/diverse datasets | Largest gains |
| Hyperparameter tuning | +1–3% |
| BERT / IndoBERT upgrade | Semantic understanding |

> ⚠️ **Language note:** this model's vocabulary is English. For Indonesian news, retrain with an Indonesian hoax dataset (e.g., turnbackhoax.id) or fine-tune **IndoBERT**.

---

## 🧠 Explainable AI (XAI)

Averaged SVM coefficients across calibration folds are stored as token-level weights:

- **Positive weight → pushes toward FAKE**
- **Negative weight → pushes toward REAL**

The API returns the top influential unigrams/bigrams per prediction, rendered in the UI as color-coded pills with exact weights. This makes each decision auditable rather than a black box.

---

## 📊 EDA Notebook

`notebooks/exploratory_data_analysis.ipynb` (executed) contains:

1. Dataset overview & class balance
2. **Leakage audit** — Reuters shortcut before vs after (96.5% → ~0%)
3. Text-length distributions
4. Top 20 words per class
5. Word clouds per class
6. Confusion matrix + ROC curve on the untouched test set
7. Key takeaways & limitations

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML & Data | Scikit-Learn, imbalanced-learn, Pandas, NumPy, Joblib |
| Explainability | Linear SVM coefficient attribution |
| Backend | Flask, Gunicorn |
| Frontend | HTML5, Tailwind CSS, Vanilla JS |
| Testing | pytest (19 tests) |
| Container | Docker |
| Analysis | Jupyter, Matplotlib, Seaborn, WordCloud |

---

## ✅ Quality Checklist

- [x] 40k+ real dataset with deduplication & shortcut removal
- [x] Leakage-free split/resampling methodology
- [x] Evidence-based model selection (CV comparison)
- [x] Tuned decision threshold (validation split only)
- [x] Calibrated probabilities
- [x] Explainable AI (token-level attribution)
- [x] Human feedback loop with retraining
- [x] Robust input validation & error handling
- [x] 19 automated tests
- [x] Executed EDA notebook with leakage audit
- [x] Docker + Gunicorn deployment setup
- [x] Professional documentation

---

## ⚠️ Limitations (Honest Notes)

- English-only vocabulary; Indonesian text requires a local dataset or IndoBERT.
- TF-IDF + linear models capture lexical patterns, not deep semantics.
- Dataset labels are inherited from the source; real-world drift requires periodic retraining.
- The model should assist — not replace — human fact-checking.
