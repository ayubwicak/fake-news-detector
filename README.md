# 📰 Fake News Detector — End-to-End NLP Machine Learning System

A production-ready Machine Learning web application that classifies news articles/headlines as **Real** or **Fake (Misinformation)**, with built-in **Explainable AI (XAI)** that highlights the exact words influencing each prediction.

Built with **Scikit-Learn**, **TF-IDF N-gram Vectorization**, **SMOTETomek**, **Calibrated Linear SVC**, and **Flask**.

---

## 🌟 Key Features

- **Text Preprocessing Pipeline** — URL/punctuation cleaning, lowercasing, stopword removal.
- **TF-IDF N-Gram Vectorization** — Unigrams + bigrams, up to 10,000 features.
- **SMOTETomek Balancing** — Handles class imbalance and removes borderline noisy samples.
- **Calibrated SVM** — `CalibratedClassifierCV(LinearSVC)` yields reliable probability scores.
- **Explainable AI (XAI)** — Returns the top influential tokens with signed weights (why "Fake"? which words triggered it?).
- **RESTful API** — `/predict`, `/metrics`, `/health` endpoints.
- **Modern Web UI** — Dark glassmorphism interface (Tailwind CSS) with confidence meters, probability bars, and XAI pills.
- **Full Test Suite** — 15 automated tests (`pytest`) covering API edge cases & pipeline logic.
- **Docker & Deployment Ready** — `Dockerfile`, `Procfile` (Gunicorn), and `requirements.txt`.
- **EDA Notebook** — Visual analysis: class distribution, word clouds, top words, confusion matrix, ROC-AUC curve.

---

## 🏗️ Architecture

```
[ User Input (Text/Headline) ]
               │
               ▼
   [ Text Cleaning & Regex ]
               │
               ▼
    [ TF-IDF Vectorizer ] (10,000 features, 1-2 grams)
               │
               ▼
[ Calibrated Linear SVC Model ]
         │            │
         ▼            ▼
   [ Class 0/1 ]   [ Calibrated Confidence % ]
         │            │
         └─────┬──────┘
               ▼
   [ XAI: Key Influential Tokens ]
               │
               ▼
   [ JSON API & Dashboard UI ]
```

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                               # Flask server — prediction API + XAI explainer
├── train_model.py                       # ML pipeline — cleaning, TF-IDF, SMOTE, SVC, evaluation, XAI weights
├── requirements.txt                     # Runtime dependencies
├── requirements-dev.txt                 # Dev dependencies (pytest, matplotlib, jupyter, ...)
├── Dockerfile                           # Container image (trains model at build time)
├── Procfile                             # Production entry point (Gunicorn)
├── .gitignore
├── README.md
├── data/
│   ├── sample_news.csv                  # Built-in seed dataset (60 labeled articles, balanced)
│   └── train.csv                        # (Optional) Full Kaggle dataset — auto-detected
├── notebooks/
│   └── exploratory_data_analysis.ipynb  # Executed EDA with plots (word clouds, ROC, confusion matrix)
├── scripts/
│   ├── batch_test.py                    # Batch-test the model on a labeled CSV
│   ├── import_feedback.py               # Bulk-import labeled corrections into feedback.csv
│   └── tune_model.py                    # GridSearchCV hyperparameter tuning
├── data/
│   ├── sample_news.csv                  # Built-in seed dataset (60 labeled articles, balanced)
│   ├── batch_examples.csv               # Example batch-test set (10 labeled rows)
│   ├── batch_holdout.csv                # Independent holdout set for honest generalization checks
│   └── feedback.csv                     # Human feedback (auto-created, git-ignored)
├── models/
│   ├── model.pkl                        # Trained calibrated model
│   ├── vectorizer.pkl                   # Fitted TF-IDF vectorizer
│   └── feature_weights.json             # XAI feature attributions (1142 tokens)
├── logs/
│   └── evaluation.json                  # Accuracy, ROC-AUC, Precision, Recall, F1, confusion matrix
├── templates/
│   └── index.html                       # Web dashboard
└── tests/
    ├── test_api.py                      # 10 API tests (validation, edge cases, XAI response)
    └── test_pipeline.py                 # 5 pipeline tests (cleaning, artifacts, inference)
```

---

## 🚀 Getting Started

### 1. Setup
```bash
cd fake-news-detector
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: for tests & notebook
```

### 2. Train the Model
```bash
python train_model.py
```
Output example (bundled 60-sample dataset):
```
Accuracy: 91.67%
ROC-AUC:  1.0000
Classification Report:
              precision    recall  f1-score
    Real (0)       1.00      0.83      0.91
    Fake (1)       0.86      1.00      0.92
```

> **Full Kaggle Dataset (recommended for production accuracy 92–95%):**
> Download `train.csv` from the [Kaggle Fake News Competition](https://www.kaggle.com/c/fake-news/data),
> place it in `data/train.csv`, and re-run `train_model.py`. The loader prioritizes it automatically.

### 3. Run the Web App
```bash
python app.py
```
Open **http://localhost:5000** — use the preset buttons to test instantly.

### 4. Run Tests
```bash
python -m pytest tests/ -v
# 15 passed
```

### 5. Run with Docker
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

---

## 🔌 API Reference

### `POST /predict`
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Anonymous whistleblowers claim covert alien bases exist beneath polar ice."}'
```

**Response (with XAI attribution):**
```json
{
  "prediction": "Fake",
  "label_code": 1,
  "confidence": 93.34,
  "probability_real": 6.66,
  "probability_fake": 93.34,
  "word_count": 17,
  "key_influences": [
    {"token": "claim",      "weight": 0.2267, "indicator": "fake", "magnitude": 0.2267},
    {"token": "anonymous",  "weight": 0.1675, "indicator": "fake", "magnitude": 0.1675},
    {"token": "covert",     "weight": 0.0889, "indicator": "fake", "magnitude": 0.0889}
  ]
}
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/predict` | POST | Classify text + return XAI tokens |
| `/metrics` | GET | Model evaluation metrics (JSON) |
| `/health` | GET | Health check (model/vectorizer/XAI status) |

---

## 🧠 Explainable AI (XAI)

The system extracts averaged SVM feature coefficients from the calibrated model and stores them in `models/feature_weights.json`. At prediction time, each input token (unigram & bigram) is matched against these weights:

- **Positive weight → pushes toward FAKE**
- **Negative weight → pushes toward REAL**

These are surfaced in the UI as color-coded pills (🔴 fake indicators, 🟢 real indicators) with their exact weights — giving users transparency into *why* the model decided as it did.

---

## 🔁 Continuous Improvement (Feedback Loop)

A model does **not** learn automatically from its predictions. It improves only when you collect **labeled corrections** and **retrain**. This project implements that full loop:

```
Predict → Human verifies (UI buttons) → feedback.csv → retrain → better model
```

### Step 1 — Collect feedback

**Via the web UI:** after each prediction, click:
- ✅ *Benar* — confirms the prediction
- ❌ *Sebenarnya Real* / ❌ *Sebenarnya Fake* — records the true label

**Via bulk import** (labeled CSV with `text,label`):
```bash
python scripts/import_feedback.py data/batch_examples.csv
```

**Via API:**
```bash
curl -X POST http://127.0.0.1:5000/feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "The central bank kept interest rates unchanged.", "label": 0}'
```

### Step 2 — Retrain
```bash
python train_model.py
# → "Merged N human-feedback samples from: data/feedback.csv"
```

### Step 3 — Measure honestly (holdout set)
```bash
python scripts/batch_test.py data/batch_holdout.csv
```
> Always evaluate on examples the model has **never** seen. Re-testing on feedback rows only proves memorization, not generalization.

### Step 4 — Tune hyperparameters
```bash
python scripts/tune_model.py --save models/best_params.json
```

### 💡 What Actually Makes It Smarter

| Action | Impact |
|--------|--------|
| Add 10–20 feedback samples | Small, local improvements |
| **Full Kaggle dataset (~20k rows)** | **Accuracy 92–95% — biggest jump** |
| Diverse writers/topics/languages | Better generalization |
| Hyperparameter tuning | +1–3% typically |
| Upgrade to BERT / IndoBERT | Semantic understanding beyond keyword matching |

> ⚠️ **Language note:** the current TF-IDF vocabulary is English. For Indonesian news, retrain with an Indonesian fake-news dataset (e.g., hoax collections from turnbackhoax.id) or fine-tune **IndoBERT** — otherwise predictions on Indonesian text will be unreliable.

---

## 📊 EDA Notebook

Open `notebooks/exploratory_data_analysis.ipynb` (already executed with outputs) to see:

1. Dataset overview & class distribution (bar + pie)
2. Word-count distribution & boxplot by class
3. Top 15 words — Fake vs Real
4. Word clouds per class
5. Confusion matrix heatmap & ROC-AUC curve
6. Key takeaways

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML & Data | Scikit-Learn, imbalanced-learn, Pandas, NumPy, Joblib |
| Explainability | Linear SVM coefficient attribution |
| Backend | Flask, Gunicorn |
| Frontend | HTML5, Tailwind CSS, Vanilla JS (Fetch API) |
| Testing | pytest (15 tests) |
| Container | Docker |
| Analysis | Jupyter, Matplotlib, Seaborn, WordCloud |

---

## ✅ Quality Checklist

- [x] End-to-end training → evaluation → serving pipeline
- [x] Class imbalance handling (SMOTETomek)
- [x] Calibrated probability output
- [x] Explainable AI (token-level attribution)
- [x] Comprehensive evaluation (Accuracy, ROC-AUC, Precision, Recall, F1)
- [x] Robust input validation & error handling
- [x] 15 automated tests (API + pipeline)
- [x] Executed EDA notebook with visualizations
- [x] Docker + Gunicorn deployment setup
- [x] Professional documentation (this README)
