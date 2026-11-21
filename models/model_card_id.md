# Model Card — Indonesian Fake-News Detector (TAHAP 2)

Generated: 2026-11-21T04:53:54.711663+00:00 (UTC)

## Task
Binary classification: Valid/Real (0) vs Hoax/Fake (1) Indonesian news text.

## Data (leakage-aware split by source)
- Train (out-of-source): 7,940 rows — HF/Rifky/indonesian-hoax-news
- Test (out-of-source): 547 rows — Mendeley/ICTS-2017 (never seen in training)
- Hold-out v2: 103 rows — manual, never used in training or tuning

## Models compared
1. TF-IDF (1-2 gram) + LogisticRegression, class_weight=balanced
2. Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2` + LogisticRegression
3. Fine-tuned IndoBERT (`indobenchmark/indobert-base-p1`) — skipped (CPU fine-tuning infeasible within time budget)

## Thresholds (three-zone reporting, same as app.py)
- Valid zone:  p_fake <= 0.30
- Hoax zone:   p_fake >= 0.70
- Gray zone:   0.30 < p_fake < 0.70

## Results — out-of-source test set

| Model | Accuracy | ROC-AUC | F1 hoax | Recall hoax | FPR | Gray zone | Brier | ms/sample |
|-------|----------|---------|---------|-------------|-----|-----------|-------|-----------|
| TF-IDF + LogisticRegression | 62.89% | 0.6410 | 0.2397 | 0.1502 | 6.59% | 10.42% | 0.3187 | 0.01 |
| Sentence-Transformer + LogisticRegression | 60.15% | 0.5632 | 0.3313 | 0.2535 | 17.66% | 57.59% | 0.2509 | 50.68 |
| IndoBERT (fine-tuned) | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED |

## Results — independent hold-out v2

| Model | Accuracy | ROC-AUC | F1 hoax | Recall hoax | FPR | Gray zone | Brier | ms/sample |
|-------|----------|---------|---------|-------------|-----|-----------|-------|-----------|
| TF-IDF + LogisticRegression | 55.34% | 0.7189 | 0.0800 | 0.0435 | 3.51% | 12.62% | 0.3443 | 0.04 |
| Sentence-Transformer + LogisticRegression | 67.96% | 0.7616 | 0.6207 | 0.5870 | 24.56% | 52.43% | 0.1977 | 25.35 |
| IndoBERT (fine-tuned) | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED |

## Best model (by hold-out accuracy)
- Sentence-Transformer + LogisticRegression: hold-out accuracy 67.96%

## Honest assessment
- Hold-out Indonesian accuracy is **below 85%**. The model is NOT reliable.
- Do not claim the model is reliable for production Indonesian fake-news detection.
- Results are research-grade and should be treated as experimental.
