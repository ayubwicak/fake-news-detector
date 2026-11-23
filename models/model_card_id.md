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

## Debunked-claim matching threshold (TAHAP B)
- Default threshold: **0.66** (cosine similarity against hoax index titles).
- This threshold was selected on the **same hold-out v2 set** used for reporting, so metrics carry selection bias. Cross-validation on hold-out v2 (split A→B / B→A) shows a wide 95% Wilson CI; do not treat 0.66 as a production-calibrated value.
- The 0.66 threshold is intentionally left unchanged. Recalibration requires an independent hold-out v3 (not yet labelled).
- Related field `/predict` includes `debunk_available` (false when sentence-transformers or the hoak index is missing at runtime).

## 🔒 Gerbang keandalan otomatis (Butir 4)

`MODEL_RELIABLE` **dihitung** oleh `reliability.py` dari `logs/holdout_v3_result.json`.
Tidak ada penulisan manual: hold-out v2 dan `models/indobert/metrics.json` **tidak**
boleh menyetelnya lagi. Nilai `true` hanya jika **SEMUA** syarat berikut terpenuhi
pada **hold-out v3** oleh **semua kandidat yang tersedia** (tidak ada pemilihan
model berdasarkan angka):

| Gerbang | Ambang |
|---|---|
| ROC-AUC | ≥ 0,85 |
| Batas bawah interval Wilson 95% akurasi | ≥ 0,80 |
| False-positive rate kelas valid | ≤ 0,15 |
| Recall hoaks | ≥ 0,75 |
| Jumlah sampel `n` | ≥ 100 |
| Jumlah sumber berbeda | ≥ 8 |
| Pangsa sumber terbesar | ≤ 0,20 |

Bila berkas hasil belum ada, validator gagal, atau salah satu gerbang tidak terpenuhi,
hasilnya **`false`**. Alasan lengkap tersedia di `GET /health` → `model_reliability.reasons`
dan di `logs/holdout_v3_result.json`. Status saat ini: **belum dievaluasi pada v3 → false**.

## 📎 Pin Revisi Hugging Face (B615)

INDOBERT (jika diaktifkan melalui `INSTALL_ML=true` di Docker) dimuat dari folder lokal `models/indobert/`, **bukan** langsung dari Hugging Face Hub, sehingga tidak ada risiko model berubah secara diam-diam.

Bila model diunduh dari Hugging Face Hub (misalnya `indobenchmark/indobert-base-p1`), harus menggunakan **revision yang di-pin** (commit hash / tag spesifik) melalui parameter `revision=` agar build deterministik dan tidak terpengaruh pembaruan model di Hub.
