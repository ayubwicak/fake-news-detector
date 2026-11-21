# LAPORAN AKHIR — Proyek "Saring Berita" (TAHAP A–D)

**Tanggal:** 21 November 2026  
**Branch:** `fix/qa-round-2` (lokal, 5 commit, **belum push**)  
**Peringatan keandalan (seluruh laporan):** Hold-out akurasi Indonesia terbaik = **67,96% < 85%**. Model **tidak dianggap andal**. `MODEL_RELIABLE=false`. Jangan mengklaim model andal untuk deteksi hoaks produksi Indonesia.

---

## Ringkasan per TAHAP

| TAHAP | Isi | Status | File kunci |
|-------|-----|--------|------------|
| TAHAP A | Audit konfounder sumber + template hold-out v3 | ✅ Selesai | `TAHAP_A_REPORT.md`, `scripts/audit_confounder_style.py`, `logs/confounder_style.json`, `data/holdout_id_v3_template.csv`, `data/HOLDOUT_V3_LABELING_GUIDE.md` |
| TAHAP B | Pencocokan klaim hoak yang sudah dibantah (fitur baru) | ✅ Selesai | `app.py`, `templates/index.html`, `tests/test_api.py`, `scripts/tune_debunk_threshold.py`, `logs/debunk_threshold.json` |
| TAHAP C | Siapkan fine-tuning IndoBERT di GPU (Colab) | ✅ Selesai | `notebooks/finetune_indobert_colab.ipynb`, `README_colab.md` |
| TAHAP D | Git + kebersihan | ✅ Selesai | `.gitignore`, `README.md`, 5 commit di `fix/qa-round-2` |

---

## Temuan TAHAP A — Audit Konfounder Sumber

### A.1 Matriks sumber × label (data latih, out-of-source train)

| Sumber | valid (0) | hoak (1) | Total |
|--------|-----------|----------|-------|
| HF/Rifky/indonesian-hoax-news | 3.556 | 4.384 | 7.940 |
| Mendeley/ICTS-2017 | 334 | 213 | 547 |

**Tidak ada konfounding sumber sempurna** — kedua kelas muncul di kedua sumber. Namun satu sumber (Rifky/turnbackhoax.id) mendominasi (7.940 dari 8.487 baris), berisiko bias gaya sumber ringan.

### A.2 Uji konfounder dengan fitur gaya saja

| Uji | Akurasi | ROC-AUC |
|-----|---------|---------|
| In-distribution (train) | 66,32% | 0,7577 |
| Hold-out v2 (out-of-source) | 55,34% | 0,4668 |
| Penurunan | 10,98% | — |

**Bukti konfounder sumber: tidak conclusif** — penurunan 10,98% tidak cukup besar untuk menyimpulkan model belajar gaya sumber daripada isi berita.

### A.3 Interval kepercayaan 95% (Wilson), hold-out v2 (n = 103)

| Model | Akurasi | CI95 (Wilson) | ≥ 85%? |
|-------|---------|---------------|--------|
| TF-IDF + LogisticRegression | 55,34% | [45,72%, 64,58%] | TIDAK |
| Sentence-Transformer + LogisticRegression | 67,96% | [58,44%, 76,19%] | TIDAK |

CI kedu model **tumpang tindih** → **jangan memilih model berdasarkan selisih akurasi kecil** (tidak bermakna statistik). Model terbaik 67,96% masih sangat di bawah 85% → `MODEL_RELIABLE=false`.

### A.4 Hold-out v3 (baru, belum digunakan untuk apa pun)

- `data/holdout_id_v3_template.csv` — template (kolom: text, label, sumber_url, tanggal, catatan).
- `data/HOLDOUT_V3_LABELING_GUIDE.md` — panduan pelabelan manual (>= 100 contoh, seimbang ~50/50, tidak untuk tuning).
- Belum diisi data; menunggu pelabelan manual oleh pengguna.

---

## Metrik Pencocokan TAHAP B (ambang terpilih = 0,66)

Ambang dipilih pada validasi hold-out v2 dengan aturan: **false match pada berita valid ≤ 5%, lalu recall hoak tertinggi**.

| Metrik | Nilai |
|--------|-------|
| Ambang kemiripan (cosine) | 0,66 |
| False match pada valid | 2 / 57 = **3,51%** (≤ 5% ✓) |
| Recall hoak | 13 / 46 = **28,26%** |
| Presisi pencocokan | **86,67%** |
| Model embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Ukuran indeks | 12.912 judul hoak (turnbackhoax.id), 384-dimensi |

**Catatan:** Hanya judul + tautan sumber disimpan (tidak ada teks penuh artikel berhak cipta). Kemiripan teks **bukan bukti** hoak; UI menyertakan penegasan dan tautan klarifikasi.

Field baru `similar_debunked` ditambahkan ke respons `/predict` (struktur field lama tidak diubah). UI menampilkan kartu "Mirip dengan klaim yang pernah dibantah" saat ada kecocokan, dan pesan "Tidak ditemukan kecocokan..." saat tidak ada.

---

## Status TAHAP C — Fine-tuning IndoBERT di GPU (Colab)

- `notebooks/finetune_indobert_colab.ipynb`: siap dijalankan di Google Colab (GPU). Memuat data dari `data/`, fine-tuning `indobenchmark/indobert-base-p1` 3 epoch, evaluasi test out-of-source + hold-out, menyimpan model + `metrics.json`.
- `README_colab.md`: panduan unggah data → jalankan → unduh → letakkan di `models/indobert/`.
- `app.py`: loader opsional IndoBERT (memakai `models/indobert/` bila ada + dependensi terpasang, fallback ke model TF-IDF lama bila tidak). `MODEL_RELIABLE` juga menilai `models/indobert/metrics.json` (hold-out ≥ 85%).
- **Belum ada model IndoBERT terlatih yang tersedia** (`models/indobert/` kosong/belum dibuat), sehingga `MODEL_RELIABLE` tetap `false`.

---

## TAHAP D — Git dan Kebersihan

- `.gitignore`: mengecualikan file besar (`models/*.npy`, `models/hoax_index_meta.json`, `models/indobert/`, `data/raw_*.csv`, `data/holdout_id_v3.csv`) dan cache (`.cache/`). Tidak ada file `.npy`/`.parquet`/`.zip` yang di-commit.
- `README.md`: klarifikasi **97,52% hanya berlaku Bahasa Inggris dalam distribusi** (bukan Indonesia); tambah bagian keterbatasan + status keandalan model Indonesia dan fitur pencocokan hoak.
- **5 commit** di branch `fix/qa-round-2` (TAHAP 1 foundation, TAHAP A, TAHAP B, TAHAP C, TAHAP D). **Belum push** — menunggu persetujuan pengguna.

---

## Hasil Tes Akhir

```
$ python -m pytest tests/ -v
82 passed in ~15 s
```

**82/82 lulus, 0 gagal** (78 tes lama + 4 tes baru TAHAP B). Tidak ada tes lama yang dihapus.

---

## Keseluruhan kesimpulan

- Model Indonesia **tidak andal** (hold-out 67,96% < 85%); `MODEL_RELIABLE=false` dan peringatan eksperimental dipasang di UI + `/predict`.
- Tidak ada konfounder sumber sempurna, tetapi sumber dominan berisiko bias gaya ringan.
- Fitur pencocokan hoak yang sudah dibantah siap (ambang 0,66; presisi pencocokan 86,67%, FMR valid 3,51%).
- Siap untuk fine-tuning IndoBERT via Colab (TAHAP C) bila pengguna menyetujui dan menyediakan GPU/data.
- Semua tes lulus; siap untuk push setelah persetujuan pengguna.
