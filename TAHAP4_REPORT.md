# Laporan Hasil — Saring Berita (Indonesian Fake-News Detector)

**Tanggal:** 21 November 2026  
**Status akhir:** Model **TIDAK andal** — hold-out accuracy Indonesia di bawah 85%.  
Jangan mengklaim model ini andal untuk deteksi hoaks produksi.

---

## Ringkasan Eksekutif

| TAHAP | Isi | Hasil |
|-------|-----|-------|
| TAHAP 0 | Perbaiki tes yang gagal | ✅ 78/78 tes lolos |
| TAHAP 1 | Bangun dataset beragam (multi-sumber) | ✅ 8.487 baris, 2 sumber, hold-out 103 baris |
| TAHAP 2 | Bandingkan model semantik | ✅ Terbaik 67,96% hold-out (< 85%) — **TIDAK andal** |
| TAHAP 3 | Tampilan jujur di UI | ✅ `MODEL_RELIABLE=False`, field `model_reliable`, banner eksperimental |
| TAHAP 4 | Laporan hasil | ✅ Dokumen ini |

**Kesimpulan utama:** Tidak ada model yang mencapai akurasi hold-out ≥ 85%. Model **tidak dianggap andal** dan dipasang dengan peringatan eksperimental di seluruh antarmuka.

---

## TAHAP 0 — Perbaikan Tes yang Gagal

### Masalah
`test_detect_red_flags_chain` gagal karena pola `tanpa_sumber` didefinisikan di `_RED_FLAGS` namun tidak pernah dievaluasi dalam fungsi `detect_red_flags()`.

### Perbaikan
`app.py`: tambahkan pemeriksaan pola `tanpa_sumber` (katanya, konon, beredar kabar, beredar informasi, menurut sumber) ke `detect_red_flags()`.

### Hasil
`python -m pytest tests/ -v` → **78 lolos, 0 gagal**.

Struktur respons `/predict` tidak diubah (hanya field tambahan diperbolehkan).

---

## TAHAP 1 — Data (Multi-sumber Indonesia)

### Skrip
`scripts/prepare_dataset_id.py` — menggabungkan sumber berlabel Indonesia, membersihkan pintasan yang bocor label, deduplikasi, pembagian berdasarkan sumber.

### Sumber yang dimuat

| Sumber | Lisensi | Baris dimuat | Valid | Hoaks | Catatan |
|--------|---------|--------------|-------|-------|---------|
| Hugging Face Rifky/indonesian-hoax-news | Public (tanpa kartu) | 7.940 | 3.556 | 4.384 | Dari turnbackhoax.id; label dari awalan `[SALAH]`/`[BENAR]`. |
| Mendeley / ICTS-2017 (Pratiwi et al. 2017) | CC BY 4.0 | 547 | 334 | 213 | Digunakan sebagai **test out-of-source** (tidak pernah dilatih). |
| Manual hold-out (tulis ulang penulis) | Asli penulis | 103 | 57 | 46 | Hanya evaluasi, tidak pernah dilatih/tuning. |

**Total baris bersih: 8.487** (≥ 3.000 terpenuhi).  
**Sumber per kelas: 2** (target ≥ 3 jika tersedia; sumber ketiga BRIN/Kaggle memerlukan login atau UUID file yang tidak dapat ditemukan otomatis — terdokumentasi di `data/README_data.md`).

### Pembagian (split by source — out-of-source)
- Train: Rifky (7.940 baris)
- Test: Mendeley/ICTS-2017 (547 baris, **tidak pernah dilihat** saat pelatihan)
- Hold-out v2: manual (103 baris, seimbang 57 valid / 46 hoaks)

### Pembersihan (tidak ada pintasan / label leak)
1. Hapus awalan label `[SALAH]`, `[BENAR]`, `HOAX:`, `FAKTA:`, dll.
2. Hapus nama situs/media (detik, kompas, tribun, cnn, republika, antara, dll.)
3. Hapus dateline dan tanggal
4. Hapus URL
5. Hapus kata penanda bocor label (hoax, hoaks, debunk, cek fakta, klarifikasi, disinformasi, misinformasi)
6. NFKC normalisasi, lowercase, huruf a-z only, rapikan spasi
7. Deduplikasi eksak dan near-duplicate

### Audit
- `logs/dataset_id_audit.json` — distribusi label, prevalensi sinyal bocor, statistik panjang, duplikasi, konflik label
- `logs/source_x_label.json` — matriks sumber × label
- Tidak ada teks dengan label konflik; 0 duplikasi eksak setelah cleaning

### File yang dihasilkan
- `data/train_id.csv` (8.487 baris)
- `data/train_id_oos.csv` / `data/test_id_oos.csv` (pembagian out-of-source)
- `data/holdout_id_v2.csv` (103 baris)
- `data/sample_news_id.csv` (60 baris smoke test)
- `data/README_data.md` — panduan sumber, lisensi, format, pembersihan

---

## TAHAP 2 — Model Semantik (Perbandingan)

### Model yang dibandingkan

| # | Model | Pendekatan |
|---|-------|------------|
| A | TF-IDF + LogisticRegression | Baseline leksikal (1-2 gram, class_weight=balanced) |
| B | Sentence-Transformer multilingual + LR | `paraphrase-multilingual-MiniLM-L12-v2` + LogisticRegression |
| C | Fine-tuned IndoBERT | `indobenchmark/indobert-base-p1` — **dilewati** (fine-tuning CPU >150 menit untuk <1 epoch, tidak memungkinkan dalam batas waktu) |

### Metrik yang dilaporkan
Akurasi, ROC-AUC, F1/recall per kelas, FPR (P(pred=hoax\|actual=valid)), gray-zone % (p_fake ∈ [0,30, 0,70]), Brier score, waktu inferensi CPU per sampel.

### Hasil — Test out-of-source (Mendeley, 547 baris)

| Model | Akurasi | ROC-AUC | F1 hoax | Recall hoax | FPR | Gray zone | Brier | ms/sampel |
|-------|---------|---------|---------|-------------|-----|-----------|-------|-----------|
| TF-IDF + LR | 62,89% | 0,641 | 0,240 | 0,150 | 6,59% | 10,42% | 0,319 | 0,01 |
| Sent-Trans + LR | 60,15% | 0,563 | 0,331 | 0,254 | 17,66% | 57,59% | 0,251 | 50,68 |
| IndoBERT | — | — | — | — | — | — | — | — (dilewati) |

### Hasil — Hold-out v2 (manual, 103 baris, **tidak pernah dilatih/tuning**)

| Model | Akurasi | ROC-AUC | F1 hoax | Recall hoax | FPR | Gray zone | Brier | ms/sampel |
|-------|---------|---------|---------|-------------|-----|-----------|-------|-----------|
| TF-IDF + LR | 55,34% | 0,719 | 0,080 | 0,043 | 3,51% | 12,62% | 0,344 | 0,04 |
| Sent-Trans + LR | **67,96%** | **0,762** | **0,621** | **0,587** | **24,56%** | **52,43%** | **0,198** | 25,35 |
| IndoBERT | — | — | — | — | — | — | — | — (dilewati) |

### Analisis
- **Terbaik hold-out: 67,96%** (Sentence-Transformer + LR) — masih **sangat di bawah 85%**.
- Model cenderung **biaya hoaks tinggi**: FPR 3,5–24,6% (banyak valid di-flag hoaks).
- Recall hoaks rendah (4,3–58,7%) — model melewatkan mayoritas hoaks.
- Gray zone tinggi untuk Sentence-Transformer (52,4%) — model tidak yakin pada lebih dari setengah prediksi.
- Brier score buruk (0,20–0,34) — kalibrasi probabilitas lemah.
- Semua model **tidak andal** untuk produksi Indonesia.

### File yang dihasilkan
- `logs/model_comparison_id_v2.json` — hasil perbandingan lengkap
- `models/model_card_id.md` — model card dengan penampilan jujur

---

## TAHAP 3 — Kejujuran Tampilan (UI)

### Perubahan `app.py`
1. **`MODEL_RELIABLE` config** — diatur `False` karena best hold-out accuracy (67,96%) < 85%. Dihasilkan secara dinamis dari `logs/model_comparison_id_v2.json`.
2. **Field `model_reliable`** ditambahkan ke respons `/predict` (nilai: `false`). Struktur respons lama tidak diubah — hanya field tambahan (diizinkan).
3. Field dikirim ke template via `render_template(..., model_reliable=MODEL_RELIABLE)`.

### Perubahan `templates/index.html`
1. Konstanta JS `const MODEL_RELIABLE = false;` di-inject dari Flask.
2. Saat `MODEL_RELIABLE == false`, ditampilkan **banner "Hasil eksperimental"** di atas setiap hasil prediksi:
   > ⚠ **Hasil eksperimental — model ini belum teruji cukup.** Akurasi pengujian luaran Indonesia masih di bawah 85%, jadi hasil ini bersifat indikasi awal dan bukan jaminan. Verifikasi selalu ke sumber tepercaya sebelum percaya atau membagikan.
3. Catatan batas sistem dipertegas (tidak ada klaim keandalan model).

### Verifikasi
- `/predict` mengembalikan `"model_reliable": false` ✓
- Struktur respons `/predict` tidak berusut (hanya menambah `model_reliable`) ✓
- Halaman `/` merender `const MODEL_RELIABLE = false;` dan CSS `.experimental-notice` ✓

---

## TAHAP 4 — Laporan & Verifikasi Akhir

### Keseluruhan hasil
- Model **tidak andal**: hold-out terbaik 67,96% < 85%.
- Tidak ada klaim keandalan model untuk produksi deteksi hoaks Indonesia.
- Semua pelaporan memperlihatkan hasil secara jujur dengan peringatan eksperimental.

### Verifikasi tes
`python -m pytest tests/ -v` → **78 lolos, 0 gagal** ✓

### File kunci
| File | Isi |
|------|-----|
| `app.py` | Endpoint `/predict` + `MODEL_RELIABLE` + field `model_reliable` |
| `templates/index.html` | UI dengan banner eksperimental saat model tidak andal |
| `scripts/prepare_dataset_id.py` | Persiapan data multi-sumber |
| `scripts/compare_models_id_v2.py` | Perbandingan model semantik TAHAP 2 |
| `data/train_id_oos.csv` / `data/test_id_oos.csv` | Pembagian out-of-source (train Rifky / test Mendeley) |
| `data/holdout_id_v2.csv` | Hold-out manual 103 baris (tidak pernah dilatih) |
| `logs/model_comparison_id_v2.json` | Hasil perbandingan model (TAHAP 2) |
| `logs/dataset_id_audit.json` | Audit data (TAHAP 1) |
| `models/model_card_id.md` | Model card dengan penampilan jujur |
| `data/README_data.md` | Panduan sumber, lisensi, format, pembersihan |

### Rekomendasi
1. Jangan deklarasikan model ini andal untuk deteksi hoaks produksi Indonesia.
2. Tampilkan selalu peringatan eksperimental (sudah terpasang di TAHAP 3).
3. Untuk improve: kumpulkan lebih banyak sumber berlabel Indonesia yang terverifikasi (≥3 sumber per kelas), kurangi bias hoaks (threshold/weight), dan coba model lebih besar dengan GPU.

---

**Peringatan keandalan:** Hold-out akurasi Indonesia = 67,96% < 85%. Model **TIDAK andal**. Jangan gunakan untuk keputusan kebenaran berita tanpa verifikasi manusia ke sumber tepercaya.
