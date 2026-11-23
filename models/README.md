# Model artefacts — `models/`

Direktori ini berisi artefak model deteksi berita palsu (Fake News Detector).

## Berkas yang tersimpan di repositori (git-tracked)

Berkas berikut dikomited ke repositori dan diverifikasi integritasnya saat aplikasi
startup melalui `CHECKSUMS.txt`:

| Berkas | Keterangan |
|--------|------------|
| `best_params.json` | Hyperparameter terbaik hasil tuning model Inggris. |
| `feature_weights.json` | Bobot fitur (n-gram) untuk penjelasan model (XAI), Inggris. |
| `threshold.json` | Ambang keputusan model Inggris. |
| `model_card_id.md` | Kartu model untuk model Indonesia. |
| `id/feature_weights.json` | Bobot fitur model Indonesia (XAI). |
| `id/metadata.json` | Metadata pelatihan model Indonesia. |
| `id/threshold.json` | Ambang keputusan model Indonesia. |

## Berkas yang TIDAK disimpan di repositori (git-ignored)

Berkas berikut **tidak** dikomited (dicantumkan di `.gitignore` di root) karena
berukuran besar dan dapat dibangun ulang dengan pelatihan:

- `model.pkl`, `vectorizer.pkl` (model utama Inggris)
- `id/model.pkl`, `id/vectorizer.pkl` (model Indonesia)
- `hoax_index.npy`, `hoax_index_meta.json` (indeks debunk berbasis embedding)

Jika berkas di atas tidak tersedia saat startup, aplikasi tetap berjalan dalam
mode terbatas (prediksi Bahasa Inggris tidak tersedia hingga artefak
tersedia). Lihat `train_model.py` / `train_model_id.py` untuk membangun ulang.

## Integritas (checksum)

`CHECKSUMS.txt` berisi berkas SHA-256 untuk setiap artefak model. Saat aplikasi
dimulai, verifikasi checksum dijalankan otomatis kecuali diatur
`SKIP_CHECKSUM=1`:

```bash
# Verifikasi normal (default)
python app.py

# Lewati verifikasi checksum (contoh: artefak belum tersedia)
SKIP_CHECKSUM=1 python app.py
```

Jika berkas yang ada di disk tidak cocok dengan `CHECKSUMS.txt`, aplikasi akan
mencetak peringatan dan tidak akan memuat artefak yang bermasalah (bukan
menghentikan seluruh proses), agar deployment tidak gagal total karena satu
artefak rusak.

## Regenerasi artefak

```bash
# Model utama (Inggris) + evaluasi
python train_model.py

# Model regional Indonesia
python train_model_id.py

# Indecks debunk (jika terdapat data klaim dibantah)
# -> lihat skrip pembangun hoax_index di repositori terkait
```

Setiap kali artefak dikemas ulang, perbarui `CHECKSUMS.txt`:

```bash
cd models
sha256sum $(find . -type f ! -name 'CHECKSUMS.txt' ! -name 'README.md' | sort) > CHECKSUMS.txt
```

## Ambang keputusan (dokumentasi)

- Model Inggris (utama): ambang 0.44 (divalidasi pada split hold-out, bukan data uji).
- Model Indonesia (regional): ambang 0.31.

> **Peringatan:** Model Indonesia masih eksperimental. Akurasi pengujian luaran
> Indonesia masih di bawah 85%, jadi hasil bersifat indikasi awal dan bukan
> jaminan. Verifikasi selalu ke sumber tepercaya sebelum percaya atau membagikan.
