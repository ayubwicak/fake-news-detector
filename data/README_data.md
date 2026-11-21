# Dataset Bahasa Indonesia — Panduan

## Sumber data yang digunakan

| # | Nama sumber | URL | Lisensi | Baris dimuat | Catatan |
|---|-------------|-----|---------|--------------|---------|
| 1 | Hugging Face: Rifky/indonesian-hoax-news | https://huggingface.co/datasets/Rifky/indonesian-hoax-news | Public (no card) | 7,940 | 17.806 baris asli dari turnbackhoax.id; label diturunkan dari awalan `[SALAH]`/`[BENAR]` pada judul. Digunakan sebagai data pelatihan utama. |
| 2 | Mendeley / ICTS-2017 (Pratiwi, Asmara & Rahutomo 2017) | https://data.mendeley.com/datasets/p3hfgr5j3m/1 | CC BY 4.0 | 547 | 600 asli; 12 topik, label oleh 3 anotator dengan voting. Digunakan sebagai test set out-of-source (tidak pernah dilatih). |
| 3 | Manual hold-out (penulis) | internal | Asli (tulis ulang) | 103 | Hanya untuk evaluasi, tidak pernah dilatih atau tuning threshold. |

## Sumber yang tersedia tapi tidak dapat dimuat otomatis (dokumentasi saja)

| Nama | URL | Lisensi | Alasan tidak dimuat |
|------|-----|---------|---------------------|
| BRIN / Indonesia Political Hoax Dataset | https://data.mendeley.com/datasets/mzx5rd397v/1 | CC BY 4.0 | UUID file unduhan langsung tidak dapat ditemukan di halaman DOI. |
| Hugging Face: nlp-brin-id/id-hoax-report-merge | https://huggingface.co/datasets/nlp-brin-id/id-hoax-report-merge | MIT | Memerlukan login Hugging Face (401 Unauthorized tanpa token). |
| Hugging Face: nlp-brin-id/id-hoax-report | https://huggingface.co/datasets/nlp-brin-id/id-hoax-report | MIT | Memerlukan login Hugging Face. |
| Kaggle: Indonesian Hoax News Dataset (Kominfo) | https://www.kaggle.com/datasets/ireddragonicy/indonesian-hoax-news-dataset | Apache 2.0 | Memerlukan login Kaggle. |
| Kaggle: Deteksi Berita Hoaks Indo | https://www.kaggle.com/datasets/mochamadabdulazis/deteksi-berita-hoaks-indo-dataset | Custom (riset dan pendidikan saja) | Memerlukan login Kaggle. |
| Kaggle: Indonesian Fact and Hoax Political News | https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news | MIT | Memerlukan login Kaggle. |

## Format kolom

`data/train_id.csv` dan `data/train_id_oos.csv` / `data/test_id_oos.csv`:

| Kolom | Deskripsi |
|-------|-----------|
| `text` | Teks asli (sebelum cleaning) |
| `clean_text` | Teks setelah cleaning dan stripping pintasan |
| `label` | 0 = Valid/Real, 1 = Hoax/Fake |
| `source` | Nama sumber data |
| `year` | Tahun terbit (kosong jika tidak diketahui) |
| `topic` | Topik/klasifikasi halus (kosong jika tidak diketahui) |
| `language` | Selalu `"id"` |

`data/holdout_id_v2.csv`: hold-out manual independen, tidak pernah digunakan dalam pelatihan atau tuning threshold. Minimal 100 baris, seimbang antara kelas valid dan hoaks, beragam topik dan panjang teks (judul, paragraf, artikel penuh, pesan berantai/penipuan).

## Pembagian data (split by source)

Data dibagi **berdasarkan sumber**, bukan acak:
- Minimal satu sumber ditempatkan dalam test set ("out-of-source") yang **tidak pernah dilihat** selama pelatihan.
- Metrik utama adalah performa pada test out-of-source, bukan test acak dalam distribusi.

## Pembersihan data (tidak ada pintasan / label leak)

Sebelum TF-IDF dipasang, setiap baris dibersihkan untuk menghapus pintasan yang bisa membuat model hafal label alami konten:

1. Hapus awalan label: `[SALAH]`, `[BENAR]`, `HOAX:`, `FAKTA:`, `SALAH:`, `BENAR:`, `Hoax:`
2. Hapus nama situs/media: detik, kompas, tribun, liputan6, kumparan, cnn, republika, antara, viva, okezone, sindonews, suara.com, rmol, jawapos, pikiranrakyat, komdigi, kominfo, beritasatu, dll.
3. Hapus dateline dan tanggal (contoh: `JAKARTA -`, `12/01/2024`)
4. Hapus tautan dan URL
5. Hapus kata penanda sumber yang bocor label: hoax, hoaks, debunk, cek fakta, klarifikasi, disinformasi, misinformasi
6. NFKC normalisasi, lowercase, hanya huruf a-z, rapikan spasi
7. Deduplikasi eksak dan near-duplicate (SequenceMatcher > 0.95)

## Audit

Hasil audit tersimpan di `logs/dataset_id_audit.json` dan `logs/source_x_label.json`, mencakup:
- Distribusi label per sumber (matriks source × label)
- Prevalensi sinyal bocor (URL, kata hoax, nama situs, dll.) sebelum dan setelah cleaning
- Statistik panjang teks per kelas
- Duplikasi eksak dan teks dengan label konflik

## Peringatan lisensi dan hak cipta

- Dataset Mendeley ICTS-2017 berlisensi CC BY 4.0 — dapat digunakan untuk pelatihan model.
- Artikel media dari sumber komersial (Kompas, Detik, CNN Indonesia, dll.) tetap hak cipta milik masing-masing pemilik. Dataset ini **tidak** menyimpan teks penuh artikel media tersebut secara massal; jika digunakan, simpan sebagai tautan + fitur turunan saja.
- Hold-out manual (`holdout_id_v2.csv`) adalah tulisan ulang asli penulis, bukan salinan artikel berhak cipta.
