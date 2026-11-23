# Panduan Pelabelan Hold-out v3 (Saring Berita)

Hold-out v3 adalah **satu-satunya** set yang boleh dipakai untuk laporan akurasi akhir.
Set ini **TIDAK** dipakai untuk training, tuning ambang, pemilihan model, atau kalibrasi.
Skrip evaluasi hanya boleh dijalankan **sekali** (lihat `scripts/evaluate_on_holdout_v3.py`).

> File kerja: `data/holdout_id_v3.csv` (di-gitignore karena memuat teks).
> Yang boleh di-commit: `data/holdout_id_v3_links_template.csv` (hanya tautan + label, tanpa teks).
> **Jangan** mengisi `data/holdout_id_v3.csv` otomatis; label dan teks harus berasal
> dari pembaca manusia. Proyek ini sengaja tidak menyediakan contoh teks berlabel
> di file csv untuk menghindari label ikut "dipraktekkan" ke model.

## 1. Definisi operasional label

| Label | Nama | Definisi operasional (harus memenuhi SEMUA) |
|---|---|---|
| `1` | **Hoaks** | Klaim/pernyataan yang **sudah dibantah** oleh pemeriksa fakta bersertifikat / lembaga resmi (MAFINDO, turnbackhoax.id, cekfakta.com, Kominfo, lembaga pemerintah, atau media tepercaya dengan artikel klarifikasi). **Wajib** mengisi `bukti_url` berisi tautan bantahan. Tanpa tautan bantahan → jangan beri label 1. |
| `0` | **Valid** | Informasi yang **diberitakan media terverifikasi Dewan Pers** ATAU pernyataan resmi lembaga pemerintah/Instansi, **dan tidak ada bantahan** yang ditemukan. Isi `sumber_url`/`nama_sumber` dengan jelas. |

Kunci: label didasarkan pada **status verifikasi eksternal**, bukan pada kesan gaya bahasa.

## 2. Kasus abu-abu → keluar dari set

Jangan memaksa label untuk kasus berikut; **buang barisnya**:

- **Satire, parodi, lelucon, fiksi, iklan dengan jelas** — bukan hoaks menurut definisi di atas.
- **Opini/analisis/editorial** — tidak bisa dibantah benar/salah sebagai klaim faktual.
- **Klaim belum terverifikasi / masih diperdebatkan** — belum ada bantahan resmi; tunggu sampai ada.
- **Klaim lama yang sudah diklarifikasi tapi bantahannya tidak bisa diakses** — tidak memenuhi syarat bukti.
- **Terjemahan mesin / ringkasan otomatis** — bukan teks sumber manusia.
- Teks dengan **data pribadi** (nama lengkap + alamat/telepon/NIK) — jangan disimpan.

## 3. Syarat komposisi set (divalidasi otomatis)

1. Minimal **100 baris**, dengan keseimbangan ±10: untuk n = 100, masing-masing kelas 40–60.
2. Minimal **8 sumber berbeda untuk kelas valid DAN 8 sumber berbeda untuk kelas hoaks**.
3. **Tidak ada satu sumber pun > 20%** dari total baris.
4. Rentang `tanggal_terbit` minimal **12 bulan** (YYYY-MM-DD).
5. Topik harus mencakup kelima kategori: **politik, kesehatan, ekonomi, bencana, penipuan_daring**
   (nama bebas, tetapi kelima maknanya harus hadir).
6. Bentuk teks harus mencakup keempat: **judul, paragraf, artikel, pesan_berantai**.
7. Panjang `text` **10–600 karakter** — cukup ringkasan/kutipan pendek + tautan.
   **Jangan menyalin artikel penuh berhak cipta.**
8. Tidak boleh ada duplikat atau near-duplicate (cosine ≥ 0,90) terhadap data latih,
   hold-out v2, maupun indeks judul hoak (`scripts/validate_holdout_v3.py` akan mengecek).

## 4. Kolom CSV

Wajib: `id, text, label, sumber_url, nama_sumber, tanggal_terbit, topik, bentuk, bukti_url, catatan`.
Opsional (disarankan untuk pelabelan ganda): `label_anotator1, label_anotator2`.

| Kolom | Isi |
|---|---|
| `id` | ID unik (mis. `v3-001`) |
| `text` | Ringkasan/kutipan pendek maks. 600 karakter, ditulis ulang bila perlu |
| `label` | `0` = valid, `1` = hoaks (label final setelah rekonsiliasi) |
| `sumber_url` | Tautan artikel/asal informasi (boleh kosong hanya untuk pesan berantai) |
| `nama_sumber` | Nama media/lembaga (dipakai untuk uji bias sumber) |
| `tanggal_terbit` | `YYYY-MM-DD` |
| `topik` | politik / kesehatan / ekonomi / bencana / penipuan_daring / lain |
| `bentuk` | judul / paragraf / artikel / pesan_berantai |
| `bukti_url` | Tautan bantahan (wajib untuk hoaks) atau liputan pendukung (valid) |
| `catatan` | Alasan label, siapa pelabel, tanggal pelabelan, dll. |
| `label_anotator1`, `label_anotator2` | Label independen dua orang; `label` diisi setelah sepakat |

Simpan berkas hasil pelabelan di `data/holdout_id_v3.csv`. Jangan pernah menjalankan
`train_model_id.py`, tuning threshold, atau pemilihan model memakai berkas ini.

## 5. Alur kerja

1. Salin `data/holdout_id_v3_links_template.csv` → `data/holdout_id_v3.csv` (abaikan
   berkas yang sudah ada bila ingin mulai ulang).
2. Isi baris satu per satu mengikuti definisi dan syarat di atas.
3. Jalankan validator:
   ```
   python scripts/validate_holdout_v3.py
   ```
   Perbaiki semua pelanggaran sampai `status: pass`. Validator **tidak mengubah data**.
4. Bila memakai dua anotator, jalankan `python scripts/agreement.py`, rekonsiliasi
   baris yang tidak sepakat, lalu isi `label` final.
5. Setelah validator lolos, jalankan **sekali**:
   ```
   python scripts/evaluate_on_holdout_v3.py
   ```
6. Laporkan siapa yang memberi label dan tanggalnya di laporan akhir.
