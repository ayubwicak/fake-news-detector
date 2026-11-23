# Audit Penutupan — Butir 1 (branch `fix/qa-round-2`)

Dibuat: 23 November 2026. Semua angka disertai ukuran sampel; interval kepercayaan
Wilson 95% dipakai untuk proporsi. Pemeriksaan deterministik (mis. keberadaan berkas)
tidak memiliki makna statistik dan ditandai demikian.

---

## 1a. Feedback: 122 vs 60 baris

**Kesimpulan: tidak ada data yang hilang.** Selisih itu adalah selisih *baris fisik*
vs *record CSV*, bukan record yang dibuang.

| Penghitungan | Nilai | Keterangan |
|---|---|---|
| Baris fisik `data/feedback.csv.bak` | 123 | 1 header + 122 baris data |
| Record CSV di `.bak` | 60 | `csv.DictReader` menghitung 60 record (ada field berkutip dengan newline) |
| Baris lanjutan di dalam record berkutip | 62 | 122 − 60 = 62 baris fisik bukan record baru |
| Record di `data/feedback.csv` hasil migrasi | 60 | sama dengan `.bak` |
| Record `.bak` yang tidak ada di CSV baru (setelah normalisasi whitespace) | 0/60 | Wilson 95% untuk 60/60 = **[93,98%, 100%]** |
| Pasangan `text`+`label` unik | 35/60 | 58,33%, Wilson 95% = **[45,73%, 69,94%]** (sisanya duplikat memang sudah ada di sumber) |

5 baris fisik yang tampak "hilang" bila menghitung `wc -l` (dipotong 80 karakter).
Semua adalah **baris lanjutan di dalam field berkutip / baris kosong**, bukan record:

```
baris  60: 'Selain itu, beredar juga informasi bahwa petugas lapangan tidak akan mendatangi '
baris  61: 'Harap berhati-hati dan jangan sampai ketinggalan informasi ini. Segera kirimkan '
baris  63: ''
baris  64: 'Dalam seminar bertajuk ""Rethinking: Data dan Kebijakan Pembangunan"" di Sekolah'
baris  65: ''
```

Bukti cadangan: `data/feedback.csv.bak` masih ada (123 baris fisik) dan masuk `.gitignore`
(`data/feedback.csv.bak`, `data/feedback*.csv`). `git log --all --oneline -- data/feedback.csv
data/feedback.csv.bak` **kosong** — tidak ada versi feedback yang pernah ter-commit/ter-push,
jadi tidak ada data terhapus dari riwayat. Dua record panjang hanya dinormalisasi
`\n\n` → `\n` saat konversi ke skema baru; isi kalimatnya tetap sama (pemeriksaan
`Counter` setelah `re.sub(r'\s+', ' ', text)` menghasilkan selisih 0 di kedua arah).

## 1b. Daftar nama tes sebelum/sesudah merge

Diekstrak dengan `git show <rev>:tests/test_*.py | grep '^\s*def test_'`, lalu
dibandingkan dengan `pytest --collect-only -q`.

| Revisi | Fungsi `test_*` | Test ter-koleksi |
|---|---:|---:|
| `f8903ba^1` | 81 | — |
| `f8903ba^2` | 58 | — |
| Gabungan (union) kedua induk | 81 | — |
| `f8903ba` (hasil merge) | 81 | 118 |
| `HEAD` saat ini | 83 | 120 |

- **Tes yang hilang: TIDAK ADA.** Union nama fungsi kedua induk (81) persis sama dengan
  daftar nama di commit merge (81); `comm -23 induk2 merge` menghasilkan 0 baris.
- Selisih 118 → 120 adalah **2 fungsi baru** yang ditambahkan setelah merge pada butir 3
  QA Round-2 (`test_predict_model_tidak_ada`, `test_health_model_tidak_ada`), bukan tes
  yang hilang. Dari 118 kasus pada `f8903ba`, sekarang 120 kasus; keduanya lulus
  (tingkat lulus 120/120 = 100%, Wilson 95% = **[96,90%, 100%]**).

## 1c. Sastrawi — dipakai atau tidak?

**Sastrawi MASIH DIPAKAI pada jalur training, jangan dihapus dari `requirements.txt`.**

Bukti baris kode:
- `id_preprocessing.py:23-30` — import opsional `StopWordRemoverFactory` dan
  `_SASTRAWI_STOPWORDS = set(StopWordRemoverFactory().get_stop_words())`.
- `id_preprocessing.py:41` — `INDONESIAN_STOPWORDS = sorted(_SASTRAWI_STOPWORDS - _KEEP_WORDS)`.
- `train_model_id.py:53` — mengimpor `INDONESIAN_STOPWORDS`.
- `train_model_id.py:151` — `TfidfVectorizer(..., stop_words=INDONESIAN_STOPWORDS)`.

`clean_text()` (`id_preprocessing.py:78-100`) **tidak** memakai stemming/Sastrawi,
klaim laporan sebelumnya itu benar untuk fungsi itu saja. Tetapi model Indonesia
**dilatih dengan stopword Sastrawi aktif**: 123 kata mentah, 102 kata efektif setelah
mengecualikan kata negasi/modality (102/123 = 82,93%, Wilson 95% =
**[75,31%, 88,55%]**). Bukti empiris pada `models/id/vectorizer.pkl` (21.251 feature,
n = 593 sampel latih menurut `models/id/metadata.json`): `yang`, `dan`, `di`,
`untuk`, `dengan`, `dari`, `pada`, `ini`, `itu`, `adalah` **tidak ada** di vocabulary,
sedangkan `tidak` dan `akan` (kata yang sengaja dipertahankan) **ada**.

Konsekuensi: menghapus Sastrawi dari `requirements.txt` akan membuat training ulang
memakai `stop_words=[]` dan menghasilkan vocabulary/model yang **berbeda** dari
artefak saat ini — skew baru, bukan perbaikan. Karena itu dependensi **dipertahankan
ter-pin** (`Sastrawi==1.0.1`) dan diberi catatan di `requirements.txt` (commit terpisah).
Tidak ada perubahan pada `clean_text`, jadi tidak ada skew training vs `/predict`.

## 1d. `/predict` tanpa model

- Sebelum: HTTP 500 + pesan umum.
- Sesudah: HTTP **503** + pesan Indonesia:
  "Model belum siap. Jalankan train_model.py (Inggris) atau train_model_id.py
  (Indonesia), lalu restart server. Lihat README.md bagian "Cara menjalankan dari nol"."
  Respons juga memuat `models_missing` dan `recovery` (`en`, `id`, `after`).
- `/health`: tetap 200, kini memuat `models_missing` (mis. `["en","id"]`),
  `model_recovery`, dan `models.<en|id>.missing`.
- UI: `templates/index.html` menangani `status === 503` secara khusus dan
  menampilkan pesan pemulihan dari server (tidak ditimpa pesan "server bermasalah").
- Tes: `test_predict_model_tidak_ada` (503 + field pemulihan),
  `test_health_model_tidak_ada` (`models_missing`), dan
  `test_ui_menampilkan_pesan_pemulihan_503` ditambah/diperkuat.

## 1e. Docker

Docker CLI ada di PATH, tetapi ** Docker daemon TIDAK tersedia** di distro WSL ini:
`docker info` keluar dengan status 1 dan pesan
"The command 'docker' could not be found in this WSL 2 distro ... aktifkan WSL integration di Docker Desktop".

Tidak dapat dijalankan: `docker build`, `docker run`, verifikasi user non-root,
folder feedback dapat ditulis, dan ukuran image. **Status: belum terverifikasi —
perlu uji manual di lingkungan dengan Docker Desktop/daemon aktif.** Tidak ada klaim
tentang image image size. (Dockerfile sudah menggunakan `appuser` non-root dan
`chown -R appuser /app`; itu hanya pemeriksaan statis, bukan bukti runtime.)

## 1f. CSP vs UI

Tidak ada headless browser (`chromium`, `chromium-browser`, `google-chrome`, `firefox`)
dan tidak ada Playwright di venv. **Status: perlu uji manual DevTools.**

Catatan statis (bukan pengganti uji browser): header CSP hanya dikirim untuk
respons `text/html` (`app.py:_set_security_headers`), mengizinkan `'unsafe-inline'`
untuk script/style karena `templates/index.html` memakai inline `<script>`, serta
`fonts.googleapis.com` / `fonts.gstatic.com` untuk font. Secara statis tidak ada
sumber eksternal lain yang dipakai UI, tetapi konsol browser tetap perlu dicek.

## 1g. Cek pra-push (TANPA push)

Perintah dijalankan; ringkasan keluaran:

- `git status --short` → **bersih** sebelum perubahan butir ini.
- `git log origin/fix/qa-round-2..HEAD --oneline` → 21 commit (teratas `0fa9026`,
  terbawah `ab03cb5`); **ahead 21, behind 0**.
- `git diff --stat origin/fix/qa-round-2...HEAD | tail -15` → 38 file,
  2238 insertions, 17637 deletions (dominan penghapusan artefak besar).
- `git ls-files | grep -Ei "\.pkl|\.npy|\.parquet|feedback|\.env|\.bak|\.coverage"`
  → hanya `scripts/import_feedback.py`, `scripts/review_feedback.py` (nama skrip,
  bukan data). Tidak ada `.pkl`, `.npy`, `.parquet`, `.env`, `.bak`, `.coverage`
  di tree `HEAD`.
- Pencarian rahasia pada `git diff origin/fix/qa-round-2...HEAD`
  (pola `SECRET_KEY=`, `api_key=`, `token=`, `password=`, `AKIA…`, `sk-…`, `ghp_…`,
  private key) → **tidak ada temuan**.
- File > 1 MB yang muncul di diff: `data/raw/rifky_indonesian_hoax.parquet`
  (57.400.873 byte) dan `data/train_id_oos.csv` (1.728.973 byte). Keduanya
  berstatus **D (deleted)** di `HEAD`, jadi **tidak ikut naik**; keduanya masih ada
  di tree `origin/fix/qa-round-2` dan di riwayat (butir 8, `HISTORY_REWRITE_PLAN.md`,
  tidak dijalankan tanpa persetujuan).

Tidak ada `git push`, `git force-push`, atau `git filter-repo` yang dijalankan.
