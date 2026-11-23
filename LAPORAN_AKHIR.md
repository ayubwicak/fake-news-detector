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

---

# Hasil Semua Butir QA Round-2 (fix/qa-round-2)

## Butir 1 — LAPORKAN HASIL BUTIR 0 (skew vs /predict + Sastrawi)

**Tidak ada perubahan kode.** Hasil verifikasi empiris:

- **Skew training vs /predict:** TIDAK ada skew. Kedua jalur pakai `clean_text` dari `id_preprocessing.py` yang sama (`app.py:908` & `:1008`, `train_model_id.py:53` & `:112`). Output identik untuk 3 contoh uji.
- **Sastrawi:** Terpasang di venv (`_SASTRAWI_STOPWORDS = 123 kata`), tapi `clean_text()` **tidak memakainya** (hanya regex URL/HTML/email/boilerplate/source/meta/lowercase/non-alpha/ws-collapse). Di lingkungan bersih (tanpa Sastrawi), `except` → `_SASTRAWI_STOPWORDS = set()`, dan output `clean_text` tetap identik.
- **Kesimpulan:** tidak terbukti skew/perbedaan → **tidak perlu evaluasi ulang hold-out**.

## Butir 2 — MERGE A.3 (origin/fix/qa-round-2, disetujui)

- Merge commit `4f79230` (satu commit di origin) ke branch lokal.
- Konflik `app.py` diselesaikan dengan mempertahankan perbaikan lokal: `import unicodedata` + NFKC di `/predict` dan `/feedback`, `parse_feedback_label`, pesan error Bahasa Indonesia, `FEEDBACK_MAX_CHARS`.
- Commit `f8903ba`. Tes 118 lulus. **Tidak push.**

## Butir 3 — UJI CLONE BERSIH di /tmp + dokumentasi model + tes "model tidak ada"

- Clone bersih ke `/tmp/opencode/saring-clone-test`, venv baru, `pip install -r requirements.txt`.
- `models/id/*.pkl` tidak ada di clone (git-ignored, benar).
- `GET /health`: 200, `status: "healthy"`, `model_loaded: false`, detail per-model EN/ID tidak tersedia — tidak crash.
- `POST /predict` (Indonesia + Inggris): 500 + pesan jelas Bahasa Indonesia `"Model belum siap. Jalankan train_model.py terlebih dahulu."` — tidak crash.
- Procfile benar (Gunicorn). Docker tidak tersedia di lingkungan uji (WSL2 tanpa Docker Desktop).
- README: dokumentasi jalur resmi mendapatkan artefak model (`scripts/prepare_dataset_id.py` → `train_model_id.py --allow-sample`, atau `train_model.py` untuk Inggris), verifikasi `CHECKSUMS.txt`, alternatif GitHub Release.
- Tes baru: `test_predict_model_tidak_ada`, `test_health_model_tidak_ada`.
- Commit `77d66f0`. Jumlah tes: 120.

## Butir 4 — FEEDBACK LAMA: migrasi 122 baris + `.bak` di-gitignore

- `data/feedback.csv` 122 baris format lama (`text,label`) diverifikasi **tidak pernah ter-commit / ter-push** (`git log --all` kosong; lokal saja).
- 60 CSV records (baris `wc -l` 122 karena baris kosong + multi-line quoted fields) dikonversi ke format baru `id,timestamp,text_hash,text,label,status,reviewer`; semua `status="baru"`, reviewer kosong.
- Asli dicadangkan ke `data/feedback.csv.bak` (di-gitignore via `*.bak`).
- `.gitignore`: tambah `data/feedback.csv.bak`.
- Commit `a6b6b85`. Tes 120 lulus.

## Butir 5 — Kelompok (b) kecil (S06, P02, PR01, S07, .coverage gitignore)

- **S06 security headers:** `@app.after_request` → CSP (sumber skrip/gaya/font dibatasi; inline untuk UI), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy` minimal.
- **P02 Docker non-root + direktori tulis:** `groupadd/useradd appuser`, `chown -R appuser /app`, `USER appuser`.
- **PR01 privacy notice + FEEDBACK_ENABLED:** `privacy-notice` di UI dekat textarea (jangan kirim data pribadi); `FEEDBACK_ENABLED` env (default `true`), `false` → 503 di `/feedback`.
- **S07 SECRET_KEY/CORS:** `SECRET_KEY` hanya dari `os.environ` (tidak ada default di repo); CORS tidak diaktifkan (tidak ada `flask-cors`, tidak ada header ACAO).
- `.gitignore`: tambah `*.bak`, `.coverage`, `htmlcov/`.
- README: bagian **🔒 Keamanan & Privasi** (S06, S07, PR01, FEEDBACK_ENABLED, retensi/hapus data).
- Commit `6d0bf5d`. Tes 120 lulus.

## Butir 6 — B.5 data/README_data.md (asal, labeler, holdout belum diverifikasi, lisensi)

- Tabel sumber data ditambah kolom **Labeler**: Sumber 1 (Rifky) otomatis dari awalan `[SALAH]`/`[BENAR]` (tanpa anotator); Sumber 2 (ICTS-2017) 3 anotator dengan voting; Sumber 3 (manual hold-out) penulis satu orang tanpa cross-check.
- `holdout_id_v2.csv` ditandai **belum diverifikasi secara independen**.
- Ditambah bagian **⚠️ Status verifikasi hold-out**: metrik sebagai indikasi awal, akurasi 67,96% < 85%, `MODEL_RELIABLE=false`.
- Lisensi per sumber tetap tercatat (CC BY 4.0, public/no card, asli tulis ulang).
- Commit `ca2f7f8`. Tes 120 lulus.

## Butir 7 — Scraping review + B615 pin HF revision + model card

**Scraping review** (skrip di `scripts/scraping/` hanya referensi, **TIDAK dijalankan**):
- Timeout: setiap `requests.get(...)` pakai `timeout=15` detik.
- User-Agent: sudah ada, browser UA yang menjelaskan identitas klien.
- `robots.txt` / Aturan Layanan: dicatat di setiap skrip (URL `robots.txt` per situs; perlu dicek manual sebelum scrape).
- Tidak dijalankan otomatis (tidak ada pemanggilan di `app.py` / `train_model*.py` / CI).

**B615 pin HF revision:** INDOBERT dimuat dari folder lokal `models/indobert/` (bukan HF Hub langsung). Catatan di README + `models/model_card_id.md`: jika diunduh dari HF Hub, harus pakai `revision` yang di-pin (commit hash/tag).

**Model card:** `models/model_card_id.md` ditambahi bagian B615.
- Tidak ada scraping yang dijalankan.
- Commit `a0c7ceb`. Tes 120 lulus.

## Butir 8 — HISTORY_REWRITE_PLAN.md (tidak dijalankan)

- Dibuat `HISTORY_REWRITE_PLAN.md`: rencana hapus `models/id/*.pkl`, `feedback.csv`, file besar (`data/raw/`, `data/raw_*.csv`, `*_oos.csv`), file hak cipta media komersial dari riwayat git.
- Opsi repositori private sebelum publikasi.
- Langkah rencana (backup branch, filter-repo, git gc, force-push setelah persetujuan, re-clone) — **TIDAK DIJALANKAN**.
- Catatan: `git rm --cached` sudah dilakukan (c6c58f0) tapi riwayat origin masih mengandung file besar.
- Commit `bb44964`. Tes 120 lulus.

---

## Ringkasan tes

- Jumlah tes: **120 passed, 0 failed** (`python -m pytest tests/ -v`).
- Sebelum butir 3: 118 passed. Setelah tambah 2 tes "model tidak ada": 120 passed.
- Setiap butir diikuti oleh `python -m pytest tests/ -v` dan semua lulus.

## Git status

- Branch: `fix/qa-round-2`
- Working tree: **bersih** (tidak ada file modified/untracked; `.coverage` di-gitignore).
- Terhadap `origin/fix/qa-round-2`: **ahead 20, behind 0**.
- Tidak ada push dilakukan selama seluruh proses.

## Git log --oneline --graph -n 25

```
* bb44964 8: HISTORY_REWRITE_PLAN.md — rencana hapus file besar/hak cipta (tidak dijalankan)
* a0c7ceb 7: Scraping review + B615 pin HF revision + model card
* ca2f7f8 6: B.5 data/README_data.md — kolom labeler + holdout belum diverifikasi
* 6d0bf5d 5: Kelompok (b) kecil — S06 headers, P02 non-root Docker, PR01 privasi, S07 SECRET_KEY/CORS, .coverage gitignore
* a6b6b85 4: gitignore backup feedback.csv.bak
* 77d66f0 3: Uji clone bersih + dokumentasi model + tes model tidak ada
*   f8903ba Merge origin/fix/qa-round-2 (A.3) — pertahankan perbaikan lokal
|\  
| *   4f79230 Merge branch 'fix/qa-bug-fixes' into fix/qa-round-2
| |\  
* | | 0c6db1d 9(ikutan): perbaikan temuan audit ruff SIM115
* | | 87e3f1f 8: Dokumentasi ambang per-model + hapus klaim split-by-source + fallback keras
* | | 3da70d9 7: L06 lengkapi blok pass build_advice matriks status x red_flag
* | | c6c58f0 6: R01 gitignore + git rm --cached + CHECKSUMS + verifikasi startup
* | | 51732f7 5: F01 URL aman di UI + server-side filter skema
* | | d28de76 4: L01/S05 feedback aman + review flow + kolom baru
* | | d486515 3: S02 rate limit ProxyFix + TRUSTED_PROXIES
* | | 3a64eb5 2: S03/L05 batas input + anti ReDoS domain regex
* | | a705dfb 1: S01/P01 nonaktifkan debug default, Gunicorn di Docker, host dari env
* | | 19faa6a 0c: hapus clean_text lokal (kode mati), pin Sastrawi, tes kesetaraan pembersihan training/inferensi
* | | dce24bd Debunk fallback, teks pendek, dan dependensi ML
* | | 263e69b TAHAP audit: file verifikasi kebocoran, debunk CV, dan bootstrap gaya
* | |   ab03cb5 Merge fix/qa-bug-fixes: resolusi konflik app.py
|\ \ \  
| |/ /  
|/| /   
| |/    
| * 29b32b9 Fix QA bugs: BUG-001 input type validation, BUG-002 feedback label validation order
* | 5ec21e5 Add model comparison and evaluation scripts for Indonesian fake-news detection
* | 5e12a73 TAHAP D: gitignore dan kebersihan README
* | 51d0f62 TAHAP C: siapkan fine-tuning IndoBERT di GPU (Google Colab)
```

## Daftar butir yang butuh persetujuan sebelum push

1. **Butir 2 (MERGE A.3):** sudah disetujui, tapi berisiko konflik (`app.py` berbeda origin vs lokal) — telah di-resolve dengan mempertahankan perbaikan lokal; butuh konfirmasi akhir sebelum push.
2. **Butir 8 (HISTORY_REWRITE_PLAN / filter-repo):** **TIDAK dijalankan** dan **dilarang** force-push/git filter-repo tanpa persetujuan; file besar masih di riwayat origin. Butuh persetujuan tim bila ingin membersihkan riwayat.
3. **Push keseluruhan branch:** belum dilakukan; siap untuk push setelah persetujuan pengguna.

---

## Catatan kendala

- **B.4 git filter-repo + force-push:** DILARANG oleh instruksi; file besar masih di riwayat origin (hanya `git rm --cached` di commit lokal).
- **Docker build:** tidak bisa diuji di lingkungan ini (WSL2 tanpa Docker Desktop); Procfile dan Dockerfile sudah divisualisasikan.
- **MODEL_RELIABLE:** tetap `false` (hold-out 67,96% < 85%); tidak diubah.
- **Field lama `/predict` dan tes:** tidak dihapus.

---

**Tanggal:** 23 November 2026  
**Branch:** `fix/qa-round-2` (lokal, belum push)
