# LAPORAN AKHIR — Proyek "Saring Berita" (TAHAP A–D)

**Tanggal:** 23 November 2026 (pembaruan penutupan)  
**Branch:** `fix/qa-round-2` (lokal, **belum push**, ahead 29 / behind 0 terhadap origin — setelah commit butir 7)  
**Peringatan keandalan (seluruh laporan):** Hold-out v2 terbaik = **67,96% < 85%**. Model **tidak dianggap andal**. `MODEL_RELIABLE=false`. Jangan mengklaim model andal untuk deteksi hoaks produksi Indonesia. Sejak Butir 4, flag ini **dihitung otomatis** oleh `reliability.py` dari hasil hold-out v3 — karena v3 **belum dievaluasi**, hasilnya tetap `false` (bukan lagi dibaca dari model card).

> **Urutan baca:** bagian "PENUTUPAN — 7 Butir" di akhir dokumen adalah sumber kebenaran
> terbaru. Angka lama di bagian TAHAP A–D dan QA Round-2 (butir 1–8) adalah catatan
> historis dan sudah dikoreksi di bagian penutupan bila bertentangan.

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

- `notebooks/finetune_indobert_colab.ipynb`: **diperbarui di Butir 6** — seed 42, early stopping (maks 6 epoch, patience 2, val by-source), revision HF di-pin (`c2cd0b51...`), evaluasi test out-of-source + hold-out v2, `metrics.json` + `model_card.md` + `checksums.txt` + `indobert.zip` + `SHA256SUMS.txt`. **Tidak memakai v3.**
- `README_colab.md`: panduan unggah → run all → verifikasi `sha256sum -c` → ekstrak ke `models/indobert/`.
- `app.py`: loader IndoBERT **hanya memuat bila `models/indobert/checksums.txt` ada dan semua checksum cocok**; kalau tidak, fallback ke TF-IDF. Status terlihat di `/health` → `indobert`. **Koreksi:** `MODEL_RELIABLE` TIDAK lagi dinilai dari `metrics.json`/model card — dihitung `reliability.py` dari v3 (Butir 4).
- **Belum ada model IndoBERT terlatih yang tersedia**, sehingga `MODEL_RELIABLE` tetap `false`.

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
- **Sastrawi — KOREKSI (Butir 1 penutupan):** `clean_text()` memang **tidak** memakai Sastrawi, **tetapi jalur training memakainya** sebagai stopword: `id_preprocessing.py:23-41` → `train_model_id.py:151` (`stop_words=INDONESIAN_STOPWORDS`, 102 kata efektif dari 123). Menghapus dependensi akan mengubah vocabulary hasil training ulang. Karena itu Sastrawi **dipertahankan ter-pin** (`Sastrawi==1.0.1`) dengan catatan di `requirements.txt`. Tidak ada skew training vs `/predict` karena keduanya memakai `clean_text` yang sama. Detail: `CLOSING_AUDIT.md`.
- **Kesimpulan:** tidak terbukti skew pada `clean_text` → **tidak perlu evaluasi ulang hold-out**; Sastrawi tetap dipertahankan untuk training.

## Butir 2 — MERGE A.3 (origin/fix/qa-round-2, disetujui)

- Merge commit `4f79230` (satu commit di origin) ke branch lokal.
- Konflik `app.py` diselesaikan dengan mempertahankan perbaikan lokal: `import unicodedata` + NFKC di `/predict` dan `/feedback`, `parse_feedback_label`, pesan error Bahasa Indonesia, `FEEDBACK_MAX_CHARS`.
- Commit `f8903ba`. Tes 118 lulus. **Tidak push.**

## Butir 3 — UJI CLONE BERSIH di /tmp + dokumentasi model + tes "model tidak ada"

- Clone bersih ke `/tmp/opencode/saring-clone-test`, venv baru, `pip install -r requirements.txt`.
- `models/id/*.pkl` tidak ada di clone (git-ignored, benar).
- `GET /health`: 200, `status: "healthy"`, `model_loaded: false`, detail per-model EN/ID tidak tersedia — tidak crash.
- `POST /predict` (Indonesia + Inggris): **503** + pesan jelas Bahasa Indonesia + `models_missing` + `recovery` — tidak crash. (Koreksi Butir 1 penutupan: sebelumnya 500.)
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

## Ringkasan tes (historis QA Round-2)

- Jumlah tes pada akhir QA Round-2: **120 passed, 0 failed**.
- Sebelum butir 3: 118 passed. Setelah tambah 2 tes "model tidak ada": 120 passed.
- Setiap butir diikuti oleh `python -m pytest tests/ -v` dan semua lulus.
- **Pembaruan penutupan: 160 passed** (lihat bagian PENUTUPAN di akhir).

## Git status

- Branch: `fix/qa-round-2`
- Working tree: **bersih** (tidak ada file modified/untracked; `.coverage` di-gitignore).
- Terhadap `origin/fix/qa-round-2`: **ahead 20, behind 0** (pada akhir QA Round-2; **sekarang ahead 28**, lihat penutupan).
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
- **MODEL_RELIABLE:** tetap `false`; sekarang **dihitung** dari v3 (belum dievaluasi), bukan diset manual dari 67,96%. Tidak ada threshold/model yang disetel berdasarkan v3.
- **Field lama `/predict` dan tes:** tidak dihapus.

---

**Tanggal:** 23 November 2026  
**Branch:** `fix/qa-round-2` (lokal, belum push)

---

# PENUTUPAN — 7 Butir (23 November 2026)

Ini adalah bagian **terbaru dan paling lengkap**. Setiap butir dikerjakan berurutan,
**satu commit per butir**, dengan `python -m pytest tests/ -v` setelahnya.
Tidak ada push, tidak ada scraping, tidak ada tuning/set ambang berdasarkan v3.

## D.1 Ringkasan 7 butir

| Butir | Isi | Commit | Hasil |
|---|---|---|---|
| 1 | Tutup ketidakkonsistenan laporan (feedback 122/60, nama tes, 503/health, UI, Docker/CSP, pra-push) + catatan Sastrawi | `ba973dd`, `1454e67` | Selesai; `CLOSING_AUDIT.md` |
| 2 | Perangkat pelabelan v3 (panduan, template, validator, agreement/kappa, gitignore) | `b79a7a9` | Selesai; belum ada data v3 |
| 3 | Evaluator v3 satu pintu (guard validator + hash + anti-rerun, metrik lengkap) | `1fa58e5` | Selesai; **belum dijalankan** |
| 4 | Gerbang keandalan otomatis `MODEL_RELIABLE` dari hasil v3 | `26f8aa7` | Selesai; hasil = **false** |
| 5 | Data tanpa scraping: GroupKFold by-source, `--random-split` warning, `docs/DATA_SOURCES.md`, tes pipeline | `202224e` | Selesai |
| 6 | Notebook Colab siap (seed, early stopping, revision pin, checksum/zip/SHA-256, loader only-on-checksum) | `c69f6d2` | Selesai; belum di-run (butuh GPU) |
| 7 | Laporan akhir + bukti (bagian ini) | commit butir ini | Selesai |

## D.2 Tabel klaim — apa adanya, dengan sampel & Wilson 95%

| Klaim | Bukti / angka | n | Wilson 95% | Status |
|---|---|---:|---|---|
| 160/160 tes lulus | `python -m pytest tests/ -v` → `160 passed in 13.86s` | 160 | **[97,66%, 100%]** | ✅ |
| Tidak ada feedback hilang saat migrasi 122→60 baris | 60 record `.bak` = 60 record `.csv` setelah normalisasi whitespace | 60 | **[93,98%, 100%]** | ✅ |
| Keunikan pasangan `text`+`label` (duplikat memang ada di sumber) | 35 unik dari 60 | 60 | **[45,73%, 69,94%]** | ⚠️ 25 duplikat |
| Tidak ada nama tes hilang saat merge | 81 fungsi di induk = 81 di merge (`comm -23` = 0 baris) | 81 | **[95,47%, 100%]** | ✅ |
| Sastrawi dipakai sebagai stopword training | 102 kata efektif dari 123 kata mentah; kata negasi dipertahankan | 123 | **[75,31%, 88,55%]** | ✅ dipertahankan |
| Model Indonesia andal (≥85%) | Terbaik v2 = 67,96%, CI [58,44%, 76,19%]; v3 belum dievaluasi | 103 | — | ❌ `MODEL_RELIABLE=false` |
| Presisi pencocokan klaim dibantah | 13 cocok dari 15 kecocokan (ambang 0,66 dipilih di v2) | 15 | **[62,12%, 96,26%]** | ⚠️ CI lebar, bias seleksi |
| Recall hoak pencocokan klaim | 13/46 | 46 | **[17,32%, 42,55%]** | ❌ rendah |
| False-match pada berita valid | 2/57 = 3,51% | 57 | **[0,97%, 11,92%]** | ✅ ≤ 5% |
| Docker build/run terverifikasi | `docker info` gagal (WSL2 tanpa daemon) | — | tidak ada (deterministik) | ⚠️ **belum terverifikasi** |
| CSP bebas error di browser | Tidak ada headless browser/Playwright | — | tidak ada (deterministik) | ⚠️ **perlu uji manual DevTools** |
| Hold-out v3 dipakai sekali untuk laporan | Belum diisi/dievaluasi; skrip satu pintu sudah menunggu | — | prosedural | ✅ belum dilanggar |

Catatan: untuk proporsi deterministik (mis. keberadaan berkas) tidak ada makna statistik;
kolom CI sengaja dikosongkan. Semua interval di atas adalah **Wilson 95%**.

## D.3 Keterbatasan (jangan disembunyikan)

1. **Model belum andal.** 67,96% pada v2 (n=103) jauh di bawah 85%; CI dua model
   tumpang tindih. `MODEL_RELIABLE=false` dan UI menampilkan mode eksperimental.
2. **Hold-out v2 belum diverifikasi independen** (label manual satu orang, tanpa
   cross-check). v3 sedang disiapkan tapi **belum berisi data**, jadi tidak ada angka
   akurasi final dari v3 sampai pelabelan selesai.
3. **Bias sumber:** satu sumber (Rifky/turnbackhoax.id) ≈ 7.940/8.487 baris latih.
   GroupKFold by-source mengurangi kebocoran, tetapi generalisasi ke portal baru
   tetap belum teruji. `--random-split` masih bisa membocorkan gaya sumber.
4. **Fitur pencocokan klaim lemah:** recall hoak hanya 28,26% (n=46); ambang 0,66
   dipilih di v2 → angka presisi mengandung selection bias. Kemiripan teks **bukan**
   bukti hoaks.
5. **IndoBERT belum terlatih** di repo ini. Notebook butuh GPU Colab; belum di-run,
   jadi tidak ada klaim performa IndoBERT.
6. **Docker & browser belum diuji** di lingkungan ini (lihat tabel klaim).
7. **Riwayat git masih mengandung file besar** (`data/raw/rifky_indonesian_hoax.parquet`
   ≈57 MB, `data/train_id_oos.csv` ≈1,7 MB) di `origin`. `HISTORY_REWRITE_PLAN.md`
   hanya rencana; **tidak dijalankan** (dilarang force-push/filter-repo tanpa izin).
8. **Tidak ada push** — 29 commit lokal belum dikirim ke origin.
9. **Satu lookup metadata HF** (`GET https://huggingface.co/api/models/indobenchmark/indobert-base-p1`)
   dilakukan hanya untuk mengambil commit hash pin. Ini **bukan scraping dataset**;
   tidak ada skrip `scripts/scraping/` yang dijalankan dan tidak ada artikel/teks
   berita yang diunduh.

## D.4 Prioritas kerja berikutnya (terurut)

1. **Isi hold-out v3** oleh ≥2 anotator (`data/HOLDOUT_V3_LABELING_GUIDE.md`),
   jalankan `scripts/validate_holdout_v3.py` sampai `pass` dan `scripts/agreement.py`
   untuk kappa. **Ini prasyarat angka akurasi final.**
2. **Jalankan sekali** `scripts/evaluate_on_holdout_v3.py` → `logs/holdout_v3_result.json`.
   Setelah itu `MODEL_RELIABLE` dihitung otomatis; jangan jalankan ulang tanpa
   `--rerun-acknowledged` (dicatat di `logs/holdout_v3_run_attempts.jsonl`).
3. **Perbaiki model** bila gerbang gagal: kumpulkan sumber berita valid tambahan
   (tanpa scraping otomatis), seimbangkan kelas, latih ulang dengan GroupKFold.
   **Jangan** menyetel ambang/model berdasarkan hasil v3 yang sama.
4. **Uji manual Docker** (build/run, user non-root, folder feedback writable, ukuran
   image) dan **CSP via DevTools** di browser nyata.
5. **Jalankan notebook Colab** bila disetujui → verifikasi `SHA256SUMS.txt` → taruh
   di `models/indobert/` → loader otomatis aktif hanya bila checksum cocok.
6. **Minta persetujuan** untuk push dan (terpisah) untuk history rewrite bila ingin
   repo publik.

## D.5 Panduan menjalankan dari clone bersih (1 halaman)

```bash
# 1. Clone & environment
git clone <URL> && cd fake-news-detector
git checkout fix/qa-round-2
python3 -m venv venv && source venv/bin/activate      # Win: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt                   # pytest & notebook

# 2. Siapkan data & latih model Indonesia
python scripts/prepare_dataset_id.py                  # default GroupKFold by-source
python train_model_id.py --allow-sample               # atau tanpa flag bila data lengkap
# (opsional Inggris) python scripts/prepare_dataset.py && python train_model.py

# 3. Verifikasi artefak & jalankan server
cd models && sha256sum -c CHECKSUMS.txt && cd ..
python -m pytest tests/ -v                            # harus 160 passed
python app.py                                          # atau: gunicorn app:app

# 4. Cek kesehatan
curl -s localhost:5000/health | python -m json.tool
# perhatikan: models_missing, model_reliability.reliable, indobert.checksums_ok
curl -s -X POST localhost:5000/predict -H 'Content-Type: application/json' \
  -d '{"text":"Pemerintah mengumumkan kebijakan baru."}'
# bila model belum ada -> HTTP 503 + models_missing + recovery

# 5. (Hanya bila ingin angka final) pelabelan v3 — sekali
# isi data/holdout_id_v3.csv menurut data/HOLDOUT_V3_LABELING_GUIDE.md
python scripts/validate_holdout_v3.py                 # harus status=pass
python scripts/agreement.py                           # kappa dua anotator
python scripts/evaluate_on_holdout_v3.py              # SATU KALI; menghasilkan result JSON
# MODEL_RELIABLE otomatis dihitung ulang saat app.py start

# 6. (Opsional) IndoBERT via Colab -> indobert.zip + SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt && unzip indobert.zip && cp -r indobert/. models/indobert/
```

Catatan: `SECRET_KEY` harus diset lewat environment variable di produksi; CSP/headers
sudah aktif. Jangan menjalankan skrip di `scripts/scraping/`.

## D.6 Hasil tes akhir

```
$ python -m pytest tests/ -v
...
tests/test_training_pipeline.py::test_bootstrap_ci_mengandung_estimasi_titik PASSED [100%]
============================= 160 passed in 13.86s =============================
```

**160/160 lulus, 0 gagal** (Wilson 95% **[97,66%, 100%]**). Setiap butir (1–7)
diikuti tes penuh; tidak ada tes yang dihapus/dilemahkan, tidak ada field lama
`/predict` yang dihapus.

## D.7 Git status (saat laporan dibuat)

```
$ git status --short
(no output — working tree clean)

$ git rev-list --left-right --count origin/fix/qa-round-2...HEAD
0	28        <- ahead 28, behind 0 (sebelum commit butir 7)
```

Snapshot di atas diambil **sebelum** commit butir 7 dibuat, sehingga kondisi akhir
adalah **ahead 29, behind 0** dengan commit butir 7 sebagai teratas (`git log -1 --oneline`
untuk hash terbaru). Tidak ada `git push`, `git force-push`, atau `git filter-repo`
yang dijalankan.

## D.8 `git log --oneline --graph -n 30`

Snapshot sebelum commit butir 7 (commit butir 7 akan muncul sebagai baris teratas
pada `git log` terbaru; hash-nya dicek dengan `git log -1 --oneline`):

```
* c69f6d2 6: Notebook Colab siap — seed, early stopping, revision pin, checksum+zip+SHA-256, loader hanya saat checksum cocok
* 202224e 5: Data tanpa scraping — GroupKFold by-source default, --random-split warning, docs sumber, tes pipeline
* 26f8aa7 4: Gerbang keandalan otomatis dari hasil v3 — MODEL_RELIABLE dihitung, UI berganti mode
* 1fa58e5 3: Evaluator hold-out v3 satu pintu — gerbang validator, anti-rerun, metrik lengkap
* b79a7a9 2: Perangkat pelabelan hold-out v3 — panduan, validator, agreement, tes
* 1454e67 1c: Sastrawi tetap ter-pin — dipakai sebagai stopword training, bukan stemming
* ba973dd 1: Tutup ketidakkonsistenan laporan — feedback 122/60, nama tes, 503/health, audit pra-push
* 0fa9026 9: Laporan akhir — hasil semua butir QA Round-2
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
```

## D.9 Butir yang butuh persetujuan

1. **Push 28 commit** ke `origin/fix/qa-round-2` — belum dilakukan; **butuh persetujuan**.
2. **History rewrite** (`HISTORY_REWRITE_PLAN.md`) — **dilarang** tanpa persetujuan
   eksplisit; tidak ada force-push/filter-repo yang dijalankan.
3. **Pelabelan hold-out v3 oleh manusia** — tidak bisa diotomatisasi; harus disetujui
   dan dikerjakan bahwa penulis/anotator sebelum ada angka akurasi final.
4. **Menjalankan notebook Colab** (butuh GPU & kuota) — butuh persetujuan; hasilnya
   baru boleh dipakai bila checksum lolos.
5. **Uji Docker & CSP manual** — butuh lingkungan dengan daemon Docker dan browser.

## D.10 Ringkasan jujur

Proyek ini **rapi secara rekayasa** (160 tes lulus, satu pintu evaluasi v3, gerbang
keandalan otomatis, loader model dengan checksum, tidak ada scraping/push diam-diam),
tetapi **belum membuktikan akurasi produksi**: model terbaik hanya 67,96% pada v2,
v3 belum berisi data, dan Docker/CSP belum teruji. Semua angka di atas disertai ukuran
sampel dan interval Wilson 95% sejauh bersifat proporsi; tidak ada angka yang diambil
dari v3 untuk tuning, karena v3 belum pernah dievaluasi.
