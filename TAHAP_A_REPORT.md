# Laporan TAHAP A — Audit Konfounder Sumber (Saring Berita)

**Tanggal:** 21 November 2026  
**Tujuan:** Memastikan model tidak belajar "gaya sumber" sebagai proxy kelas, dan menyiapkan hold-out independen v3 untuk evaluasi jujur.

---

## A.1 Matriks Sumber × Label (data latih, out-of-source train)

Sumber: `logs/source_x_label.json`

| Sumber | valid (0) | hoax (1) | Total |
|--------|-----------|----------|-------|
| HF/Rifky/indonesian-hoax-news | 3.556 | 4.384 | 7.940 |
| Mendeley/ICTS-2017 | 334 | 213 | 547 |

**Kesimpulan A.1:** Tidak ada sumber yang hanya memiliki satu kelas → **tidak ada konfounding sumber sempurna** (source–class confounding). Kedua kelas muncul di kedua sumber. Namun, data latih didominasi oleh satu sumber (Rifky/turnbackhoax.id, 7.940 dari 8.487 baris), sehingga model masih bisa belajar gaya gaya gaya gaya Rifky tanpa konfounding sempurna.

## A.2 Uji Konfounder dengan Fitur Gaya Saja

Skrip: `scripts/audit_confounder_style.py`  
Hasil: `logs/confounder_style.json`

Fitur gaya yang digunakan: panjang teks, jumlah kata, jumlah kalimat, kata/kalimat, rasio huruf kapital, tanda seru, tanya, dan awalan 3 kata pertama (one-hot top-20).

| Uji | Akurasi | ROC-AUC |
|-----|---------|---------|
| In-distribution (train) | 66,32% | 0,7577 |
| Hold-out v2 (out-of-source) | 55,34% | 0,4668 |
| Penurunan | 10,98% | — |

**Kesimpulan A.2:** Penurunan 10,98% tidak cukup besar untuk dianggap bukti kuat konfounder sumber. Model cenderung belajar isi, bukan hanya gaya sumber — namun sumber dominan (Rifky) masih berisiko memberikan bias gaya ringan.

## A.3 Interval Kepercayaan 95% (Wilson) — Akurasi Hold-out v2 (n=103)

Hold-out v2: 103 baris (57 valid / 46 hoak), sumber: `manual_holdout` (diberi label manual oleh pelaku proyek, bukan dari dataset publik otomatis). Teks ditulis ulang/parafrase untuk menghindari salin teks penuh berhak cipta.

| Model | Akurasi | CI95 (Wilson) | ≥ 85%? |
|-------|---------|---------------|--------|
| TF-IDF + LogisticRegression | 55,34% | [45,72%, 64,58%] | TIDAK |
| Sentence-Transformer + LogisticRegression | 67,96% | [58,44%, 76,19%] | TIDAK |
| IndoBERT (fine-tuned) | dilewati | — | — |

**Cek overlap CI:** CI kedu model tumpang tindih (YA).  
**Aturan:** Jangan memilih model berdasarkan selisih akurasi yang berada di dalam interval kepercayaan (tidak bermakna statistik). Model terbaik hold-out = 67,96%, masih **sangat di bawah 85%** → model tidak andal, `MODEL_RELIABLE=False`.

## A.4 Hold-out v3 (baru, belum digunakan untuk apa pun)

File yang disiapkan:
- `data/holdout_id_v3_template.csv` — template CSV (kolom: text, label, sumber_url, tanggal, catatan) dengan 5 contoh.
- `data/HOLDOUT_V3_LABELING_GUIDE.md` — panduan pelabelan singkat.

**Persyaratan hold-out v3:**
- ≥ 100 baris, seimbang ~50 valid / 50 hoaks (±10).
- Label oleh pengguna manual (tidak otomatis dari dataset publik).
- Tidak digunakan untuk pelatihan, tuning threshold, atau pemilihan model.
- Jangan salin teks penuh artikel berhak cipta secara massal; simpan judul + tautan + ringkasan pendek.
- Simpan hasil akhir di `data/holdout_id_v3.csv`.

---

## Ringkasan TAHAP A

| Item | Temuan |
|------|--------|
| Konfounding sumber sempurna | Tidak ada (kedua kelas di kedua sumber) |
| Sumber dominan bias | Ya (Rifky = 7.940/8.487 baris) |
| Fitur gaya = bukti konfounder | Tidak conclusif (drop 10,98% only) |
| Model terbaik hold-out | 67,96% (Sent-Trans + LR) < 85% |
| CI overlap antar model | Ya → jangan pilih model dari selisih kecil |
| MODEL_RELIABLE | False (tetap) |
| Hold-out v3 siap | Template + panduan sudah ada, belum diisi data |

TAHAP A selesai. Lanjut TAHAP B (pencocokan dengan hoaks yang sudah dibantah).
