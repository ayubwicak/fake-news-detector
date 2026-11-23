# Sumber Data, Lisensi, dan Kebijakan Split

Dokumen ini merangkum asal data latih proyek **Saring Berita**, status lisensinya,
dan — yang terpenting — **bagaimana data dibagi agar metrik tidak menipu**.
Dibuat sebagai bagian dari Butir 5.

## 1. Sumber yang dipakai

Sumber dibaca oleh `scripts/prepare_dataset_id.py` (tidak ada scraping baru;
seluruh unduhan sudah dilakukan sebelumnya dan artefak mentah tidak ikut
di-commit).

| Sumber | Lisensi | Dipakai? | Catatan |
|---|---|---|---|
| Mendeley / ICTS-2017 (Pratiwi, Asmara & Rahutomo) | CC BY 4.0 | Ya | ~600 artikel, voting 3 anotator, 12 topik |
| HF `nlp-brin-id/id-hoax-report-merge` | MIT | Ya | butuh login HF; dilewati bila tidak terautentikasi |
| HF `nlp-brin-id/id-hoax-report` | MIT | Ya | butuh login HF; dilewati bila tidak terautentikasi |
| HF `Rifky/indonesian-hoax-news` | tanpa dataset card | opsional | butuh login HF |
| Hold-out manual (v2) | original (penulis) | Ya | ≥100 baris, seimbang, terpisah dari latih |
| Hold-out v3 | original (pelabelan manusia) | **hanya laporan akhir** | lihat `data/HOLDOUT_V3_LABELING_GUIDE.md` |

Sumber yang **tersedia tapi tidak diunduh** (butuh login Kaggle/HF, dicatat agar
tidak dianggap terpakai): Kaggle *Indonesian Hoax News Dataset (Kominfo)*
(Apache 2.0), Kaggle *Deteksi Berita Hoaks Indo* (research only), Kaggle
*Indonesian Fact and Hoax Political News* (MIT), HF `Rifky/indonesian-hoax-news`.

Rincian jumlah baris per sumber ada di `logs/dataset_id_audit.json` (field
`sources_attempted`, `source_x_label`), yang dihasilkan ulang setiap kali
`prepare_dataset_id.py` dijalankan.

## 2. Kebijakan split — default **GroupKFold by source**

Masalah dengan split acak biasa: dua artikel dari **sumber/portal yang sama**
sering memakai gaya bahasa, template, dan kata kunci yang mirip. Bila sumber yang
sama muncul di latih **dan** test, model bisa "mengenali" portalnya, bukan
memahami hoaks — metrik jadi optimistis (kebocoran gaya sumber).

Karena itu:

- **Default:** `scripts/prepare_dataset_id.py` membagi data dengan
  `group_kfold_split()`, yang mengelompokkan (`groups=`) berdasarkan kolom
  `source` memakai `sklearn.model_selection.GroupKFold`. **Tidak ada satu sumber
  pun yang muncul di train dan test sekaligus.** Fold dipilih secara
  deterministik dengan proporsi test terdekat ke `--test-size` (default 0.20).
- Rincian fold (metode, jumlah sumber di train/test, proporsi) disimpan di
  `logs/dataset_id_audit.json` → `split_info`.
- `--random-split` adalah **opt-out eksplisit** untuk smoke test. Skrip mencetak
  peringatan keras dan `split_info.warning = "source leakage possible"`.
  **Dilarang** memakai hasil split acak untuk klaim metrik.
- `data/holdout_id_v2.csv` dan `data/holdout_id_v3.csv` **tidak pernah**
  dipakai untuk training atau tuning; keduanya independen dari split di atas.

## 3. Reproduksi

```bash
python scripts/prepare_dataset_id.py                 # default: GroupKFold by source
python scripts/prepare_dataset_id.py --test-size 0.3 # proporsi test lain
python scripts/prepare_dataset_id.py --random-split  # HANYA smoke test (ada peringatan)
```

Keluaran: `data/train_id.csv`, `data/train_id_oos.csv`, `data/test_id_oos.csv`,
`data/sample_news_id.csv`, `data/holdout_id_v2.csv`, `logs/dataset_id_audit.json`,
`logs/source_x_label.json`.

## 4. Pembersihan & jejak audit

`clean_and_dedup()` melakukan: buang prefiks label (`[SALAH]`, `HOAX:` ...), buang
marker portal/tanggal/URL, normalisasi NFKC + lowercase, hapus kata bocor
(`hoax`, `cek fakta`, `klarifikasi`, ...), serta deduplikasi near-duplicate
(Jaccard 5-gram > 0.85). Semua langkah tercatat di `logs/dataset_id_audit.json`
(field `cleaning`). Label: `0` = Valid/Real, `1` = Hoax/Fake.

> Catatan jujur: dataset **tidak seimbang lintas sumber** dan sebagian sumber
> berita memiliki gaya khas. Meskipun split by-source mengurangi kebocoran,
> generalisasi ke portal/newspaper baru tetap menjadi keterbatasan — lihat
> `LAPORAN_AKHIR.md`.
