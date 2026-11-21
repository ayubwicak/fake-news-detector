# Panduan Fine-tuning IndoBERT di Google Colab (TAHAP C)

File notebook: `notebooks/finetune_indobert_colab.ipynb`

## Tujuan
Fine-tuning `indobenchmark/indobert-base-p1` pada GPU gratis Google Colab untuk deteksi hoaks bahasa Indonesia, dengan evaluasi jujur pada test out-of-source dan hold-out independen. Model dianggap **andal hanya jika akurasi hold-out Indonesia ≥ 85%**.

## Persiapan di repo lokal
Pastikan ketiga berkas berikut ada di folder `data/` (hasil TAHAP 1):
- `data/train_id_oos.csv` — data latih (7.940 baris, sumber Rifky)
- `data/test_id_oos.csv`  — data uji out-of-source (547 baris, sumber Mendeley/ICTS-2017)
- `data/holdout_id_v2.csv` — hold-out independen (103 baris, label manual)

Jika belum ada, jalankan `scripts/prepare_dataset_id.py` terlebih dahulu.

## Langkah di Google Colab
1. Buka `notebooks/finetune_indobert_colab.ipynb` di Google Colab.
2. Ubah runtime ke GPU: **Runtime → Change runtime type → Tipe accelerator: GPU**.
3. Unggah ketiga berkas CSV di atas ke Colab (tab folder di sebelah kiri → ikon unggah).
4. Jalankan semua sel secara berurutan (**Runtime → Run all**).
   - Sel instalasi: memasang transformers, datasets, evaluate, scikit-learn, torch.
   - Sel data: memuat ketiga CSV, memastikan pembagian by-source tidak diulang.
   - Sel fine-tuning: IndoBERT 3 epoch (berjalan di GPU, biasanya beberapa menit).
   - Sel evaluasi: melaporkan akurasi, ROC-AUC, F1 per kelas, FPR, Brier, CI Wilson 95%.
   - Sel simpan: menuliskan model + `metrics.json` ke `/content/indobert_model`.
5. Unduh folder `indobert_model` dari Colab (kanan klik folder → Download).

## Menaruh model ke repo
1. Ekstrak hasil unduhan.
2. Letakkan isi folder ke `models/indobert/` di repo:
   ```
   models/indobert/
     config.json
     model.safetensors (atau pytorch_model.bin)
     tokenizer.json / vocab.txt / tokenizer_config.json
     special_tokens_map.json
     metrics.json
   ```
3. `app.py` (loader opsional TAHAP C) akan memakai model ini bila folder `models/indobert/` ada dan dependensi terpasang; jika tidak, fallback ke model lama (`models/id/`).

## MODEL_RELIABLE
`MODEL_RELIABLE` di `app.py` **tetap ditentukan oleh metrik di model card** (hold-out akurasi ≥ 85%), tidak otomatis hanya karena model tersedia. Buka `models/model_card_id.md` / `metrics.json` untuk menentukan status keandalan sebelum memakai model baru di produksi.

## Catatan lisensi
- `indobenchmark/indobert-base-p1`: model terbuka dari IndoNLU/indobenchmark. Periksa lisensi model di Hugging Face sebelum distribusi ulang.
- Data: Rifky/turnbackhoax.id (tanpa kartu lisensi resmi — gunakan hanya untuk penelitian, jangan mass-copy) dan Mendeley/ICTS-2017 (CC BY 4.0). Jangan menyimpan teks penuh artikel berhak cipta di repo.
