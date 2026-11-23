# Panduan Fine-tuning IndoBERT di Google Colab (TAHAP C)

Notebook: `notebooks/finetune_indobert_colab.ipynb`

## Tujuan
Fine-tuning `indobenchmark/indobert-base-p1` di GPU gratis Colab untuk deteksi
hoaks bahasa Indonesia, dengan evaluasi jujur (setiap angka disertai `n` dan
interval Wilson 95%). Hold-out **v3 tidak dipakai** di notebook — lihat
`data/HOLDOUT_V3_LABELING_GUIDE.md`.

## Yang FIX di notebook (Butir 6)
- **By-source:** data sudah dibagi per sumber oleh `scripts/prepare_dataset_id.py`.
  Validation split untuk early stopping juga diambil **per sumber** bila kolom
  `source` tersedia (jika tidak, stratified random + peringatan).
- **Seed:** `SEED=42` untuk Python, NumPy, dan PyTorch (termasuk DataLoader).
- **Early stopping:** maks `MAX_EPOCHS=6`, `PATIENCE=2`, checkpoint terbaik
  disimpan berdasarkan loss validasi. (Sebelumnya: 3 epoch tanpa early stopping.)
- **Revision di-pin:** `HF_REVISION=c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2`
  (commit `indobenchmark/indobert-base-p1`, dilihat 23 Nov 2026). Dipakai pada
  `AutoTokenizer.from_pretrained(..., revision=...)` dan `AutoModel...from_pretrained`.
- **Perbaikan bug:** import sklearn sebelumnya `SyntaxError` (koma tanpa
  tanda kurung); f-string bersarang di sel Wilson juga diperbaiki.
- **Artefak lengkap:** `metrics.json`, `model_card.md`, `checksums.txt`,
  `indobert.zip`, dan `SHA256SUMS.txt`.

## Persiapan di repo lokal
Pastikan berkas di `data/` (dari `scripts/prepare_dataset_id.py`, default
GroupKFold by-source):
- `data/train_id_oos.csv` — data latih
- `data/test_id_oos.csv` — data uji out-of-source
- `data/holdout_id_v2.csv` — hold-out independen

Jangan unggah `data/holdout_id_v3.csv` ke Colab.

## Langkah di Colab
1. Buka notebook, ganti runtime ke **GPU** (Runtime → Change runtime type).
2. Unggah tiga CSV di atas ke `/content`.
3. **Runtime → Run all.** Sel akhir mencetak SHA-256 `indobert.zip`.
4. Unduh **`indobert.zip`** DAN **`SHA256SUMS.txt`**.

## Menaruh model ke repo
```bash
# 1. verifikasi integritas arsip
sha256sum -c SHA256SUMS.txt        # harus: indobert.zip: OK

# 2. ekstrak (struktur zip: indobert/...)
unzip indobert.zip

# 3. taruh isi folder indobert/ ke models/indobert/
cp -r indobert/. models/indobert/
```
Isi `models/indobert/`: `config.json`, `model.safetensors` (atau
`pytorch_model.bin`), `tokenizer.*`, `special_tokens_map.json`, `metrics.json`,
`model_card.md`, dan **`checksums.txt`**.

`app.py` **hanya memuat IndoBERT bila `checksums.txt` ada dan semua checksum
cocok**. Bila ada berkas berubah/hilang, loader fallback ke model TF-IDF lama
(`models/id/`) — tidak ada model setengah jadi yang dipakai. Statusnya bisa
dicek di `GET /health` → `indobert.checksums_ok` / `indobert.checksum_reason`.

## MODEL_RELIABLE — jangan tertukar
`metrics.json` dan `model_card.md` dari notebook **hanya catatan eksperimen
hold-out v2**. Flag `MODEL_RELIABLE` **tidak** ditentukan oleh keduanya, melainkan
dihitung `reliability.py` dari `logs/holdout_v3_result.json` (Butir 4) dengan 7
gerbang. Baca `models/model_card_id.md` bagian "Gerbang keandalan otomatis".
Nilai saat ini: **false** (v3 belum dievaluasi).

## Catatan lisensi
- `indobenchmark/indobert-base-p1`: IndoNLU/indobenchmark, lisensi MIT — periksa
  kembali kartu model di Hugging Face sebelum distribusi ulang.
- Data: Rifky/turnbackhoax.id (tanpa kartu lisensi resmi — hanya penelitian,
  jangan mass-copy) dan Mendeley/ICTS-2017 (CC BY 4.0). Jangan menyimpan teks
  penuh artikel berhak cipta di repo.
