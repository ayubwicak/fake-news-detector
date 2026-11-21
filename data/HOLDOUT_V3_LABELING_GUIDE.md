# Panduan Pelabelan Hold-out v3 (Saring Berita)

## Tujuan
Hold-out v3 adalah satu set independen (>= 100 contoh, seimbang antara valid dan hoaks) yang **TIDAK** digunakan untuk pelatihan, tuning threshold, atau pemilihan model. Digunakan hanya untuk laporan akurasi akhir yang jujur.

## Siapa yang memberi label
Label diberikan secara **manual oleh pengguna/pelaku proyek** (bukan otomatis dari dataset publik), agar hold-out benar-benar independen dari data latih.

## Sumber contoh
Contoh boleh ditulis ulang sendiri (parafrase) atau diambil dari berita publik, dengan syarat:
- Jika mengambil dari artikel berhak cipta: simpan hanya **judul + tautan sumber + ringkasan pendek**, jangan teks penuh artikel.
- Isi kolom `sumber_url` dengan tautan asli jika ada.
- Isi kolom `tanggal` (format YYYY-MM-DD) jika diketahui.
- Isi kolom `catatan` dengan keterangan tambahan (topik, alasannya diberi label tersebut, dll.).

## Kolom CSV (`data/holdout_id_v3_template.csv`)

| Kolom | Deskripsi | Contoh |
|-------|-----------|--------|
| `text` | Teks berita/pernyataan yang akan diprediksi (min. 10 karakter) | "JAKARTA - Kementerian Kesehatan..." |
| `label` | 0 = Valid/Real, 1 = Hoax/Fake | 0 atau 1 |
| `sumber_url` | Tautan sumber asli (kosong boleh) | https://turnbackhoax.id/... |
| `tanggal` | Terbit (YYYY-MM-DD, kosong boleh) | 2026-11-20 |
| `catatan` | Keterangan tambahan (topik, alasannya label, dll.) | Pemerintah / Kesehatan |

## Kriteria label

### Label 0 — Valid/Real
- Berita yang dipublikasikan oleh media terpercaya dan dapat diverifikasi.
- Pernyataan resmi lembaga pemerintah/instansi.
- Tidak mengandung klaim medis misterius, tekanan waktu, atau ajakan menyebar tanpa dasar.

### Label 1 — Hoax/Fake
- Klaim yang telah dibantah oleh turnbackhoax.id, cekfakta.com, MAFINDO, atau lembaga serupa.
- Klaim medis aneh (obat ajaib, mencegah semua penyakit, dll.) tanpa bukti ilmiah.
- Rantai pesan (chain message) dengan ancaman/tekanan waktu.
- Klaim yang tidak dapat diverifikasi ke sumber resmi.

## Persyaratan kuantitas
- Minimal 100 baris.
- Seimbang: sekitar 50 valid / 50 hoaks (toleransi ±10).
- Beragam topik (pemerintahan, kesehatan, lingkungan, perdagangan, pendidikan, dll.).
- Beragam panjang (judul pendek, paragraf, artikel panjang, pesan rantai).

## Aturan penting
1. **Jangan** salin teks penuh artikel berhak cipta secara massal.
2. Hold-out v3 **tidak boleh** digunakan untuk tuning threshold atau pemilihan model.
3. Simpan hasil pelabelan di `data/holdout_id_v3.csv` (dibuat dari template ini).
4. Laporan akhir harus menyebutkan siapa yang memberi label dan tanggal pelabelan.
