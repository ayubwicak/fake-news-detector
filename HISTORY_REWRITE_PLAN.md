# Rencana Penulisan Ulang Riwayat Git (HISTORY_REWRITE_PLAN)

**STATUS: RENCANA SAJA — TIDAK DIJALANKAN.**

Dokumen ini mencatat langkah-langkah untuk menghapus file besar dan file hak
cipta dari riwayat Git (melalui `git filter-repo` atau `git filter-branch`),
agar repositori yang dipublikasikan tidak menyimpan artefak berukuran besar atau
materi berhak cipta.

⚠️ Peringatan: penulisan ulang riwayat **mengubah hash commit** dan memaksa
semua kontributor melakukan re-clone / rebase. **Jangan force-push** tanpa
persetujuan semua tim.

## Alasan

Beberapa file besar dan file yang mungkin melibatkan hak cipta telah terlacak
(di-commit) di riwayat origin dan perlu dibuang dari riwayat sebelum repositori
dipublikasikan:

| File / pola | Alasan masuk ke rencana |
|-------------|-------------------------|
| `models/id/*.pkl` | Artefak model Indonesia (pickle) berukuran besar; tidak boleh di-repo, diverifikasi via `models/CHECKSUMS.txt` saat runtime. Sudah `git rm --cached` di commit lokal, tapi masih ada di riwayat origin. |
| `models/*.pkl`, `models/*.npy` | Artefak model/ml lainnya (ukuran besar). |
| `data/raw/` | Mentah hasil scraping/kumpul data (bisa besar dan berhak cipta media komersial). |
| `data/raw_train.csv`, `data/raw_test.csv`, `data/raw_eval.csv` | Mentah, berukuran besar (19–60 MB). |
| `data/train_id_oos.csv`, `data/test_id_oos.csv` | File out-of-source berukuran besar; `test_id_oos.csv` masih terlacak. |
| `data/train_id.csv` | Pelatihan Indonesia, 3.9 MB (terlacak, bisa dikeluarkan dari riwayat). |
| `data/feedback.csv` | Masukan pengguna (122 baris asli, format lama) — berisi teks asli yang mungkin berhak cipta dan data sensitif; tidak boleh di-repo (sudah di-gitignore). |
| `data/feedback.csv.bak` | Cadangan migrasi feedback — tidak boleh di-repo. |
| Berkas dengan hak cipta media komersial | Artikel penuh dari Kompas, Detik, CNN, dll. (hak cipta pemilik masing-masing); dataset hanya menyimpan tautan + fitur turunan. |

## Opsi: repositori private

Karena beberapa file masih mungkin berisi data sensitif / teks berhak cipta,
pertimbangkan **repositori GitHub private** sebagai langkah awal, lalu baru
publikasikan setelah riwayat bersih dan seluruh file besar/hak cipta telah
dihapus dari riwayat.

## Langkah rencana (tidak dijalankan)

1. Cadangkan branch saat ini (`backup-local-<hash>`) sebagai safety net.
2. Identifikasi semua commit yang menambahkan file dalam daftar di atas:
   `git log --all --oneline -- models/id/ data/raw/ data/feedback.csv ...`
3. Jalankan `git filter-repo --invert-paths --path models/id/ --path data/raw/ ...`
   (atau `git filter-branch --index-filter`) untuk menghapusnya dari riwayat.
4. Perbarui `.gitignore` sehingga pola tersebut tidak akan terlacak kembali.
5. Lakukan `git gc --aggressive --prune=now` untuk membersihkan objek lama.
6. Push paksa ke repositori baru (atau private) **setelah persetujuan tim**;
   jangan force-push ke origin tanpa koordinasi.
7. Memberi tahu semua kontributor untuk re-clone (tidak bisa fetch/push biasa
   karena riwayat telah ditulis ulang).

## Catatan keputusan saat ini

- `git rm --cached` file besar sudah dilakukan di commit lokal (`c6c58f0`),
  tetapi file masih ada di riwayat origin; filter-repo diperlukan untuk
  membersihkan riwayat secara utuh.
- `data/test_id_oos.csv` masih terlacak di branch lokal; masuk ke daftar
  filter-repo jika ingin dibuang dari riwayat.
- `data/feedback.csv` 122 baris format lama sudah di-gitignore dan tidak
  pernah ter-commit / ter-push (hasil verifikasi `git log --all`); tidak ada
  riwayat yang perlu dibersihkan untuk berkas ini.
- Opsi repositori private dipertimbangkan sebelum publikasi.

## Tanggal

Dibuat: 23 November 2026 (branch `fix/qa-round-2`).
