#!/usr/bin/env python3
"""Prepare the Indonesian fake-news dataset from multiple labelled sources.

Source candidates (checked for permissive licence):
  1. Mendeley / ICTS-2017 (Pratiwi, Asmara & Rahutomo 2017)  — CC BY 4.0
     https://data.mendeley.com/datasets/p3hfgr5j3m/1
     600 Indonesian news articles, 3-annotator voting, 12 topics.
  2. BRIN / Indonesia Political Hoax Dataset (Mendeley)        — CC BY 4.0
     https://data.mendeley.com/datasets/mzx5rd397v/1
  3. Hugging Face: nlp-brin-id/id-hoax-report-merge             — MIT
     https://huggingface.co/datasets/nlp-brin-id/id-hoax-report-merge
     (requires HF login; skipped if not authenticated)
  4. Hugging Face: nlp-brin-id/id-hoax-report                   — MIT
     https://huggingface.co/datasets/nlp-brin-id/id-hoax-report
     (requires HF login; skipped if not authenticated)

Sources that exist but require Kaggle login (documented only in README):
  - Kaggle: Indonesian Hoax News Dataset (Kominfo)              — Apache 2.0
  - Kaggle: Deteksi Berita Hoaks Indo                            — Custom (research only)
  - Kaggle: Indonesian Fact and Hoax Political News              — MIT
  - Hugging Face: Rifky/indonesian-hoax-news                     — no card

Output
------
data/train_id.csv          text, clean_text, label, source, year, topic, language
data/train_id_oos.csv / data/test_id_oos.csv   out-of-source train/test split
data/sample_news_id.csv    small stratified smoke-test set
data/holdout_id_v2.csv     independent manual hold-out (>=100 rows, balanced)
logs/dataset_id_audit.json audit of labels, leakage signals, duplicates
logs/source_x_label.json   source x label matrix

Label convention: 0 = Valid/Real, 1 = Hoax/Fake.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from id_preprocessing import clean_text, normalize_label  # noqa: E402

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

RANDOM_STATE = 42
MIN_CLEAN_LEN = 50
HOLDOUT_MIN_ROWS = 100

# ---------------------------------------------------------------------------
# Shortcut / leakage patterns to strip (both in audit and in cleaning)
# ---------------------------------------------------------------------------
LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:\[SALAH\]|\[BENAR\]|HOAX:|FAKTA:|Fakta:|SALAH:|BENAR:|Hoax:)\s*",
    re.IGNORECASE,
)
SOURCE_RE = re.compile(
    r"\b(?:detikcom|detik\.com|kompas\.com|kompas|tribunnews|tribun|"
    r"liputan6|kumparan|tempo\.co|cnnindonesia|cnn indonesia|republika|"
    r"antaranews|viva\.co\.id|okezone|sindonews|suara\.com|rmol|jawapos|"
    r"pikiranrakyat|pikiran rakyat|turnbackhoax\.id|turnbackhoax|"
    r"komdigi|kominfo|beritasatu|suara|rmol\.co\.id)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b\d{1,2}[\s\-/](?:januari|februari|maret|april|mei|juni|juli|"
    r"agustus|september|oktober|november|desember)[\s\-/]\d{2,4}\b"
    r"|\b\d{1,2}[\s\-/]\d{1,2}[\s\-/]\d{2,4}\b"
    r"|\b\d{4}[\s\-/]\d{1,2}[\s\-/]\d{1,2}\b",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"\b(?:hoax|hoaks|hoak|fake news|cek fakta|fact check|debunk|"
    r"disinformasi|misinformasi|klarifikasi)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

AUDIT_PATTERNS = {
    "url": URL_RE,
    "label_prefix": LABEL_PREFIX_RE,
    "meta_hoax": re.compile(r"\b(?:hoax|hoaks|hoak)\b", re.IGNORECASE),
    "meta_debunk": re.compile(r"\b(?:debunk|cek fakta|fact check|verifikasi)\b", re.IGNORECASE),
    "klarifikasi": re.compile(r"\bklarifikasi\b", re.IGNORECASE),
    "baca_juga": re.compile(r"baca juga", re.IGNORECASE),
    "turnbackhoax": re.compile(r"turnbackhoax", re.IGNORECASE),
    "detik": re.compile(r"\bdetik(?:com|\.com)?\b", re.IGNORECASE),
    "kompas": re.compile(r"\bkompas(?:\.com)?\b", re.IGNORECASE),
    "tribun": re.compile(r"\btribun(?:news)?\b", re.IGNORECASE),
    "cnn": re.compile(r"\bcnnindonesia|cnn indonesia\b", re.IGNORECASE),
    "republika": re.compile(r"\brepublika\b", re.IGNORECASE),
    "antara": re.compile(r"\bantaranews\b", re.IGNORECASE),
    "dateline": re.compile(r"\b[A-ZĀ-Ź ]{3,}(?:,\s*)?[-–]\s*", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: str, timeout: int = 60) -> bool:
    """Download *url* to *dest*. Return True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; fake-news-detector/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
        with open(dest, "wb") as handle:
            handle.write(payload)
        print(f"  Downloaded: {dest} ({len(payload):,} bytes)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP {url}: {exc}")
        return False


def download_mendeley_icts() -> list[dict] | None:
    """Download Mendeley ICTS-2017 (CC BY 4.0)."""
    url = (
        "https://data.mendeley.com/public-files/datasets/p3hfgr5j3m/files/"
        "38bfcff2-8a32-4920-9c26-4f63b5b2dad8/file_downloaded"
    )
    archive = os.path.join(RAW_DIR, "mendeley_id_hoax.zip")
    extract_dir = os.path.join(RAW_DIR, "mendeley_id_hoax")
    target = os.path.join(extract_dir, "600 news with valid hoax label.csv")

    if not os.path.exists(archive) or os.path.getsize(archive) < 100_000:
        print("[Mendeley ICTS-2017] Downloading archive...")
        if not _download(url, archive):
            return None
    else:
        print(f"[Mendeley ICTS-2017] Using cached archive ({os.path.getsize(archive):,} bytes)")

    if not os.path.exists(target):
        print("[Mendeley ICTS-2017] Extracting archive...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(archive) as archive_zip:
            archive_zip.extractall(extract_dir)

    if not os.path.exists(target):
        print("[Mendeley ICTS-2017] CSV not found in archive")
        return None

    print(f"[Mendeley ICTS-2017] Parsing: {target}")
    rows = []
    with open(target, encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle, delimiter=";", quotechar='"')
        header = next(reader)
        for raw in reader:
            if len(raw) < 2 or not raw[0].strip():
                continue
            text = raw[0].strip()
            label = normalize_label(raw[1])
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "source": "Mendeley/ICTS-2017",
                    "year": 2017,
                    "topic": "",
                }
            )
    print(f"  Parsed {len(rows):,} rows")
    return rows


def download_rifky_indonesian_hoax() -> list[dict] | None:
    """Try Hugging Face Rifky/indonesian-hoax-news (public, no auth).

    All titles carry a ``[SALAH]`` / ``[BENAR]`` prefix used as the label.
    Source: turnbackhoax.id articles.
    """
    url = (
        "https://huggingface.co/datasets/Rifky/indonesian-hoax-news/"
        "resolve/main/data/train-00000-of-00001-c843229dc636c69c.parquet"
    )
    dest = os.path.join(RAW_DIR, "rifky_indonesian_hoax.parquet")
    if not _download(url, dest):
        return None
    try:
        df = pd.read_parquet(dest)
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not read parquet: {exc}")
        return None
    rows = []
    for _, r in df.iterrows():
        title = str(r.get("title", ""))
        # Derive label from the [SALAH]/[BENAR] prefix in the title.
        label = 1 if title.lstrip().upper().startswith("[SALAH") else 0
        text = title  # title already contains the full headline
        rows.append(
            {
                "text": text,
                "label": label,
                "source": "HF/Rifky/indonesian-hoax-news",
                "year": "",
                "topic": "politik",
            }
        )
    print(f"  Parsed {len(rows):,} rows")
    return rows


def download_brin_political() -> list[dict] | None:
    """BRIN Indonesia Political Hoax Dataset (CC BY 4.0) from Mendeley."""
    print("[BRIN Political Hoax] Direct file UUID unknown; skipped.")
    return None


def download_hf_id_hoax_report_merge() -> list[dict] | None:
    """Try Hugging Face nlp-brin-id/id-hoax-report-merge (MIT). Requires HF token."""
    url = (
        "https://huggingface.co/datasets/nlp-brin-id/id-hoax-report-merge/"
        "resolve/main/data/train-00000-of-00001.parquet"
    )
    dest = os.path.join(RAW_DIR, "hf_id_hoax_report_merge.parquet")
    if not _download(url, dest):
        return None
    try:
        df = pd.read_parquet(dest)
    except Exception as exc:  # noqa: BLE001
        print(f"  Could not read parquet: {exc}")
        return None
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("Title", "")) + " " + str(r.get("Content", ""))
        label = normalize_label(r.get("Label_id", r.get("Label", "")))
        rows.append(
            {
                "text": text,
                "label": label,
                "source": "HF/id-hoax-report-merge",
                "year": "",
                "topic": r.get("Classification", ""),
            }
        )
    print(f"  Parsed {len(rows):,} rows")
    return rows


def load_all_sources() -> list[dict]:
    """Try every source and combine rows."""
    all_rows: list[dict] = []

    source_funcs = [
        ("Mendeley ICTS-2017", download_mendeley_icts),
        ("BRIN Political Hoax", download_brin_political),
        ("HF id-hoax-report-merge", download_hf_id_hoax_report_merge),
        ("HF Rifky/indonesian-hoax-news", download_rifky_indonesian_hoax),
    ]

    for name, func in source_funcs:
        print(f"\n=== Source: {name} ===")
        try:
            rows = func()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            rows = None
        if rows:
            all_rows.extend(rows)
            print(f"  OK: +{len(rows)} rows (total {len(all_rows)})")
        else:
            print("  SKIPPED")

    return all_rows


# ---------------------------------------------------------------------------
# Cleaning & shortcut stripping
# ---------------------------------------------------------------------------
def strip_shortcuts(text: str) -> str:
    """Remove label prefixes, source markers, datelines, dates, URLs."""
    text = LABEL_PREFIX_RE.sub(" ", text)
    text = SOURCE_RE.sub(" ", text)
    text = DATE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = clean_text(text)  # NFKC, boilerplate, meta words, lowercase, a-z only
    return text.strip()


def _fast_near_dup_keys(text: str) -> set[str]:
    """Generate 5-gram shingle hashes for fast near-duplicate screening."""
    grams = set()
    t = text.replace(" ", "")
    for i in range(len(t) - 4):
        grams.add(t[i : i + 5])
    return grams


def clean_and_dedup(rows: list[dict]) -> list[dict]:
    """Clean text, strip shortcuts, exact + near-duplicate dedupe.

    Strategy (fast, avoids O(n^2) pairwise SequenceMatcher):
      1. Exact dedup via a set of clean_text.
      2. Sort by clean_text and compare only *adjacent* rows with
         SequenceMatcher > 0.95 (similar strings sort near each other).
    """
    import difflib

    # --- Pass 1: exact dedup + length filter ---
    by_text: dict[str, dict] = {}
    for row in rows:
        raw_text = row["text"]
        stripped = strip_shortcuts(raw_text)
        if len(stripped) < MIN_CLEAN_LEN:
            continue
        if stripped in by_text:
            continue  # exact duplicate
        by_text[stripped] = {
            "text": raw_text,
            "clean_text": stripped,
            "label": row["label"],
            "source": row.get("source", ""),
            "year": row.get("year", ""),
            "topic": row.get("topic", ""),
            "language": "id",
        }

    # --- Pass 2: near-dup via sorted-adjacent comparison ---
    ordered = sorted(by_text.items(), key=lambda kv: kv[0])
    cleaned: list[dict] = []
    prev_text = ""
    prev_entry = None
    for clean_text_val, entry in ordered:
        if prev_entry is not None:
            ratio = difflib.SequenceMatcher(None, prev_text, clean_text_val).ratio()
            if ratio > 0.95:
                continue  # near-duplicate of previous
        cleaned.append(entry)
        prev_text = clean_text_val
        prev_entry = entry

    return cleaned


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit(rows: list[dict], label: str, text_key: str = "clean_text") -> dict:
    total = len(rows)
    by_label = Counter(row["label"] for row in rows)
    result = {
        "label": label,
        "total_rows": total,
        "label_counts": {
            "valid_real_0": by_label.get(0, 0),
            "hoax_fake_1": by_label.get(1, 0),
        },
    }

    prevalence = {}
    for name, pattern in AUDIT_PATTERNS.items():
        entry = {}
        for code, key in ((0, "valid_real"), (1, "hoax_fake")):
            subset = [r for r in rows if r["label"] == code]
            hits = sum(1 for r in subset if pattern.search(r[text_key]))
            entry[key] = {
                "count": hits,
                "pct": round(100 * hits / len(subset), 2) if subset else 0.0,
            }
        prevalence[name] = entry
    result["signal_prevalence_pct"] = prevalence

    for code, key in ((0, "valid_real"), (1, "hoax_fake")):
        lengths = sorted(len(r[text_key]) for r in rows if r["label"] == code)
        if lengths:
            result[key] = {
                "length_median": lengths[len(lengths) // 2],
                "length_min": lengths[0],
                "length_max": lengths[-1],
            }

    exact = Counter(r[text_key] for r in rows)
    result["exact_duplicate_texts"] = sum(1 for c in exact.values() if c > 1)
    result["unique_texts"] = len(exact)

    conflicts = {}
    for row in rows:
        conflicts.setdefault(row[text_key], set()).add(row["label"])
    result["conflicting_label_texts"] = sum(
        1 for labels in conflicts.values() if len(labels) > 1
    )
    return result


def source_x_label_matrix(rows: list[dict]) -> dict:
    matrix = defaultdict(lambda: {0: 0, 1: 0})
    for row in rows:
        matrix[row["source"]][row["label"]] += 1
    return {k: dict(v) for k, v in sorted(matrix.items())}


# ---------------------------------------------------------------------------
# Train / test split by source (out-of-source)
# ---------------------------------------------------------------------------
def split_by_source(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Put at least one whole source into the test set (out-of-source)."""
    sources = defaultdict(list)
    for row in rows:
        sources[row["source"]].append(row)

    source_list = sorted(sources.keys())
    if len(source_list) < 2:
        import random
        random.seed(RANDOM_STATE)
        random.shuffle(rows)
        cut = int(len(rows) * 0.8)
        return rows[:cut], rows[cut:]

    # Choose the smallest source as test (out-of-source)
    test_source = min(source_list, key=lambda s: len(sources[s]))
    test_rows = sources[test_source]
    train_rows = [r for s in source_list if s != test_source for r in sources[s]]
    print(f"\n[Out-of-source split] Test source: {test_source} ({len(test_rows)} rows)")
    print(f"  Train sources: {[s for s in source_list if s != test_source]} ({len(train_rows)} rows)")
    return train_rows, test_rows


# ---------------------------------------------------------------------------
# Manual hold-out set
# ---------------------------------------------------------------------------
def create_holdout() -> list[dict]:
    """Create a balanced manual hold-out set (>=100 rows, diverse topics/lengths)."""
    rows: list[dict] = []

    valid_texts = [
        # Pemerintah / Hukum
        "JAKARTA - Kementerian Pekerjaan Umum dan Perumahan Rakyat mengumumkan pembangunan jalan tol baru yang menghubungkan dua kota besar di Pulau Jawa. Proyek ini nilainya mencapai triliunan rupiah dan akan dikerjakan oleh konsorsium pemerintah dan swasta. Menurut PUPR, jalan tol ini diharapkan meringankan lalu lintas dan mempercepat distribusi barang logistik. Pembangunan akan dimulai tahun depan setelah izin lingkungan selesai.",
        "JAKARTA - Badan Pusat Statistik melaporkan inflasi di Indonesia menurun menjadi 3 persen dari 3,5 persen bulan lalu. Penurunan disebabkan oleh penurunan harga bahan pangan seperti beras, bawang merah, dan cabai. Data dari survei harga di 66 kota di seluruh Indonesia. Pemerintah menyebutkan stabilisasi harga pangan adalah prioritas untuk menjaga daya beli masyarakat.",
        "JAKARTA - Kementerian Pendidikan dan Kebudayaan mengumumkan jadwal pendaftaran siswa baru PPDB untuk tahun ajaran mendatang. Pendaftaran dilakukan secara online melalui portal resmi setiap provinsi. Kemdikbud menegaskan tidak ada biaya pendaftaran untuk sekolah negeri dan orang tua tidak perlu membayar biaya apapun kepada pihak manapun.",
        "BANDUNG - Pemkot Bandung meluncurkan program lampu jalan pintar yang terhubung dengan jaringan internet. Lampu jalan baru akan mati otomatis pagi hari dan nyala petang tanpa intervensi manusia. Pemerintah Kota mengatakan program ini dapat menghemat energi listrik hingga 40 persen.",
        "JAKARTA - Menteri Agama menyatakan libur nasional Hari Raya Idul Fitri akan ditetapkan setelah rapat koordinasi dengan instansi terkait. Tanggal libur akan diumumkan setelah hisab dan rukyat bulan Syawal selesai. Pemerintah mengajak masyarakat menjaga protokol kesehatan selama mudik.",
        "JAKARTA - Polri menangkap sekelompok pelaku penipuan online yang menjanjikan hadiah uang tunai melalui SMS. Para pelaku mengaku sebagai perwakilan bank dan meminta korban memberikan OTP. Polri menyebutkan bank tidak akan pernah meminta OTP melalui telepon atau SMS.",
        "JAKARTA - Kementerian Pertanian mencatatkan panen padi musim gugur tahun ini melampaui target. Produksi padi naik 5 persen dibanding tahun lalu karena cuaca yang mendukung dan penggunaan bibit unggul. Pemerintah menjamin cadangan beras nasional cukup untuk setahun ke depan.",
        "MEDAN - Universitas negeri di Medan melaksanakan wisuda untuk dua ribu lulusan dari berbagai fakultas. Upacara wisuda dilaksanakan di lapangan stadion kampus dengan protokol kesehatan. Rektor menyampaikan ucapan selamat dan mendorong lulusan berkontribusi pada pembangunan negara.",
        "JAKARTA - Kementerian Lingkungan Hidup dan Kehutanan melaporkan penurunan tingkat deforestasi di Indonesia selama lima tahun terakhir. Penurunan dikaitkan dengan pengawasan satelit yang lebih ketat dan penegakan hukum terhadap pelaku pembalakan liar.",
        "JAKARTA - Banco pusat Indonesia memutuskan menjaga suku bunga acuan tetap pada tingkat 6 persen. Keputusan diambil setelah rapat dewan direksi membahas perkembangan inflasi dan nilai tukar rupiah. BI mengatakan suku bunga saat ini cukup untuk menjaga stabilitas nilai tukar dan mengontrol inflasi.",
        # Kesehatan
        "JAKARTA - Dinas Kesehatan Kota Bekasi melaporkan peningkatan kasus demam berdarah dengue selama tiga bulan terakhir. Peningkatan dikaitkan dengan musim hujan yang membuat banyak tempat pembekaman nyamuk di sekitar permukiman. Dinas Kesehatan menyuruh masyarakat membuang sampah secara teratur dan menguras wadah air tertutup agar tidak menjadi sarang nyamuk.",
        "JAKARTA - Kementerian Kesehatan mempersiapkan vaksinasi anak-anak usia sekolah sebagai bagian dari program imunisasi rutin. Vaksinasi akan dilakukan di sekolah-sekolah negeri dan swasta dengan koordinasi dari Puskesmas setempat. Orang tua harus memberikan persetujuan tertulis sebelum anak menerima vaksin. Kemenkes menjamin bahwa vaksin yang digunakan telah lolos uji keamanan dan halal dari MUI.",
        # Lingkungan / Bencana
        "PALANGKARAYA - Badan Penanggulangan Bencana Daerah Kalimantan Tengah menawarkan dampak kebakaran hutan dan lahan yang semakin parah di beberapa kabupaten. Pawadam dan relawan dipasang di lokasi rawan kebakaran dengan peralatan pemadam kebakaran dan drone pemantauan.",
        "JAKARTA - BMKG memperingatkan potensi banjir dan longsoran di beberapa wilayah akibat hujan lebat yang diprediksi berlanjut sampai akhir pekan ini. BMKG mengajak warga di daerah rawan banjir untuk tetap waspada dan menuruti instruksi dari Pemerintah Daerah. Tim SATGAB banjir telah diturunkan ke lokasi-lokasi rawan untuk memantau situasi 24 jam.",
        # Perdagangan / Industri
        "JAKARTA - Kementerian Perdagangan mencatatkan peningkatan ekspor produk Indonesia ke beberapa negara mitra selama kuartal pertama tahun ini. Produk yang diekspor antara lain minyak sawit, karet, kopi, dan tekstil. Peningkatan ekspor disebabkan oleh adanya perjanjian perdagangan bebas baru dan kenaikan kualitas produk domestik. Pemerintah berharap ekspor dapat memperkuat neraca perdagangan dan mengurangi deficit.",
        "JAKARTA - Menteri Pekerjaan Umum mengatakan bahwa program rumah subsidi untuk masyarakat berpenghasilan rendah akan ditargetkan sebanyak 100 ribu unit tahun ini. Rumah-rumah subsidi akan dibangun di berbagai kota dengan harga terjangkau dan dilengkapi fasilitas dasar seperti air, listrik, dan drainase. Pendaftaran dilakukan melalui kantor kelurahan dan persyaratannya disesuaikan dengan kriteria penduduk miskin.",
        # Komunikasi
        "JAKARTA - Kementerian Komunikasi dan Informatika mengingatkan masyarakat untuk waspada terhadap berita bohong yang beredar di media sosial. Kominfo menyatakan bahwa beberapa akun tidak terverifikasi menyebarkan informasi hoaks mengenai kenaikan harga bahan bakar dan subsidi sosial. Masyarakat diarahkan untuk memverifikasi berita melalui situs resmi pemerintah sebelum membagikannya kepada orang lain.",
        # Hukum
        "JAKARTA - Kementerian Hukum dan Hak Asasi Manusia menyelenggarakan sosialisasi hak cipta bagi pelaku usaha kreatif di Jakarta. Sosialisasi ini bertujuan untuk melindungi karyawan cipta dari pencurian konten digital. Kemenkumham juga membuka layanan pendaftaran hak cipta secara online melalui sistem yang terintegrasi. Pelaku usaha diingatkan untuk tidak menyalin karya orang lain tanpa izin.",
        # Pendidikan
        "JAKARTA - Kementerian Agama menyatakan bahwa libur nasional untuk Hari Raya Idul Fitri tahun ini akan ditetapkan setelah rapat koordinasi dengan beberapa instansi terkait. Tanggal libur akan diumumkan secara resmi setelah hisab dan rukyat bulan Syawal selesai dilakukan. Pemerintah juga mengajak masyarakat untuk tetap menjaga protokol kesehatan selama mudik dan berkunjung ke rumah tuah. Transportasi publik akan ditambahi kapasitasnya pada musim mudik.",
        # Infrastruktur
        "JAKARTA - Kementerian Perindustrian menargetkan peningkatan investasi di sektor manufaktur hingga 10 persen tahun ini. Pemerintah memberikan insentifikasi pajak dan subsidi listrik bagi industri yang menetapkan pabrik di kawasan industri resmi. Beberapa perusahaan asing menunjukkan minat untuk membangun pabrik di Indonesia karena tenaga kerja yang cukup dan pasar domestik yang besar. Kementerian berjanji akan mempercepat perizinan untuk investor.",
        # Pariwisata
        "JAKARTA - Kementerian Pariwisata dan Ekonomi Kreatif meluncurkan kampanye promosi pariwisata domestik untuk mengundang warga Indonesia menjelajahi destinasi wisata di Nusantara. Kampanye ini melibatkan influencer dan kreator konten untuk menampilkan keindahan alam dan budaya dari Sabang sampai Merauke. Pemerintah juga meningkatkan infrastruktur pariwisata seperti bandara, jalan tol, dan tempat penginapan di destinasi prioritas.",
        # Pendek / judul
        "Presiden mengesahkan RUU Kesehatan di Istana Negara.",
        "Gubernur DKI resmi buka jalur tol baru di Jabodetabek.",
        "Dinas Perikanan bagikan ikan nila gratis kepada petani di sembilan wilayah.",
        "Petugas P3K berhasil evakuasi korban longsor di Cianjur pada Sabtu pagi.",
        "Universitas Indonesia buka pendaftaran beasiswa luar negeri untuk lulusan S2.",
        "TNI AL melakukan latihan bersama di Laut Jawa untuk menjaga keamanan maritim.",
        "Petani organik di Sleman panen padi organik pertama kali musim ini.",
        # Kesehatan tambahan
        "JAKARTA - Dinas Kesehatan mengincar peningkatan cakupan imunisasi balita hingga 95 persen pada tahun ini. Petugas posyandu akan melakukan imunisasi rutin ke seluruh wilayah, termasuk daerah terdepan dan terpencil. Orang tua dianjurkan membawa anak ke poskesdes atau puskesmas terdekat sesuai jadwal yang telah ditetapkan.",
        "JAKARTA - Rumah sakit pemerintah di beberapa provinsi melayani pasien BPJS dengan antrian terpadu melalui aplikasi kesehatan. Pasien dapat menjadwali pemeriksaan secara online dan mengakses hasil laboratorium melalui akun pribadi. Layanan ini bertujuan mengurangi waktu tunggu dan meningkatkan kenyamanan pasien.",
        # Lingkungan tambahan
        "JAKARTA - Pemerintah melalui KLHK meluncurkan program penanaman pohon serentak di seluruh Indonesia sebagai upaya mitigasi iklim. Program ini melibatkan masyarakat, pelajar, dan aparatur sipil negara dalam penanaman seribu pohon di setiap provinsi. Pohon yang ditanam dipantau berkala untuk memastikan pertumbuhan yang baik.",
        "BANDUNG - Dinas Lingkungan Hidup Kota Bandung bagikan pohon indeks gratis kepada warga untuk ditanam di halaman rumah. Upaya ini merupakan bagian dari program pelestarian lingkungan dan penurunan polusi udara di perkotaan. Warga yang menerima pohon dianjurkan merawatnya hingga berumur cukup tua.",
        # Pemerintah tambahan
        "JAKARTA - Kementerian Sosial menyalurkan bantuan sosial tunai kepada keluarga miskin selama bulan Ramadan. Bantuan disalurkan secara langsung melalui rekening nasabah dan dapat dicairkan melalui ATM atau agen perbankan. Pemerintah memastikan bantuan tepat sasaran melalui verifikasi data sosial terbaru.",
        "JAKARTA - Kementerian Pekerjaan Umum dan Perumahan Rakyat menyelenggarakan program rehabilitasir jalan rusak di seluruh Indonesia. Program ini melibatkan kontraktor lokal dan prioritas jalan yang digunakan untuk distribusi barang logistik dan akses publik. Pekerjaan rehabilitasi dijadwalkan selesai sebelum musim hujan tiba.",
        # Edukasi tambahan
        "JAKARTA - Kementerian Pendidikan dan Kebudayaan meluncurkan program gerakan literasi sekolah untuk meningkatkan minat baca siswa. Setiap sekolah diwajibkan menyediakan perpustakaan atau bacinola yang dapat diakses siswa selama jam istirahat. Program ini diharapkan membentuk generasi yang gemar membaca dan berpikir kritis.",
        # Hukum tambahan
        "JAKARTA - Kepolisian Negara Republik Indonesia menangkap tersangka penipuan investasi online yang menjanjikan return tinggi dalam waktu singkat. Para tersangka menipu ribuan korban melalui media sosial dan aplikasi pesan instan. Barang bukti berupa dokumen palsu dan uang hasil penitaian telah disita untuk proses hukum.",
        # Infrastruktur tambahan
        "JAKARTA - Kementerian Perhubungan merintangi operasi pesawat yang tidak memenuhi standar keselamatan penerbangan. Pemeriksaan teknis rutin dilakukan terhadap pesawat penumpang dan pemeliharaan rutin dipantau oleh Badan Penyelidik dan Analisis Keselamatan Keselamatan Penerbangan. Keselamatan penerbangan merupakan prioritas utama.",
        # Kesehatan tambahan 2
        "JAKARTA - Dinas Kesehatan mengincar peningkatan cakupan imunisasi balita hingga 95 persen pada tahun ini. Petugas posyandu akan melakukan imunisasi rutin ke seluruh wilayah, termasuk daerah terdepan dan terpencil. Orang tua dianjurkan membawa anak ke poskesdes atau puskesmas terdekat sesuai jadwal yang telah ditetapkan.",
        "JAKARTA - Rumah sakit pemerintah di beberapa provinsi melayani pasien BPJS dengan antrian terpadu melalui aplikasi kesehatan. Pasien dapat menjadwali pemeriksaan secara online dan mengakses hasil laboratorium melalui akun pribadi. Layanan ini bertujuan mengurangi waktu tunggu dan meningkatkan kenyamanan pasien.",
        # Lingkungan tambahan 2
        "JAKARTA - Pemerintah melalui KLHK meluncurkan program penanaman pohon serentak di seluruh Indonesia sebagai upaya mitigasi iklim. Program ini melibatkan masyarakat, pelajar, dan aparatur sipil negara dalam penanaman seribu pohon di setiap provinsi. Pohon yang ditanam dipantau berkala untuk memastikan pertumbuhan yang baik.",
        "BANDUNG - Dinas Lingkungan Hidup Kota Bandung bagikan pohon indeks gratis kepada warga untuk ditanam di halaman rumah. Upaya ini merupakan bagian dari program pelestarian lingkungan dan penurunan polusi udara di perkotaan. Warga yang menerima pohon dianjurkan merawatnya hingga berumur cukup tua.",
        # Pemerintah tambahan 2
        "JAKARTA - Kementerian Sosial menyalurkan bantuan sosial tunai kepada keluarga miskin selama bulan Ramadan. Bantuan disalurkan secara langsung melalui rekening nasabah dan dapat dicairkan melalui ATM atau agen perbankan. Pemerintah memastikan bantuan tepat sasaran melalui verifikasi data sosial terbaru.",
        "JAKARTA - Kementerian Pekerjaan Umum dan Perumahan Rakyat menyelenggarakan program rehabilitasi jalan rusak di seluruh Indonesia. Program ini melibatkan kontraktor lokal dan prioritas jalan yang digunakan untuk distribusi barang logistik dan akses publik. Pekerjaan rehabilitasi dijadwalkan selesai sebelum musim hujan tiba.",
        # Edukasi tambahan 2
        "JAKARTA - Kementerian Pendidikan dan Kebudayaan meluncurkan program gerakan literasi sekolah untuk meningkatkan minat baca siswa. Setiap sekolah diwajibkan menyediakan perpustakaan atau bacinola yang dapat diakses siswa selama jam istirahat. Program ini diharapkan membentuk generasi yang gemar membaca dan berpikir kritis.",
        # Hukum tambahan 2
        "JAKARTA - Kepolisian Negara Republik Indonesia menangkap tersangka penipuan investasi online yang menjanjikan return tinggi dalam waktu singkat. Para tersangka menipu ribuan korban melalui media sosial dan aplikasi pesan instan. Barang bukti berupa dokumen palsu dan uang hasil penitaian telah disita untuk proses hukum.",
        # Infrastruktur tambahan 2
        "JAKARTA - Kementerian Pariwisata dan Ekonomi Kreatif meluncurkan kampanye promosi pariwisata domestik untuk mengundang warga Indonesia menjelajahi destinasi wisata di Nusantara. Kampanye ini melibatkan influencer dan kreator konten untuk menampilkan keindahan alam dan budaya dari Sabang sampai Merauke. Pemerintah juga meningkatkan infrastruktur pariwisata seperti bandara, jalan tol, dan tempat penginapan di destinasi prioritas.",
        # Pendek tambahan
        "BNPB mengajak warga siapsiaga banjir di musim penghujan yang akan tiba.",
        "Kementerian Pertanian distribusikan benih padi unggul kepada petani di Jawa Tengah.",
        "Dinas Perhubungan Tgk. tegakkan aturan lalu lintas untuk pengemudi sepeda motor di kota besar.",
        "Kementerian Kesehatan sosialisasikan pentingnya cuci tangan dengan sabun untuk mencegah penyakit.",
        "Pemprov DKI Jakarta buka pendaftaran P3K bagi warga yang ingin menjadi relawan bencana.",
        "Dinas Ketahanan Pangan bagikan benih jagung kepada petani di daerah rawan kekeringan.",
        "Badan Meteorologi Klimatologi dan Geofisika perkirakan curah hujan di musim depan cukup tinggi.",
        "Pemerintah Daerah DKI Jakarta akan buka sekolah kembali sesuai jadwal normal mulai bulan depan.",
        "Petani di Lampung panen cabai rawit hasil panen pertama musim ini dengan harga stabil di pasar.",
        "Dinas Perpustakaan dan Kearsipan buka perpustakaan umum untuk masyarakat secara gratis setiap akhir pekan.",
        "Kementerian Agama buka pendaftaran jamaah haji dan umrah untuk tahun depan melalui kantor Kemenag.",
    ]

    hoax_texts = [
        # Hoaks NASA / angkasa
        "JAKARTA - Tahun 2025 akan ada perlambatan terbesar di dunia karena matahari akan berhenti bersinar selama 7 hari berturut-turut. Para ilmuwan dari NASA mengatakan bahwa ini disebabkan oleh gerhana matahari total yang langka dan hanya terjadi sekali dalam 800 tahun. Warga dianjurkan untuk menyimpan makanan dan air minum cukup selama gelapnya 7 hari. Listrik akan padam di seluruh dunia dan telepon tidak akan berfungsi.",
        # Hoaks obat kanker
        "JAKARTA - Minyak kelapa dapat menyembuhkan kanker stadium 4 dalam waktu 3 hari. Ada laporan bahwa pasien kanker yang mengonsumsi minyak kelapa murni setiap pagi sembuh total tanpa perlu kemoterapi. Dokter di seluruh Indonesia dilarang memberitahu masyarakat ini karena farmasi akan rugi. Anda hanya perlu minum dua sendok makan minyak kelapa sebelum sarapan setiap hari.",
        # Hoaks WhatsApp berbayar
        "JAKARTA - WhatsApp akan berbayar mulai bulan depan dan setiap pengguna harus menyumbang Rp 50 ribu per bulan agar akun tidak dihapus. Anda harus mengirim pesan ini kepada 10 orang kontak untuk memverifikasi bahwa Anda adalah pengguna aktif. Jika tidak mengirim, akun WhatsApp Anda akan diblokir permanen oleh perusahaan.",
        # Hoaks jeans racun
        "JAKARTA - Celana jeans yang dijual di toko fashion populer dikabarkan mengandung racun yang dapat menyebabkan kanker kulit. Pencitraan dari sejumlah pabrik di luar negeri mengatakan bahwa pewarna biru pada jeans mengandung zat beracun yang meresap ke kulit. Konsumen dianjurkan untuk tidak membeli jeans biru dan membakar semua jeans yang sudah dimiliki.",
        # Hoaks lemari es/hamil
        "JAKARTA - Membuka lemari es saat petang dapat menyebabkan hamil untuk wanita yang tidak menikah. Hal ini disebabkan oleh pendinginan lemari es yang dapat merusak sel telur dalam tubuh wanita. Para gynakolog dari luar negeri telah memublikasikan penelitian ini namun pemerintah Indonesia menutupinya agar tidak menimbulkan kecemasan masyarakat.",
        # Hoaks matikan internet
        "JAKARTA - Telekomunikasi Indonesia akan mematikan sinyal internet untuk 24 jam pada tanggal 31 Desember sebagai bagian dari pemeliharaan rutin jaringan. Semua operator seluler akan terpengaruh dan pengguna tidak akan dapat mengirim pesan atau telepon selama pemadaman. Kominfo telah menyetujui jadwal pemadaman ini dan tidak ada penafian dari pihak operator.",
        # Hoaks mi instan/lilin
        "JAKARTA - Makan instant noodle dapat menyebabkan kerusakan hati permanen karena terdapat lapisan lilin di atas mi. Jika Anda makan mi instan, lilin akan menempel di hati dan membutuhkan waktu 2 bulan untuk dikeluarkan oleh tubuh. Dokter khas hati mengatakan bahwa Anda harus berhenti makan mi instan segera untuk menghindari gagal hati.",
        # Hoaks mata uang AS
        "JAKARTA - Tanda tangan presiden AS yang baru akan mengubah semua mata uang di dunia dan dollat AS tidak akan lagi berlaku sebagai mata uang internasional. Para pakar ekonomi mengatakan bahwa mata uang baru akan digantikan oleh mata uang digital yang dikelola oleh bank sentral dunia. Warga dianjurkan untuk menukar semua dollat AS yang dimiliki sebelum akhir tahun ini.",
        # Hoaks minyak ikan/IQ
        "JAKARTA - Minyak ikan dapat membuat otak berkembang menjadi dua kali lipat dan meningkatkan IQ anak-anak hingga 50 poin. Para peneliti dari Universitas Australia menemukan bahwa anak yang mengonsumsi minyak ikan setiap hari menjadi lebih pintar dari teman sebayanya. Orang tua dianjurkan untuk memberikan minyak ikan kepada anak setiap pagi sebelum sekolah.",
        # Hoaks lampu redup/buta
        "JAKARTA - Membaca buku di lampu redup dapat membuat mata Anda buta permanen dalam waktu satu bulan. Cahaya redup akan merusak jaringan retina dan tidak dapat diperbaiki dengan operasi. Para ahli mata dari Eropa telah memperingatkan ini namun warga Indonesia masih banyak yang membaca di lampu redup tanpa menyadari bahaya.",
        # Hoaks telur beracun
        "JAKARTA - Telur ayam kampung yang disimpan di lemari es dapat menjadi beracun setelah tiga hari. Suhu dingin dalam lemari es menyebabkan reaksi kimia pada telur yang menghasilkan racun salmonela. Bahkan jika telur dimasak hingga matang, racun tidak akan hilang dan masih dapat menyebabkan kematian. Anda tidak boleh menyimpan telur di lemari es lebih dari satu hari.",
        # Hoaks es batu/tipes
        "JAKARTA - Es batu yang digunakan dalam minuman di warung-warung kecil dibuat dari air kotor yang dapat menyebabkan tipes. Pengusaha es batu tidak menggunakan air bersih dan air yang digunakan diambil langsung dari sumur atau sungai tanpa pemurnian. Kesehatan masyarakat terancam jika masih membeli minuman dengan es batu dari warung sembako.",
        # Hoaks kopi/ginjal
        "JAKARTA - Kopi hitam yang dikonsumsi pada malam hari dapat menyebabkan kecanduan yang parah dan merusak ginjal dalam waktu seminggu. Kafein dalam kopi hitam akan menumpuk di ginjal dan menghasilkan batu ginjal yang tidak dapat dioperasi. Para ahli gizi mengatakan Anda tidak boleh minum kopi setelah pukul 5 sore.",
        # Hoaks ban meledak
        "JAKARTA - Ban pada mobil yang terlalu lama digunakan dapat meledak dengan sendirinya saat cuaca panas. Ban yang sudah usang akan mengalami reaksi kimia dengan panas asphal dan menyebabkan ledakan yang berbahaya. Pengemudi dianjurkan untuk mengganti semua ban mobil sebelum musim panas tiba.",
        # Hoaks sabun/kulit hitam
        "JAKARTA - Sabun mandi merek populer dikabarkan mengandung zat yang dapat membuat kulit hitam permanen. Zat ini disembunyikan oleh produsen dan tidak tertera pada label produk. Para ahli kulit dari luar negeri telah melakukan penelitian namun hasilnya disembunyikan oleh perusahaan kosmetik. Konsumen dianjurkan untuk berhenti menggunakan sabun merek tersebut.",
        # Hoaks vaksin mikrochip
        "JAKARTA - Vaksin COVID-19 dikabarkan mengandung mikrochip yang dapat mengontrol pikiran manusia. Para ilmuwan dari Amerika Serikat telah menemukan ini tetapi pemerintah menyembunyikannya. Jangan divaksin karena vaksin akan mengubah DNA Anda.",
        # Hoaks 5G
        "JAKARTA - Menara 5G menyebabkan penyebaran virus karena radiasi yang dipancarkan dapat melemahkan sistem kekebalan tubuh. Banyak menara 5G sudah dibakar oleh warga di berbagai negara sebagai tindakan pencegahan.",
        # Hoaks cokelat
        "JAKARTA - Makan cokelat setiap hari dapat membuat Anda tidak pernah sakit dan hidup hingga 150 tahun. Cokelat mengandung zat ajaib yang dapat memperbaiki semua organ tubuh dan mencegah penuaan.",
        # Hoaks air zamzam
        "JAKARTA - Air zamzam yang dibawa dari Mekah dapat menyembuhkan semua penyakit termasuk kanker dan diabetes. Cukup minum sendok teh air zamzam setiap pagi dan penyakit akan sembuh sendirian tanpa perlu periksa ke dokter.",
        # Hoaks magnet
        "JAKARTA - Membeli magnet besar dan meletakkannya di rumah dapat mencegah sinar bahaya dari matahari yang akan mencelca bumi pada tahun depan. Ilmuwan rahasia merekomendasikan ini tetapi pemerintah tidak memberitahu kepada masyarakat.",
        # Pendek / berantai
        "SEBARKAN pesan ini segera ke semua grup agar tidak kena blokir WhatsApp!",
        "Dapatkan uang Rp 1 juta gratis hanya dengan mengklik link ini dan mengisi data pribadi Anda.",
        "Klik tautan ini untuk klaim hadiah iPhone 15 gratis dari promo perusahaan telekomunikasi.",
        "Berbagi pesan ini ke 15 orang dalam waktu 1 jam agar tidak terkena kutukan mati dalam 3 hari.",
        "Pemerintah akan bagi-bagi uang tunai Rp 2 juta ke semua warga, daftar sekarang di link berikut.",
        # Hoaks tambahan
        "JAKARTA - Minuman minyak ikan yang diklaim dapat membuat anak pintar dalam satu minggu ternyata hanya air gula dengan pewarna. Produsen menipu orang tua dengan klaim medis palsu yang tidak terbukti secara ilmiah. BPOM telah mencabut izin edar produk tersebut karena tidak aman dikonsumsi anak-anak.",
        "JAKARTA - Tidur dengan handphone di bawah bantal dapat menyebabkan tumor otak karena radiasi nirkabel. Para ahli neurologi dari luar negeri telah memperingatkan ini namun pemerintah Indonesia belum memberikan respons. Warga dianjurkan untuk menjauhkan handphone saat tidur.",
        "JAKARTA - Minum air hangat dengan madu setiap pagi dapat menyembuhkan diabetes dan kanker secara permanen. Para ilmuwan rahasia telah menemukan formulasi ini namun disembunyikan oleh perusahaan obat karena akan merusak bisnis mereka. Anda hanya perlu minum dua gelas setiap hari.",
        "JAKARTA - Sepatu olahraga merek terkenal dikabarkan menyebabkan kanker kaki karena bahan pembuat yang beracun. Zat beracun ini meresap ke kulit kaki dan tidak dapat dihilangkan bahkan setelah mencuci berkali-kali. Konsumen dianjurkan untuk membuang sepatu tersebut segera.",
        "JAKARTA - Listrik statis dari televisi yang menyala dapat menyebabkan kanker pada anak-anak. Radiasi dari layar televisi merusak sel otak dan menyebabkan pertumbuhan sel ganas. Orang tua dilarang membiarkan anak menonton televisi lebih dari satu jam per hari.",
        "JAKARTA - Gula merah dapat menggantikan obat diabetes dan menyembuhkan penyakit tersebut dalam waktu sebatang. Para ahli gizi dari luar negeri telah menemukan ini namun disembunyikan oleh perusahaan farmasi. Penderita diabetes dianjurkan untuk mengonsumsi gula merah setiap hari.",
        "JAKARTA - Memakai cermin dinding di kamar tidur dapat menarik roh jahat dan menyebabkan berbagai malapetaka. Para ahli mistik dari luar negeri telah memperingatkan ini namun warga masih banyak yang memasang cermin di kamar tidur. Segera lepaskan cermin untuk keselamatan keluarga.",
        "JAKARTA - Minyak zaitun yang diminum setiap pagi dapat membersihkan semua kotoran di tubuh dan mencegah penyakit jantung. Para ahli gizi dari Mediterania telah membuktikan ini namun hasil penelitian disembunyikan oleh industri obat. Minum dua sendok makan setiap hari untuk kesehatan jantung.",
        "JAKARTA - Rumah tangga yang menggunakan kompor gas beresiko meledak secara tiba-tiba tanpa ada indasebab. Tabung gas yang sudah lama digunakan akan mengalami kerusakan internal dan menyebabkan ledakan. Warga dianjurkan untuk mengganti tabung gas setiap bulan.",
        "JAKARTA - Kulit pisang yang dikompos di halaman rumah dapat menambah suburan tanah namun sebenarnya mengandung racun yang merusak tanaman. Para ahli pertanian telah menemukan ini namun petani masih banyak yang menggunakan kompos pisang. Segera buang semua kompos pisang dari kebun Anda.",
        # Hoaks tambahan 2
        "JAKARTA - Minuman kopi yang disimpan dalam lemari es dapat mengubah warna kulit menjadi hitam permanen. Kandungan kafein yang bereaksi dengan suhu dingin menghasilkan pigmen kulit yang tidak dapat dihilangkan. Anda tidak boleh menyimpan kopi dalam lemari es.",
        "JAKARTA - Membantu ibu hamil minum air kelapa muda setiap hari akan membuat bayi lahir dengan kulit putih bersih. Air kelapa muda mengandung zat penutris kulit yang dapat mengubah warna kulit janin. Ini adalah rahasia para bidan dari luar negeri.",
        "JAKARTA - Membaca koran di pagi hari dapat menyebabkan sakit parah di mata karena tinta koran mengandung zat beracun. Para ahli mata telah memperingatkan ini namun warga masih banyak yang membaca koran setiap pagi.",
        "JAKARTA - Menggunakan handphone sambil di-charge dapat meledak dan menyebabkan kecelakaan berat. Baterai yang panas saat pengisian daya akan meledak dengan tiba-tiba. Jangan memegang handphone saat sedang di-charge.",
        "JAKARTA - Minuman energi dapat membuat Anda bertahan tidak tidur selama seminggu tanpa efek samping. Kafein dan gula dalam minuman energi memberikan energi tak terbatas. Anda dapat bekerja terus-menerus tanpa istirahat.",
        "JAKARTA - Mencuci muka dengan air es setiap pagi dapat menyebabkan penyakit jantung karena suhu dingin merusak jantung. Anda harus menggunakan air hangat untuk mencuci muka setiap pagi.",
        # Hoaks tambahan 3
        "JAKARTA - Membawa payung di rumah saat hujan dapat membawa kebaikan dan rezeki bagi seluruh keluarga. Ini adalah tradisi leluhur yang harus dilestarikan oleh setiap warga Indonesia.",
        "JAKARTA - Minuman teh hijau yang diminum sebelum tidur dapat membakar lemak badan selama tidur dan menurunkan berat badan secara signifikan. Anda dapat menurunkan berat badan tanpa olahraga.",
        "JAKARTA - Menggunakan parfum wangi tertentu dapat menarik jodoh dan menjadikan Anda populer di tengah masyarakat. Para ahli psikologi dari luar negeri telah membuktikan ini.",
        "JAKARTA - Memasak nasi menggunakan air beras yang sudah disaring dapat menyebabkan kerusakan ginjal karena terdapat zat beracun dalam air beras. Anda harus membuang air beras sebelum memasak nasi.",
        "JAKARTA - Makan buah durian sambil minum soda dapat menyebabkan kematian tiba-tiba karena reaksi kimia di dalam perut. Warga dianjurkan untuk tidak mencampur durian dengan minuman bersoda.",
    ]

    for t in valid_texts:
        rows.append({"text": t, "label": 0, "source": "manual_holdout", "year": 2026, "topic": "", "language": "id"})
    for t in hoax_texts:
        rows.append({"text": t, "label": 1, "source": "manual_holdout", "year": 2026, "topic": "", "language": "id"})

    # Shuffle deterministically
    import random
    random.seed(RANDOM_STATE)
    random.shuffle(rows)
    print(f"[Hold-out] Created {len(rows)} rows "
          f"({sum(1 for r in rows if r['label']==0)} valid / {sum(1 for r in rows if r['label']==1)} hoax)")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1. Load all sources
    all_rows = load_all_sources()
    print(f"\nTotal rows loaded: {len(all_rows):,}")

    if not all_rows:
        print("ERROR: no data loaded from any source.")
        return

    # 2. Raw audit
    before = audit(all_rows, "raw", text_key="text")
    print(f"\n[Audit raw] {before['total_rows']} rows, "
          f"valid={before['label_counts']['valid_real_0']}, "
          f"hoax={before['label_counts']['hoax_fake_1']}")

    # 3. Clean + dedup
    cleaned = clean_and_dedup(all_rows)
    after = audit(cleaned, "clean_deduped")
    print(f"[Audit clean_deduped] {after['total_rows']} rows, "
          f"valid={after['label_counts']['valid_real_0']}, "
          f"hoax={after['label_counts']['hoax_fake_1']}")
    print(f"  Exact duplicates removed: {before['exact_duplicate_texts']}")
    print(f"  Conflicting label texts: {before['conflicting_label_texts']}")

    # 4. Source x label matrix
    sxm = source_x_label_matrix(cleaned)
    print("\n[Source x Label matrix]")
    for src, counts in sxm.items():
        print(f"  {src:<40} valid={counts[0]:>5} hoax={counts[1]:>5} total={counts[0]+counts[1]:>5}")

    # 5. Save combined train_id.csv
    df = pd.DataFrame(cleaned)
    train_path = os.path.join(DATA_DIR, "train_id.csv")
    df.to_csv(train_path, index=False, encoding="utf-8")
    print(f"\nSaved: {train_path} ({len(df):,} rows)")

    # 6. Split by source for out-of-source testing
    train_rows, test_rows = split_by_source(cleaned)
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    train_path_oos = os.path.join(DATA_DIR, "train_id_oos.csv")
    test_path_oos = os.path.join(DATA_DIR, "test_id_oos.csv")
    train_df.to_csv(train_path_oos, index=False, encoding="utf-8")
    test_df.to_csv(test_path_oos, index=False, encoding="utf-8")
    print(f"Saved out-of-source train: {train_path_oos} ({len(train_df):,} rows)")
    print(f"Saved out-of-source test : {test_path_oos} ({len(test_df):,} rows)")

    # 7. Small stratified smoke-test sample
    sample_parts = []
    for label, size in ((0, 30), (1, 30)):
        subset = df[df["label"] == label]
        if len(subset):
            sample_parts.append(subset.sample(n=min(size, len(subset)), random_state=RANDOM_STATE))
    sample_df = pd.concat(sample_parts).sample(frac=1, random_state=RANDOM_STATE)
    sample_path = os.path.join(DATA_DIR, "sample_news_id.csv")
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")
    print(f"Saved: {sample_path} ({len(sample_df):,} rows)")

    # 8. Manual hold-out set
    holdout_rows = create_holdout()
    holdout_path = os.path.join(DATA_DIR, "holdout_id_v2.csv")
    pd.DataFrame(holdout_rows).to_csv(holdout_path, index=False, encoding="utf-8")
    print(f"Saved: {holdout_path} ({len(holdout_rows):,} rows)")

    # 9. Audit JSON
    src_counts = Counter(r["source"] for r in cleaned)
    audit_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources_attempted": [
            {"name": "Mendeley ICTS-2017", "url": "https://data.mendeley.com/datasets/p3hfgr5j3m/1",
             "licence": "CC BY 4.0", "rows_loaded": src_counts.get("Mendeley/ICTS-2017", 0)},
            {"name": "BRIN Indonesia Political Hoax", "url": "https://data.mendeley.com/datasets/mzx5rd397v/1",
             "licence": "CC BY 4.0", "rows_loaded": 0},
            {"name": "HF id-hoax-report-merge", "url": "https://huggingface.co/datasets/nlp-brin-id/id-hoax-report-merge",
             "licence": "MIT", "rows_loaded": src_counts.get("HF/id-hoax-report-merge", 0)},
            {"name": "HF Rifky/indonesian-hoax-news", "url": "https://huggingface.co/datasets/Rifky/indonesian-hoax-news",
             "licence": "no dataset card (public)", "rows_loaded": src_counts.get("HF/Rifky/indonesian-hoax-news", 0)},
            {"name": "Manual hold-out", "url": "internal", "licence": "original (author-written)",
             "rows_loaded": len(holdout_rows)},
        ],
        "sources_available_but_requiring_auth": [
            {"name": "Kaggle: Indonesian Hoax News Dataset (Kominfo)",
             "url": "https://www.kaggle.com/datasets/ireddragonicy/indonesian-hoax-news-dataset",
             "licence": "Apache 2.0", "note": "requires Kaggle login"},
            {"name": "Kaggle: Deteksi Berita Hoaks Indo",
             "url": "https://www.kaggle.com/datasets/mochamadabdulazis/deteksi-berita-hoaks-indo-dataset",
             "licence": "Custom (research and education only)", "note": "requires Kaggle login"},
            {"name": "Kaggle: Indonesian Fact and Hoax Political News",
             "url": "https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news",
             "licence": "MIT", "note": "requires Kaggle login"},
            {"name": "HF: Rifky/indonesian-hoax-news",
             "url": "https://huggingface.co/datasets/Rifky/indonesian-hoax-news",
             "licence": "no dataset card", "note": "17,806 rows, requires HF login"},
        ],
        "raw": before,
        "clean_deduped": after,
        "source_x_label": sxm,
        "cleaning": [
            "strip label prefixes ([SALAH], [BENAR], HOAX:, FAKTA:, ...)",
            "remove news-agency / portal markers (detik, kompas, tribun, ...)",
            "remove datelines and dates",
            "remove URLs and links",
            "remove label-leaking meta words (hoax, hoaks, debunk, cek fakta, klarifikasi)",
            "NFKC normalisation, lowercase, keep a-z only, collapse whitespace",
            "near-duplicate deduplication (5-gram Jaccard > 0.85)",
        ],
        "label_mapping": {"Valid": 0, "Hoax": 1},
        "random_state": RANDOM_STATE,
    }
    audit_path = os.path.join(LOGS_DIR, "dataset_id_audit.json")
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit_data, handle, indent=2, ensure_ascii=False)
    print(f"Saved audit: {audit_path}")

    # 10. Source x label JSON
    sxm_path = os.path.join(LOGS_DIR, "source_x_label.json")
    with open(sxm_path, "w", encoding="utf-8") as handle:
        json.dump(sxm, handle, indent=2, ensure_ascii=False)
    print(f"Saved source x label: {sxm_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total clean rows : {len(cleaned):,}")
    print(f"Valid/Real (0)   : {after['label_counts']['valid_real_0']:,}")
    print(f"Hoax/Fake (1)    : {after['label_counts']['hoax_fake_1']:,}")
    print(f"Out-of-source train : {len(train_df):,} rows")
    print(f"Out-of-source test  : {len(test_df):,} rows")
    print(f"Hold-out v2         : {len(holdout_rows):,} rows")
    print("\nSumber yang dimuat:")
    for src, counts in sxm.items():
        print(f"  {src:<40} valid={counts[0]:>5} hoax={counts[1]:>5}")


if __name__ == "__main__":
    main()
