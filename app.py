import os
import re
import csv
import json
import hashlib
import uuid
import fcntl
import joblib
import unicodedata
import numpy as np
from datetime import datetime, timezone
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
INDOBERT_DIR = os.path.join(MODELS_DIR, 'indobert')
FEEDBACK_PATH = os.path.join(DATA_DIR, 'feedback.csv')

app = Flask(__name__)

# SECRET_KEY hanya dari env (untuk sesi/CSRF bila diaktifkan di masa depan).
# Tidak ada default aman yang dikodekan ke dalam repositori.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', '')

# Fitur feedback dapat dimatikan melalui env (contoh: lingkungan publik
# yang tidak ingin menyimpan masukan pengguna).
FEEDBACK_ENABLED = os.environ.get('FEEDBACK_ENABLED', 'true').lower() in (
    '1', 'true', 'yes', 'on')

# Rate limiting: 30 requests/minute/IP by default, overridable via env var.
# A callable limit value allows tests to lower the limit without re-importing.
RATE_LIMIT = os.environ.get('RATE_LIMIT_PER_MINUTE', '30 per minute')
app.config['RATE_LIMIT'] = RATE_LIMIT
app.config.setdefault(
    'RATELIMIT_STORAGE_URI',
    os.environ.get('RATELIMIT_STORAGE_URI', 'memory://'),
)

# ProxyFix: hanya percaya header X-Forwarded-* dari N proxy terdekat yang
# diatur melalui TRUSTED_PROXIES (default 0 = abaikan semua header, gunakan
# alamat koneksi langsung). Ini mencegah penyerang menyamar IP melalui
# X-Forwarded-For palsu.
_TRUSTED_PROXIES = int(os.environ.get('TRUSTED_PROXIES', '0'))
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=_TRUSTED_PROXIES)

limiter = Limiter(app=app, key_func=get_remote_address)

# Peringatan saat storage in-memory dengan banyak worker: setiap worker
# punya counter sendiri, jadi rate limit tidak akurat di produksi multi-worker.
_WORKERS = int(os.environ.get('WEB_CONCURRENCY', '1'))
if (app.config['RATELIMIT_STORAGE_URI'] == 'memory://' and _WORKERS > 1):
    print(
        'Peringatan: rate limiter memory:// dengan beberapa worker '
        f'({_WORKERS}) tidak sinkron. Gunakan 1 worker + thread, atau '
        'RATELIMIT_STORAGE_URI=redis://... untuk produksi multi-worker.'
    )

# ---------------------------------------------------------------------------
# Konfigurasi feedback
# ---------------------------------------------------------------------------
FEEDBACK_MAX_CHARS = int(os.environ.get('FEEDBACK_MAX_CHARS', '20000'))
FEEDBACK_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
_FEEDBACK_FIELDS = ['id', 'timestamp', 'text_hash', 'text', 'label',
                    'status', 'reviewer']
_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r')


def _sanitize_feedback_cell(value: str) -> str:
    """Hindari formula injection CSV: sel yang diawali karakter khusus
    (=, +, -, @, tab, CR) diberi awalan tanda kutip tunggal."""
    if isinstance(value, str) and value and value[0] in _FORMULA_PREFIXES:
        return "'" + value
    return value


def _count_feedback() -> int:
    """Hitung jumlah baris feedback (tidak termasuk header)."""
    if not os.path.exists(FEEDBACK_PATH):
        return 0
    try:
        with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Batas input (antisipasi DoS / payload besar)
# ---------------------------------------------------------------------------
# Maksimal panjang teks yang dikirim ke /predict (karakter, setelah strip).
MAX_TEXT_CHARS = int(os.environ.get('MAX_TEXT_CHARS', '20000'))
# Maksimal ukuran body request (256 KB). Flask akan menanggapi 413 terlebih
# dahulu sebelum memproses body.
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024


def current_rate_limit() -> str:
    """Return the active rate limit (config first, env/default as fallback)."""
    return app.config.get('RATE_LIMIT', RATE_LIMIT)


def missing_models(primary_unavailable: bool = False) -> list[str]:
    """Daftar model yang tidak tersedia pada runtime ini.

    ``primary_unavailable`` dipakai oleh /predict saat alias ``model``/``tfidf``
    kosong walaupun artefak Inggris masih ada di disk (mis. monkeypatch tes).
    """
    missing = []
    if primary_unavailable or model_en is None or tfidf_en is None:
        missing.append('en')
    if model_id is None or tfidf_id is None:
        missing.append('id')
    return missing


MODEL_UNAVAILABLE_MESSAGE = (
    'Model belum siap. Jalankan train_model.py (Inggris) atau '
    'train_model_id.py (Indonesia), lalu restart server. '
    'Lihat README.md bagian "Cara menjalankan dari nol".'
)


@app.errorhandler(429)
def too_many_requests(error):
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': f'Request terlalu besar. Maksimal {app.config["MAX_CONTENT_LENGTH"] // 1024} KB.'
    }), 413

# ---------------------------------------------------------------------------
# Honest result reporting: three-zone status, red-flag rules, language hint.
# All tunable values live here so they are easy to adjust in one place.
# ---------------------------------------------------------------------------
# Three-zone thresholds use probability_fake as a fraction (0..1), not percent.
THRESHOLD_FAKE = 0.70
THRESHOLD_VALID = 0.30

# Common Indonesian function words used for the lightweight language hint.
INDONESIAN_COMMON_WORDS = frozenset({
    'yang', 'dan', 'di', 'untuk', 'dengan', 'dari', 'pada', 'ini', 'itu',
    'tidak', 'akan', 'adalah', 'karena', 'oleh', 'juga',
})

# Rule-based red flags. Each rule fires at most once; order is display order.
RED_FLAG_RULES = (
    {
        'kode': 'ajakan_menyebar',
        'label': 'Mengajak menyebarkan pesan ke banyak orang',
        'pola': (
            r'\bsebarkan\b', r'\bbagikan\b', r'\bkirim\s+ke\s+grup\b',
            r'\bforward\b', r'\bviralkan\b', r'\bshare\b',
        ),
    },
    {
        'kode': 'tekanan_waktu',
        'label': 'Menekan agar bertindak cepat',
        'pola': (
            r'\bsegera\b', r'\bsebelum\s+ditutup\b', r'\bditutup\b',
            r'\bhangus\b', r'\bterakhir\b', r'\bdarurat\b',
            r'\bjangan\s+sampai\b', r'\bhari\s+ini\s+juga\b',
        ),
    },
    {
        'kode': 'pemicu_panik',
        'label': 'Memakai bahasa yang memicu panik',
        'pola': (
            r'\bterhapus\b', r'\bdiblokir\b', r'\bdibekukan\b',
            r'\bbahaya\b', r'\bawas\b', r'\bwaspada\b',
        ),
    },
    {
        'kode': 'tanpa_sumber',
        'label': 'Merujuk sumber tanpa menyebut nama atau tautan resmi',
        'pola': (
            r'\bkatanya\b', r'\bkonon\b', r'\bberedar\s+kabar\b',
            r'\bberedar\s+informasi\b', r'\bmenurut\s+sumber\b',
        ),
    },
)

_SHORTENER_RE = re.compile(
    r'\b(?:bit\.ly|tinyurl\.com|t\.co|s\.id|cutt\.ly)\b', re.IGNORECASE)
_URL_RE = re.compile(r'(?:https?://|www\.)[^\s<>"\']+', re.IGNORECASE)
_DOMAIN_RE = re.compile(
    # Satu titik (label.tld): linear, tidak ada nested quantifier, aman dari
    # ReDoS bahkan saat findall dijalankan pada input ribuan karakter.
    r'\b[a-z0-9-]+\.[a-z]{2,}\b',
    re.IGNORECASE,
)


def determine_status(probability_fake: float) -> str:
    """Map probability_fake (0..1) to one of three honest result zones."""
    try:
        value = float(probability_fake)
    except (TypeError, ValueError):
        value = 0.0
    if value >= THRESHOLD_FAKE:
        return 'cenderung_hoaks'
    if value <= THRESHOLD_VALID:
        return 'cenderung_valid'
    return 'belum_dapat_disimpulkan'


def _has_suspicious_link(text: str) -> bool:
    """True when the text has a shortener or any URL that is not *.go.id."""
    if _SHORTENER_RE.search(text):
        return True
    for raw_url in _URL_RE.findall(text):
        candidate = (raw_url if re.match(r'^https?://', raw_url, re.IGNORECASE)
                     else 'http://' + raw_url)
        host = (urlparse(candidate).hostname or '').lower().rstrip('.')
        if not (host == 'go.id' or host.endswith('.go.id')):
            return True
    # Periksa domain tulis (tanpa skema) hanya pada teks non-URL agar
    # subdomain *.go.id di dalam URL tidak salah terdeteksi (mis. kominfo.go
    # di dalam https://kominfo.go.id/...).  Pola satu titik = linear, aman ReDoS.
    text_tanpa_url = _URL_RE.sub(' ', text)
    for domain in _DOMAIN_RE.findall(text_tanpa_url):
        host = domain.lower().rstrip('.')
        if not (host == 'go.id' or host.endswith('.go.id')):
            return True
    return False


# Keywords that may appear in urgency contexts, but only fire ceria when
# near an invitation keyword within 80 characters.
URGENCY_WITHIN = {
    'segera', 'terakhir', 'ditutup', 'darurat', 'jangan sampai'
}
# Urgency keywords that may ONLY fire if a personal-threat word is also present.
PERSONAL_THREAT = {
    'terhapus', 'diblokir', 'dibekukan', 'hangus', 'dicabut', 'denda'
}
# Personal-data request patterns (OTP, PIN, password, etc.)
PERSONAL_DATA_PATTERN = re.compile(
    r'\b(?:OTP|PIN|password|nomor\s+rekening|nomor\s+KTP)\b', re.IGNORECASE)

def detect_red_flags(text: str):
    """Return rule-based warning signs (no model involved), at most five."""
    if not isinstance(text, str) or not text.strip():
        return []

    flags = []
    text_lower = text.lower()
    n = len(text)

    # ---------- 1. Cari apakah ada ajakan_menyebar ----------
    ajakan_keywords = [
        'sebarkan', 'bagikan', 'kirim', 'forward', 'viralkan', 'share',
        'klik', 'daftar', 'transfer', 'hubungi nomor',
    ]
    has_ajakan = any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
                     for kw in ajakan_keywords)

    # ---------- 2. Tekanan waktu: aktif hanya jika ada kata urgensi
    # di dalam jarak 80 karakter (ke kiri atau kanan) dari kata ajakan ----------
    URGENCY_WITHIN = {'segera', 'terakhir', 'ditutup', 'darurat', 'jangan sampai'}
    has_tekanan_waktu = False
    if has_ajakan:
        # temukan semua posisi kata ajakan
        ajakan_positions = [m.start()
                            for m in re.finditer(
                                '|'.join(r'\b' + re.escape(kw) + r'\b' for kw in ajakan_keywords),
                                text_lower)]
        # untuk setiap posisi ajakan, cek apakah ada kata urgensi dalam +/- 80 char
        for pos in ajakan_positions:
            window_start = max(0, pos - 80)
            window_end = min(n, pos + 80)
            window = text_lower[window_start:window_end]
            if any(k in window for k in URGENCY_WITHIN):
                has_tekanan_waktu = True
                break

    if has_ajakan:
        flags.append({
            'kode': 'ajakan_menyebar',
            'label': 'Mengajak menyebarkan pesan ke banyak orang',
        })
    if has_tekanan_waktu:
        flags.append({
            'kode': 'tekanan_waktu',
            'label': 'Menekan agar bertindak cepat',
        })

    # ---------- 2b. Tanpa sumber: merujuk sumber tanpa menyebut nama atau tautan resmi ----------
    TANPA_SUMBER_PATTERNS = (
        r'\bkatanya\b', r'\bkonon\b', r'\bberedar\s+kabar\b',
        r'\bberedar\s+informasi\b', r'\bmenurut\s+sumber\b',
    )
    has_tanpa_sumber = any(re.search(p, text_lower) for p in TANPA_SUMBER_PATTERNS)
    if has_tanpa_sumber:
        flags.append({
            'kode': 'tanpa_sumber',
            'label': 'Merujuk sumber tanpa menyebut nama atau tautan resmi',
        })

    # ---------- 3. Pemicu panik: aktif hanya jika ada kata ancangan pribadi
    # bersamaan dengan ajakan ----------
    PERSONAL_THREAT = {
        'terhapus', 'diblokir', 'dibekukan', 'hangus', 'dicabut', 'denda',
    }
    PERSONAL_DATA = re.compile(r'\b(?:OTP|PIN|password|nomor\s+rekening|nomor\s+KTP)\b',
                               re.IGNORECASE)

    has_panik = False
    if has_ajakan:
        # Cek apakah ada kata ancangan pribadi atau data pribadi di teks
        for pw in PERSONAL_THREAT:
            if re.search(r'\b' + re.escape(pw) + r'\b', text_lower):
                has_panik = True
                break
        if not has_panik:
            if PERSONAL_DATA.search(text_lower):
                has_panik = True

    if has_ajakan and has_panik:
        flags.append({
            'kode': 'pemicu_panik',
            'label': 'Memakai bahasa yang memicu panik',
        })

    # ---------- 4. Tautan mencurigakan ----------
    if _has_suspicious_link(text):
        flags.append({
            'kode': 'tautan_mencurigakan',
            'label': 'Memuat tautan pendek atau tidak resmi',
        })

    # ---------- 5. Huruf kapital berlebihan ----------
    letters = [char for char in text if char.isalpha()]
    uppercase_ratio = (
        sum(1 for char in letters if char.isupper()) / len(letters)
        if letters else 0.0
    )
    if (len(text) > 40 and uppercase_ratio > 0.30) or '!!' in text:
        flags.append({
            'kode': 'huruf_berlebihan',
            'label': 'Huruf kapital atau tanda seru berlebihan',
        })

    return flags[:5]


def detect_credibility_signals(text: str):
    """Return rule-based credibility signals (no model involved), at most four.

    These are format / completeness markers, not truth indicators.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    signals = []
    text_norm = text.strip()
    n = len(text_norm)

    # 1. dateline: capitalize city name followed by dash or colon at the very start
    DATETIME_RE = re.compile(
        r'^[A-Z][A-Za-z\s]+[-\:]\s', re.IGNORECASE,
    )
    if re.search(DATETIME_RE, text_norm):
        signals.append({
            'kode': 'dateline',
            'label': 'Memuat lokasi dan format penulisan berita',
        })

    # 2. narasumber_bernama: jabatan/gelar diikuti nama, atau kata "menyatakan"/"menegaskan"/"menyampaikan" didekat nama berhuruf kapital
    NARASUMBER_RE = re.compile(
        r'\b(?:Presiden|Menteri|Anggota\s+DPR|Kepala|Direktur|Dr\.?|Prof\.)\s+[A-Z][A-Za-z\s]+'
        r'|(?:menyatakan|menegaskan|menyampaikan)\s+[A-Z][A-Za-z\s]+',
        re.IGNORECASE,
    )
    if re.search(NARASUMBER_RE, text_norm):
        signals.append({
            'kode': 'narasumber_bernama',
            'label': 'Menyebut narasumber dengan nama dan jabatan',
        })

    # 3. waktu_tempat_spesifik: ada tanggal atau hari + bulan/tahun
    WAKTU_RE = re.compile(
        r'\b(?:Jumat|Senin|Selasa|Rabu|Kamis|Jumat)\s*\(?\s*\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}\s*\)?'
        r'|\b\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{2,4}\b',
        re.IGNORECASE,
    )
    if re.search(WAKTU_RE, text_norm):
        signals.append({
            'kode': 'waktu_tempat_spesifik',
            'label': 'Menyebut waktu dan peristiwa secara spesifik',
        })

    # 4. rujukan_resmi: institusi resmi atau domain .go.id
    RUJUN_RE = re.compile(
        r'\b(?:DPR|Kementerian|Polri|BPS|Rapat\s+Paripurna|Perpres)'
        r'|(?:https?://|www\.)[^\s<>"\']+\.go\.id\b', re.IGNORECASE,
    )
    if re.search(RUJUN_RE, text_norm):
        signals.append({
            'kode': 'rujukan_resmi',
            'label': 'Merujuk lembaga atau dokumen resmi',
        })

    # 5. tanpa_ajakan: tidak ada ciri ajakan_menyebar
    # kita cek di sini karena sudah punya has_ajakan dari fungsi di atas;
    # kurangkan agar tidak kembar dengan detect_red_flags.
    # Kita cek di sini dengan pola kasar apakah ada ajakan_menyebar sama sekali.
    AJakan_KOSAK = re.compile(
        r'\b(?:sebarkan|bagikan|kirim\s+ke\s+grup|forward|viralkan|share|'
        r'klik|daftar|transfer|hubungi\s+nomor)\b', re.IGNORECASE,
    )
    has_ajakan_sama_aja = bool(re.search(AJakan_KOSAK, text_norm))
    if not has_ajakan_sama_aja:
        signals.append({
            'kode': 'tanpa_ajakan',
            'label': 'Tidak mengajak menyebarkan pesan',
        })

    return signals[:4]


def is_indonesian(text: str) -> bool:
    """Lightweight hint: >=8% of words are common Indonesian function words."""
    if not isinstance(text, str):
        return False
    words = re.findall(r'[a-z]+', text.lower())
    if not words:
        return False
    hits = sum(1 for word in words if word in INDONESIAN_COMMON_WORDS)
    return (hits / len(words)) >= 0.08


def build_advice(status: str, probability_real: float,
                 probability_fake: float, red_flag_count: int = 0,
                 credibility_count: int = 0) -> dict:
    """Return honest, non-verdict wording for the given status.

    ``red_flag_count`` and ``credibility_count`` are derived from rule-based
    checks (no model involvement) and only influence the advice text, not
    the model status or badge colour.
    """
    # --- Saran khusus untuk status abu-abu: belum dapat disimpulkan ---
    # Matriks: status × jumlah red_flag (0, 1-2, >=3).
    if status == 'belum_dapat_disimpulkan':
        ringkasan = (
            f'Kemungkinan valid {probability_real:.1f}%, '
            f'kemungkinan hoaks {probability_fake:.1f}%.'
        )
        if red_flag_count == 0 and credibility_count == 0:
            return {
                'judul': 'Belum dapat disimpulkan',
                'ringkasan': ringkasan,
                'saran': (
                    'Tidak ditemukan ciri pesan berantai atau penipuan, tetapi '
                    'model belum yakin. Cocokkan dengan media tepercaya sebelum '
                    'mempercayai atau membagikannya.'
                ),
            }
        if red_flag_count >= 3:
            return {
                'judul': 'Belum dapat disimpulkan, tetapi patut diwaspadai',
                'ringkasan': ringkasan,
                'saran': (
                    'Jangan dibagikan dulu. Teks ini memiliki beberapa ciri yang '
                    'sering muncul dalam hoaks dan penipuan. Pastikan '
                    'kebenarannya lewat situs resmi atau media tepercaya sebelum '
                    'mempercayai atau membagikannya.'
                ),
            }
        # 1-2 red flag: indikasi mencurigakan, tapi belum cukup bukti.
        if 1 <= red_flag_count <= 2:
            return {
                'judul': 'Belum dapat disimpulkan, ada indikasi mencurigakan',
                'ringkasan': ringkasan,
                'saran': (
                    'Tidak tersedia cukup bukti untuk menyimpulkan, tetapi ditemukan '
                    'ciri mencurigakan. Verifikasi ke media tepercaya sebelum '
                    'membagikannya.'
                ),
            }
        # red_flag_count == 0 (dengan credibility saja): saran netral.
        return {
            'judul': 'Belum dapat disimpulkan',
            'ringkasan': ringkasan,
            'saran': (
                'Tidak ditemukan ciri pesan berantai atau penipuan, tetapi '
                'model belum yakin. Cocokkan dengan media tepercaya sebelum '
                'mempercayai atau membagikannya.'
            ),
        }

    # --- Standar ---
    if status == 'cenderung_hoaks':
        return {
            'judul': 'Cenderung hoaks',
            'ringkasan': (
                'Pola bahasa teks ini mirip dengan berita palsu yang pernah '
                'dipelajari sistem.'
            ),
            'saran': (
                'Jangan dibagikan dulu. Periksa di turnbackhoax.id atau situs '
                'resmi instansi terkait.'
            ),
        }

    if status == 'cenderung_valid':
        ringkasan = (
            'Pola bahasa teks ini mirip dengan berita yang wajar. Ini '
            'bukan jaminan isinya benar.'
        )
        if red_flag_count >= 3:
            ringkasan += (
                ' Model menilai teks ini wajar, tetapi ada beberapa ciri '
                'yang perlu diwaspadai.'
            )
        return {
            'judul': 'Cenderung valid',
            'ringkasan': ringkasan,
            'saran': (
                'Tetap cocokkan dengan sumber resmi, terutama untuk informasi '
                'penting seperti kesehatan, uang, dan bantuan sosial.'
            ),
        }

    # Fallback untuk status yang tidak diketahui.
    return {
        'judul': 'Belum dapat disimpulkan',
        'ringkasan': (
            f'Kemungkinan valid {probability_real:.1f}%, '
            f'kemungkinan hoaks {probability_fake:.1f}%.'
        ),
        'saran': (
            'Anggap informasi ini belum terverifikasi. Cari berita yang sama '
            'di media tepercaya atau situs resmi sebelum percaya atau '
            'membagikannya.'
        ),
    }


# ---------------------------------------------------------------------------
# Model artefacts: English (primary) and Indonesian (optional regional model).
# ---------------------------------------------------------------------------
ID_MODELS_DIR = os.path.join(MODELS_DIR, 'id')

# English (primary) artefacts.
model_en_path = os.path.join(MODELS_DIR, 'model.pkl')
vec_en_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')
weights_en_path = os.path.join(MODELS_DIR, 'feature_weights.json')
threshold_en_path = os.path.join(MODELS_DIR, 'threshold.json')

# Indonesian artefacts.
model_id_path = os.path.join(ID_MODELS_DIR, 'model.pkl')
vec_id_path = os.path.join(ID_MODELS_DIR, 'vectorizer.pkl')
weights_id_path = os.path.join(ID_MODELS_DIR, 'feature_weights.json')
threshold_id_path = os.path.join(ID_MODELS_DIR, 'threshold.json')

# ---------------------------------------------------------------------------
# Integritas artefak model (checksum SHA-256).
# models/CHECKSUMS.txt berisi <sha256>  <rel_path> untuk setiap artefak.
# Saat startup, berkas yang ada di disk tetapi hash-nya tidak cocok akan
# ditandai rusak dan tidak dimuat. Lewati seluruh verifikasi dengan
# SKIP_CHECKSUM=1 (contoh: artefak belum tersedia saat fresh clone).
# ---------------------------------------------------------------------------
CHECKSUMS_PATH = os.path.join(MODELS_DIR, 'CHECKSUMS.txt')
SKIP_CHECKSUM = os.environ.get('SKIP_CHECKSUM', '').lower() in ('1', 'true', 'yes', 'on')


def _load_checksums(path=CHECKSUMS_PATH):
    """Baca CHECKSUMS.txt -> dict {rel_path_normalized: expected_sha256}."""
    expected = {}
    if not os.path.exists(path):
        return expected
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                digest, rel = parts[0].lower(), parts[1].strip()
                # Normalisasi './' dan pemisah direktori.
                rel = rel.replace('./', '').replace('\\', '/')
                expected[rel] = digest
    except Exception as e:
        print(f"Warning: Could not read checksums file {path}: {e}")
    return expected


def _corrupt_artifact_paths(expected):
    """Kembalikan set rel_path yang ada di disk tetapi hash-nya tidak cocok."""
    corrupt = set()
    if SKIP_CHECKSUM or not expected:
        return corrupt
    for rel, digest in expected.items():
        full = os.path.join(MODELS_DIR, rel)
        if not os.path.exists(full):
            # Artefak git-ignored / belum tersedia: tidak dianggap rusak.
            continue
        try:
            with open(full, 'rb') as fh:
                actual = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            actual = ''
        if actual != digest:
            corrupt.add(rel)
            print(
                f"Checksum MISMATCH for model artefact '{rel}': "
                f"expected {digest}, got {actual}. Artefak tidak akan dimuat."
            )
    return corrupt


eval_path = os.path.join(LOGS_DIR, 'evaluation.json')

model_en = None
tfidf_en = None
feature_weights_en = {}
model_id = None
tfidf_id = None
feature_weights_id = {}
eval_metrics = None
DECISION_THRESHOLD_EN = 0.5
DECISION_THRESHOLD_ID = 0.5


def _load_json(path, default):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default


# Muat ekspektasi checksum dan hitung artefak yang rusak (ada di disk namun
# hash tidak cocok) SEBELUM artefak dimuat, sehingga blok pemuatan di bawah
# dapat melewati artefak korup.
_checksums_expected = _load_checksums()


def _rel_models_path(abs_path):
    """Relasi path terhadap MODELS_DIR, serasi dengan entri CHECKSUMS.txt."""
    try:
        rel = os.path.relpath(abs_path, MODELS_DIR).replace('\\', '/')
    except ValueError:
        rel = abs_path
    return rel


_corrupt_paths = _corrupt_artifact_paths(_checksums_expected)
if _corrupt_paths:
    print(f"Artefak model dilewatkan karena checksum tidak cocok: "
          f"{sorted(_corrupt_paths)}")


def _artifact_is_usable(abs_path):
    """False bila artefak ada di disk tetapi tercantum sebagai rusak."""
    return _rel_models_path(abs_path) not in _corrupt_paths


try:
    if os.path.exists(threshold_en_path):
        DECISION_THRESHOLD_EN = float(
            _load_json(threshold_en_path, {}).get('threshold', 0.5))
        print(f"English decision threshold loaded: {DECISION_THRESHOLD_EN}")
except Exception as e:
    print(f"Warning: Could not load English threshold: {e}")

try:
    if os.path.exists(threshold_id_path):
        DECISION_THRESHOLD_ID = float(
            _load_json(threshold_id_path, {}).get('threshold', 0.5))
        print(f"Indonesian decision threshold loaded: {DECISION_THRESHOLD_ID}")
except Exception as e:
    print(f"Warning: Could not load Indonesian threshold: {e}")

try:
    if (os.path.exists(model_en_path) and os.path.exists(vec_en_path)
            and _artifact_is_usable(model_en_path)
            and _artifact_is_usable(vec_en_path)):
        model_en = joblib.load(model_en_path)
        tfidf_en = joblib.load(vec_en_path)
        print("English model and vectorizer loaded successfully.")
    if os.path.exists(weights_en_path) and _artifact_is_usable(weights_en_path):
        feature_weights_en = _load_json(weights_en_path, {})
        print(f"Loaded {len(feature_weights_en)} English feature weights for XAI.")
except Exception as e:
    print(f"Error loading English model artifacts: {e}")

try:
    if (os.path.exists(model_id_path) and os.path.exists(vec_id_path)
            and _artifact_is_usable(model_id_path)
            and _artifact_is_usable(vec_id_path)):
        model_id = joblib.load(model_id_path)
        tfidf_id = joblib.load(vec_id_path)
        print("Indonesian model and vectorizer loaded successfully.")
    if os.path.exists(weights_id_path) and _artifact_is_usable(weights_id_path):
        feature_weights_id = _load_json(weights_id_path, {})
        print(f"Loaded {len(feature_weights_id)} Indonesian feature weights for XAI.")
except Exception as e:
    print(f"Error loading Indonesian model artifacts: {e}")

# Backwards-compatible aliases: the default / active artefacts are English.
model = model_en
tfidf = tfidf_en
feature_weights = feature_weights_en
DECISION_THRESHOLD = DECISION_THRESHOLD_EN

try:
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_metrics = json.load(f)
except Exception as e:
    print(f"Warning: Could not load evaluation metrics: {e}")

# ---------------------------------------------------------------------------
# Gerbang keandalan: MODEL_RELIABLE DIHITUNG dari logs/holdout_v3_result.json.
# Tidak ada penulisan manual dan tidak ada jalur pintas dari metrik hold-out v2
# atau model card IndoBERT (keduanya bukan bukti untuk hold-out v3).
# ---------------------------------------------------------------------------
from reliability import compute_model_reliable  # noqa: E402

RELIABILITY = compute_model_reliable()
MODEL_RELIABLE = bool(RELIABILITY['reliable'])
print(f"MODEL_RELIABLE={MODEL_RELIABLE} ({'; '.join(RELIABILITY['reasons']) or 'semua gerbang lulus'})")

# ---------------------------------------------------------------------------
# TAHAP B: Pencocokan dengan hoaks yang sudah dibantah (debunked-claim index).
# Menggunakan indeks embedding judul klaim hoak dari turnbackhoax.id yang sudah
# dibangun (models/hoax_index.npy + models/hoax_index_meta.json). Ambang
# kemiripan ditetapkan dari hasil validasi hold-out v2 (logs/debunk_threshold.json)
# agar false match pada berita valid rendah. Hanya judul + tautan disimpan,
# tidak ada teks penuh artikel berhak cipta.
# ---------------------------------------------------------------------------
HOAX_INDEX = None          # np.ndarray (n_index, embed_dim), sudah L2-normalisasi
HOAX_META = None           # list of dict {judul, url, tanggal, summary}
HOAX_EMBED_MODEL = None    # SentenceTransformer (dimuat lazy)
_index_loaded = False
DEBUNK_THRESHOLD = 0.66    # default; ditimpa dari logs/debunk_threshold.json bila ada
DEBUNK_TOP_K = 3
DEBUNK_AVAILABLE = False   # True bila sentence-transformers + indeks tersedia


def _load_debunk_index():
    """Muha-muat indeks embedding hoak yang sudah dibantah (hanya sekali)."""
    global HOAX_INDEX, HOAX_META, HOAX_EMBED_MODEL, _index_loaded, DEBUNK_THRESHOLD, DEBUNK_AVAILABLE
    if _index_loaded:
        return
    _index_loaded = True
    index_path = os.path.join(MODELS_DIR, "hoax_index.npy")
    meta_path = os.path.join(MODELS_DIR, "hoax_index_meta.json")
    try:
        if os.path.exists(index_path) and os.path.exists(meta_path):
            HOAX_INDEX = np.load(index_path).astype(np.float32)
            raw_meta = json.load(open(meta_path, encoding="utf-8"))
            # Validasi skema URL sisi server: hanya terima http/https;
            # buang entri dengan skema lain (mis. javascript:) sebagai
            # pertahanan dalam bertingkat terhadap XSS melalui href.
            HOAX_META = [
                m for m in raw_meta
                if urlparse(m.get('url', '')).scheme in ('http', 'https')
            ]
            # Ambang dari hasil validasi hold-out v2, bila tersedia.
            thr_path = os.path.join(LOGS_DIR, "debunk_threshold.json")
            if os.path.exists(thr_path):
                try:
                    with open(thr_path, "r", encoding="utf-8") as f:
                        thr_data = json.load(f)
                    DEBUNK_THRESHOLD = float(thr_data.get("chosen_threshold", DEBUNK_THRESHOLD))
                except Exception:
                    pass
            print(f"Debunk index dimuat: {len(HOAX_META)} entri (dari {len(raw_meta)} mentah), ambang={DEBUNK_THRESHOLD}")
            DEBUNK_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Tidak dapat memuat indeks debunked: {e}")


def _embed_texts(texts):
    """Embed teks dengan model multibahasa yang sama seperti indeks."""
    global HOAX_EMBED_MODEL
    if HOAX_EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        HOAX_EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return HOAX_EMBED_MODEL.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def find_similar_debunked(text, top_k=DEBUNK_TOP_K):
    """Cari hoak yang sudah dibantah mirip dengan teks.

    Mengembalikan list berurutan menurun menurut kemiripan:
        {"judul": str, "url": str, "similarity": float}
    Hanya item dengan similarity >= DEBUNK_THreshold diikutsertakan.
    Jika sentence-transformers tidak terpasang atau indeks tidak tersedia,
    mengembalikan list kosong tanpa error (graceful fallback).
    """
    _load_debunk_index()
    if HOAX_INDEX is None or HOAX_META is None:
        return []
    try:
        q = _embed_texts([text])[0]
    except Exception:
        # sentence-transformers tidak terpasang / gagal dimuat -> fallback kosong
        return []
    # Vektor already L2-normalisasi -> dot product = cosine similarity.
    sims = HOAX_INDEX @ q
    order = np.argsort(-sims)[: max(1, int(top_k))]
    results = []
    for i in order:
        sim = float(sims[int(i)])
        if sim < DEBUNK_THRESHOLD:
            continue
        m = HOAX_META[int(i)]
        results.append({
            "judul": m.get("judul", ""),
            "url": m.get("url", ""),
            "similarity": round(sim, 4),
        })
    return results


# ---------------------------------------------------------------------------
# TAHAP C: Loader opsional IndoBERT (fine-tuned via Google Colab GPU).
# Dipakai untuk teks Indonesia HANYA bila folder models/indobert/ tersedia,
# semua checksum di checksums.txt cocok, dan dependensi terpasang. Jika tidak,
# fallback ke model lama (TF-IDF + LogisticRegression). MODEL_RELIABLE tidak
# ditentukan di sini: dihitung reliability.py dari hold-out v3 (Butir 4).
# ---------------------------------------------------------------------------
INDOBERT_MODEL = None
INDOBERT_TOKENIZER = None
INDOBERT_DEVICE = None
_INDOBERT_READY = False
INDOBERT_THRESHOLD = 0.5


def _indobert_checksums_ok(indobert_dir=INDOBERT_DIR):
    """True hanya jika checksums.txt ada dan SEMUA entri cocok.

    Mengembalikan (ok, alasan). Tidak ada model yang dimuat bila berkas
    checksum hilang/tidak cocok — mencegah artefak setengah jadi dipakai.
    """
    checksums = os.path.join(indobert_dir, 'checksums.txt')
    if not os.path.isdir(indobert_dir):
        return False, 'folder models/indobert/ tidak ada'
    if not os.path.exists(checksums):
        return False, 'models/indobert/checksums.txt tidak ada'
    verified = 0
    try:
        with open(checksums, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                expected, name = parts[0].lower(), parts[1].strip().replace('./', '')
                path = os.path.join(indobert_dir, name)
                if not os.path.isfile(path):
                    return False, f'berkas {name} tidak ada'
                with open(path, 'rb') as f:
                    actual = hashlib.sha256(f.read()).hexdigest()
                if actual != expected:
                    return False, f'checksum {name} tidak cocok'
                verified += 1
    except Exception as exc:
        return False, f'gagal membaca checksums.txt: {exc}'
    if verified < 2:
        return False, 'checksum terlalu sedikit (minimal 2 berkas)'
    return True, f'{verified} berkas terverifikasi'


def _load_indobert_optional():
    """Muat IndoBERT dari models/indobert/ (hanya sekali, checksum harus cocok)."""
    global INDOBERT_MODEL, INDOBERT_TOKENIZER, INDOBERT_DEVICE, \
        _INDOBERT_READY, INDOBERT_THRESHOLD
    if _INDOBERT_READY:
        return
    _INDOBERT_READY = True
    ok, reason = _indobert_checksums_ok()
    if not ok:
        print(f'TAHAP C: IndoBERT dilewati ({reason}), fallback ke model lama.')
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception:
        print('TAHAP C: torch/transformers tidak terpasang, fallback ke model lama.')
        return
    try:
        INDOBERT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        INDOBERT_TOKENIZER = AutoTokenizer.from_pretrained(INDOBERT_DIR)
        INDOBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            INDOBERT_DIR
        ).to(INDOBERT_DEVICE)
        INDOBERT_MODEL.eval()
        # Ambang keputusan: dari metrics.json bila tersedia, else 0.5.
        met_path = os.path.join(INDOBERT_DIR, 'metrics.json')
        if os.path.exists(met_path):
            try:
                with open(met_path, 'r', encoding='utf-8') as f:
                    met = json.load(f)
                ho = met.get('holdout_v2', {})
                if ho.get('accuracy') is not None:
                    INDOBERT_THRESHOLD = float(ho.get('decision_threshold', 0.5))
            except Exception:
                pass
        print(f'TAHAP C: IndoBERT dimuat dari {INDOBERT_DIR}, ambang={INDOBERT_THRESHOLD}')
    except Exception as e:
        print(f'TAHAP C: Gagal memuat IndoBERT ({e}), fallback ke model lama.')
        INDOBERT_MODEL = None
        INDOBERT_TOKENIZER = None


def indobert_predict_proba(text):
    """Probabilitas hoak (0..1) dari IndoBERT, atau None jika tidak tersedia."""
    _load_indobert_optional()
    if INDOBERT_MODEL is None or INDOBERT_TOKENIZER is None:
        return None
    try:
        import torch
        enc = INDOBERT_TOKENIZER(
            text, truncation=True, padding=True, max_length=256,
            return_tensors='pt',
        )
        enc = {k: v.to(INDOBERT_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = INDOBERT_MODEL(**enc).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(prob)
    except Exception as e:
        print(f'TAHAP C: IndoBERT inferensi gagal ({e}), fallback ke model lama.')
        return None


from id_preprocessing import detect_language, clean_text as id_clean_text


def parse_feedback_label(value):
    """Strictly parse a feedback label.

    Accept only 0/1 integers or the strings '0'/'1'.  Booleans and floats are
    rejected even though ``True == 1`` and ``1.0 == 1`` in Python, because
    silently accepting them makes the saved label ambiguous.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in {'0', '1'}:
        return int(value.strip())
    return None


def explain_prediction(cleaned_text: str, top_n: int = 5, weights: dict | None = None):
    """Identify top influential keywords in input using model weights (XAI).

    ``weights`` defaults to the module-level ``feature_weights`` (English) so
    existing callers keep working; callers serving another language pass that
    language's feature weights instead.
    """
    if weights is None:
        weights = feature_weights
    words = cleaned_text.split()
    matched_features = []

    # Check unigrams and bigrams
    n_words = len(words)
    candidates = set(words)
    for i in range(n_words - 1):
        candidates.add(f"{words[i]} {words[i+1]}")

    for candidate in candidates:
        if candidate in weights:
            weight = weights[candidate]
            matched_features.append({
                'token': candidate,
                'weight': round(float(weight), 4),
                'indicator': 'fake' if weight > 0 else 'real',
                'magnitude': abs(weight)
            })

    # Sort by magnitude of influence
    matched_features.sort(key=lambda x: x['magnitude'], reverse=True)
    return matched_features[:top_n]


# ---------------------------------------------------------------------------
# Security headers (S06): diterapkan pada semua respons.
# ---------------------------------------------------------------------------
_CSP = (
    "default-src 'self'; "
    # Inline JS/CSS diperlukan oleh template (diperbaiki ke nonce di versi mendatang).
    "script-src 'self' 'unsafe-inline' https://www.google.com/recaptcha/ "
    "https://www.gstatic.com/recaptcha/; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-src 'self' https://www.google.com/recaptcha/ "
    "https://www.gstatic.com/recaptcha/; "
    "object-src 'none'; "
    "base-uri 'self';"
)


@app.after_request
def _set_security_headers(response):
    """Setel header keamanan pada setiap respons HTTP.

    - Content-Security-Policy: membatasi sumber skrip/gaya/font. Inline
      script/diizinkan hanya melalui nonce atau hash; Google Fonts diizinkan
      karena templates/index.html memakainya.
    - X-Content-Type-Options: nosniff mencegah MIME sniffing.
    - X-Frame-Options: DENY mencegah clickjacking via iframe.
    - Referrer-Policy: no-referrer tidak mengirim URL halaman ke situs luar.
    - Permissions-Policy: minimal, nonaktifkan fitur browser yang tidak perlu.
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Permissions-Policy'] = (
        'geolocation=(), microphone=(), camera=(), '
        'payment=(), usb=(), magnetometer=()'
    )
    # CSP hanya untuk respons HTML (jangan setel pada JSON API yang tidak
    # perlu membebankan browser dengan kebijakan konten).
    if 'text/html' in response.headers.get('Content-Type', ''):
        response.headers['Content-Security-Policy'] = _CSP
    return response


@app.route('/')
def home():
    return render_template('index.html', metrics=eval_metrics, model_reliable=MODEL_RELIABLE)


@app.route('/metrics', methods=['GET'])
@limiter.limit(current_rate_limit)
def get_metrics():
    if eval_metrics:
        return jsonify(eval_metrics)
    return jsonify({'status': 'no_metrics_found'}), 404


@app.route('/predict', methods=['POST'])
@limiter.limit(current_rate_limit)
def predict():
    if model is None or tfidf is None:
        # 503 = service unavailable (bukan server error): artefak memang belum
        # dibuat/dimuat. Berikan pesan pemulihan yang dapat ditindaklanjuti.
        return jsonify({
            'error': MODEL_UNAVAILABLE_MESSAGE,
            'models_missing': missing_models(primary_unavailable=True),
            'recovery': {
                'en': 'python train_model.py',
                'id': 'python train_model_id.py',
                'after': 'Restart server, lalu cek GET /health.',
            },
        }), 503

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Parameter "text" wajib diisi.'}), 400

    # Validate input type - must be a string
    raw_text = data.get('text')
    if not isinstance(raw_text, str):
        return jsonify({
            'error': 'Parameter "text" harus berupa teks (must be a string).'
        }), 400

    # Validate input type - must be a string
    raw_text = data.get('text')
    if not isinstance(raw_text, str):
        return jsonify({'error': 'Parameter "text" must be a string.'}), 400

    raw_text = raw_text.strip()
    # Normalize Unicode (NFKC): convert full-width chars to standard, remove compatibility
    raw_text = unicodedata.normalize('NFKC', raw_text)
    if len(raw_text) < 10:
        return jsonify({'error': 'Teks terlalu pendek. Masukkan minimal 10 karakter.'}), 400

    if len(raw_text) > MAX_TEXT_CHARS:
        return jsonify({
            'error': f'Teks terlalu panjang. Maksimal {MAX_TEXT_CHARS:,} karakter.'
        }), 400

    # Detect language using lexicon-based approach
    lang_result = detect_language(raw_text)
    lang_code = lang_result['code']
    lang_name = lang_result['name']
    supported = lang_result['supported']

    # Clean text using Indonesian preprocessing (consistent with training)
    cleaned = id_clean_text(raw_text)
    if not cleaned:
        return jsonify({'error': 'Teks hanya berisi simbol atau angka. Masukkan teks berita yang valid.'}), 400

    # E: teks terlalu pendek (< 5 kata) -> jangan jalankan model;
    # kembalikan status teks_terlalu_pendek. Bahasa yang tidak terdeteksi
    # dengan yakin tetap dianalisis dengan model fallback (perilaku yang sudah
    # ada sebelumnya) agar kompatibel dengan pemanggil lama.
    word_count = len(raw_text.split())
    short_text = (word_count < 5)

    # Select per-language artefacts: Indonesian text uses the Indonesian model,
    # everything else (English, regional, unknown) falls back to the English
    # primary model.  This keeps predictions in-domain while degrading
    # gracefully when the Indonesian artefacts are missing.
    model_used = 'en'
    active_model = model_en
    active_tfidf = tfidf_en
    active_weights = feature_weights_en
    active_threshold = DECISION_THRESHOLD_EN
    use_indobert = False
    if lang_code == 'id' and model_id is not None and tfidf_id is not None:
        model_used = 'id'
        active_model = model_id
        active_tfidf = tfidf_id
        active_weights = feature_weights_id
        active_threshold = DECISION_THRESHOLD_ID
    # TAHAP C: IndoBERT mengesampingkan model TF-IDF lama untuk teks Indonesia
    # bila tersedia (folder models/indobert/ + dependensi terpasang).
    if lang_code == 'id' and INDOBERT_MODEL is not None and INDOBERT_TOKENIZER is not None:
        model_used = 'indobert'
        use_indobert = True
        active_threshold = INDOBERT_THRESHOLD

    # Vectorize (hanya untuk model TF-IDF; IndoBERT mendapat teks mentah)
    # E: jika teks terlalu pendek / bahasa tidak yakin, jangan jalankan model.
    if not short_text:
        if use_indobert:
            vec = None
        else:
            vec = active_tfidf.transform([cleaned])

        # Probabilities (calibrated)
        if use_indobert:
            prob_fake = indobert_predict_proba(cleaned)
            if prob_fake is None:
                # Fallback ke model TF-IDF lama saat IndoBERT gagal inference.
                use_indobert = False
                model_used = 'id'
                active_threshold = DECISION_THRESHOLD_ID
                vec = active_tfidf.transform([cleaned])
                proba = active_model.predict_proba(vec)[0]
                prob_real = float(proba[0]) * 100
                prob_fake = float(proba[1]) * 100
            else:
                prob_fake = prob_fake * 100
                prob_real = 100.0 - prob_fake
        elif hasattr(active_model, 'predict_proba'):
            proba = active_model.predict_proba(vec)[0]
            prob_real = float(proba[0]) * 100
            prob_fake = float(proba[1]) * 100
        else:
            prob_real, prob_fake = 0.0, 100.0
    else:
        vec = None
        prob_real = None
        prob_fake = None

    # Decision uses the tuned threshold (optimized on validation split)
    # E: teks terlalu pendek / bahasa tidak yakin -> jangan buat keputusan model.
    if short_text:
        pred = None
        label = 'Tidak dapat dianalisis'
        confidence = None
    else:
        pred = 1 if (prob_fake / 100.0) >= active_threshold else 0
        label = 'Fake' if pred == 1 else 'Real'
        confidence = prob_fake if pred == 1 else prob_real

    # Explainability (XAI) - get top influential features.  Tidak tersedia
    # untuk model IndoBERT (basis transformer), maka dikosongkan dengan catatan.
    if use_indobert:
        explanation = []
        explanation_note = (
            'Penjelasan kata kunci (XAI) tidak tersedia untuk model IndoBERT. '
            'Gunakan pencocokan klaim yang sudah dibantah sebagai referensi tambahan.'
        )
    else:
        explanation = explain_prediction(cleaned, weights=active_weights)
        explanation_note = None
        if not explanation:
            explanation_note = (
                'Tidak ada kata kunci yang dikenali pada teks ini. '
                'Coba gunakan kalimat berita yang lebih lengkap.'
            )

    # Language-aware handling.  The primary model artifact (models/model.pkl)
    # is trained primarily on English news; key-influence weights come from its
    # vocabulary.  Regional-language / unknown inputs are flagged so callers can
    # show a user-visible warning instead of trusting the score blindly.
    regional_warning = None
    if not supported:
        regional_warning = {
            'code': lang_code,
            'name': lang_name,
            'message': (
                f'Prediction for {lang_name} may be less accurate. '
                f'The primary model is trained mainly on English news, and key '
                f'indicators are drawn from that vocabulary.'
            ),
        }

    # Honest three-zone reporting plus rule-based warning signs.  The status
    # is derived from probability_fake as a fraction, independent of the
    # model's binary decision threshold above.
    if short_text:
        status = 'teks_terlalu_pendek'
    else:
        status = determine_status(prob_fake / 100.0)
    red_flags = detect_red_flags(raw_text)
    language_warning = bool(is_indonesian(raw_text) and model_used == 'en')
    credibility = detect_credibility_signals(raw_text)

    # TAHAP B: cari klaim hoak yang sudah dibantah dan mirip dengan teks.
    # Hanya untuk teks Indonesia (indeks berisi judul hoak Indonesia); untuk
    # bahasa lain dikembalikan list kosong agar tidak memuat model embeddings
    # secara tidak perlu.
    similar_debunked = []
    # Muat indeks debunk (lazy) sehingga DEBUNK_AVAILABLE terupdate sebelum
    # kita memutuskan apakah fitur pencocokan tersedia untuk request ini.
    _load_debunk_index()
    debunk_available = bool(DEBUNK_AVAILABLE)
    if lang_code == 'id' and debunk_available:
        try:
            similar_debunked = find_similar_debunked(raw_text, top_k=DEBUNK_TOP_K)
        except Exception:
            similar_debunked = []

    if short_text:
        advice = 'Teks terlalu pendek untuk dianalisis. Tempelkan paragraf atau berita yang lebih lengkap.'
    else:
        advice = build_advice(status, prob_real, prob_fake, len(red_flags), len(credibility))

    return jsonify({
        'prediction': label,
        'label_code': pred,
        'confidence': None if confidence is None else round(confidence, 2),
        'probability_real': None if prob_real is None else round(prob_real, 2),
        'probability_fake': None if prob_fake is None else round(prob_fake, 2),
        'decision_threshold': round(active_threshold, 4),
        'model_used': model_used,
        'snippet': raw_text[:140] + ('...' if len(raw_text) > 140 else ''),
        'character_count': len(raw_text),
        'word_count': len(raw_text.split()),
        'key_influences': explanation,
        'key_influences_note': explanation_note,
        'language_detection': {
            'code': lang_code,
            'name': lang_name,
            'supported': supported,
            'warning': regional_warning['message'] if regional_warning else None
        },
        # New, additive fields (old field names above are unchanged).
        'status': status,
        'red_flags': red_flags,
        'language_warning': language_warning,
        'advice': advice,
        # TAHAP B: hoak yang sudah dibantah dan mirip dengan teks input.
        'similar_debunked': similar_debunked,
        'debunk_available': bool(debunk_available),
        # Honest reliability flag: False when hold-out accuracy < 85%.
        'model_reliable': bool(MODEL_RELIABLE),
    })


@app.route('/feedback', methods=['POST'])
@limiter.limit(current_rate_limit)
def feedback():
    """Catat label manusia untuk pelatihan berkelanjutan (status "baru").

    Format berkas: id, timestamp, text_hash, text, label, status, reviewer.
    Semua masukan baru diberi status "baru" dan hanya baris dengan status
    "disetujui" yang dibaca oleh pipeline pelatihan.
    """
    if not FEEDBACK_ENABLED:
        return jsonify({
            'error': 'Fitur feedback saat ini dinonaktifkan oleh administrator.'
        }), 503

    data = request.get_json(silent=True)
    if not data or 'text' not in data or 'label' not in data:
        return jsonify({'error': 'Parameter "text" dan "label" wajib diisi.'}), 400

    label = parse_feedback_label(data['label'])
    if label is None:
        return jsonify({'error': 'Label harus 0 (Real) atau 1 (Palsu).'}), 400

    text = str(data['text']).strip()
    text = unicodedata.normalize('NFKC', text)

    if len(text) < 10:
        return jsonify({'error': 'Teks terlalu pendek. Minimal 10 karakter.'}), 400

    if len(text) > FEEDBACK_MAX_CHARS:
        return jsonify({
            'error': f'Teks terlalu panjang. Maksimal {FEEDBACK_MAX_CHARS:,} karakter.'
        }), 400

    text_hash = hashlib.sha256((text + str(label)).encode('utf-8')).hexdigest()
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Deduplikasi berdasarkan hash teks+label ---------------------------
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('text_hash') == text_hash:
                        return jsonify({
                            'status': 'duplicate',
                            'message': 'Feedback duplikat telah dicatat sebelumnya.',
                            'total_feedback_samples': _count_feedback(),
                        }), 200
        except Exception:
            # Berkas rusak: lanjutkan tanpa dedup (baris baru tetap ditambahkan)
            pass

    # --- Batas ukuran berkas ----------------------------------------------
    if os.path.exists(FEEDBACK_PATH) and os.path.getsize(FEEDBACK_PATH) >= FEEDBACK_MAX_FILE_SIZE:
        return jsonify({
            'error': 'Berkas feedback penuh (maksimal 5 MB). Hubungi administrator.'
        }), 503

    # --- Tulis dengan file lock + append atomik ---------------------------
    need_header = not (os.path.exists(FEEDBACK_PATH)
                       and os.path.getsize(FEEDBACK_PATH) > 0)
    try:
        with open(FEEDBACK_PATH, 'a', newline='', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=_FEEDBACK_FIELDS)
                if need_header:
                    writer.writeheader()
                writer.writerow({
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'text_hash': text_hash,
                    'text': _sanitize_feedback_cell(text),
                    'label': label,
                    'status': 'baru',
                    'reviewer': '',
                })
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        # Jangan bocorkan detail exception ke klien
        return jsonify({'error': 'Gagal menyimpan feedback.'}), 500

    return jsonify({
        'status': 'saved',
        'message': ('Feedback dicatat dengan status "baru". '
                    'Akan diverifikasi sebelum digunakan pelatihan.'),
        'total_feedback_samples': _count_feedback(),
    })


@app.route('/health', methods=['GET'])
def health():
    feedback_count = 0
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, 'r', encoding='utf-8') as f:
                feedback_count = max(sum(1 for _ in f) - 1, 0)
        except Exception:
            feedback_count = 0

    missing = missing_models()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': tfidf is not None,
        'feature_weights_loaded': len(feature_weights) > 0,
        'decision_threshold': DECISION_THRESHOLD,
        # Model mana yang tidak tersedia (list kosong = semua siap).
        'models_missing': missing,
        'model_recovery': MODEL_UNAVAILABLE_MESSAGE if missing else None,
        'model_reliability': {
            'reliable': bool(RELIABILITY['reliable']),
            'reasons': list(RELIABILITY['reasons']),
            'result_path': RELIABILITY['result_path'],
            'rules': dict(RELIABILITY['rules']),
        },
        'indobert': {
            'checksums_ok': _indobert_checksums_ok()[0],
            'checksum_reason': _indobert_checksums_ok()[1],
            'loaded': INDOBERT_MODEL is not None,
        },
        'models': {
            'en': {
                'loaded': model_en is not None,
                'vectorizer_loaded': tfidf_en is not None,
                'feature_weights_loaded': len(feature_weights_en) > 0,
                'decision_threshold': DECISION_THRESHOLD_EN,
                'missing': 'en' in missing,
            },
            'id': {
                'loaded': model_id is not None,
                'vectorizer_loaded': tfidf_id is not None,
                'feature_weights_loaded': len(feature_weights_id) > 0,
                'decision_threshold': DECISION_THRESHOLD_ID,
                'missing': 'id' in missing,
            },
        },
        'feedback_samples_pending_retrain': feedback_count
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')
    print(f"Starting server on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
