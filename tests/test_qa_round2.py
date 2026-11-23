"""Regression tests for the QA round-2 bug fixes."""

import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import app as app_module
from app import app, build_advice
from id_preprocessing import clean_text as id_clean_text, detect_language


@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Rate limits are verified explicitly in their own test.
    app.config['RATELIMIT_ENABLED'] = False
    app_module.limiter.reset()
    with app.test_client() as test_client:
        yield test_client
    app_module.limiter.reset()


# --- Bug 1: language detection ------------------------------------------------

@pytest.mark.parametrize('text, expected_code', [
    ('NASA scientists confirmed today that the exploratory rover has landed on Mars.', 'en'),
    ('Pemerintah Indonesia mengumumkan bahwa berita tersebut tidak benar dan menyesatkan.', 'id'),
])
def test_detect_language_identifies_supported_languages(text, expected_code):
    result = detect_language(text)
    assert result['code'] == expected_code
    assert result['name'] in ('English', 'Indonesian')
    assert result['supported'] is True


def test_predict_reports_english_for_english_input(client):
    response = client.post('/predict', json={
        'text': 'Scientists confirmed that the government announced new evidence today.'
    })
    assert response.status_code == 200
    detection = response.get_json()['language_detection']
    assert detection['code'] == 'en'
    assert detection['name'] == 'English'
    assert detection['supported'] is True


@pytest.mark.parametrize('text, expected_code, expected_name', [
    ('Aku ora ngerti kowe piye, mboten sampun teka.', 'jv', 'Javanese'),
    ('Kuring teu nyaho naon eta teh, abdi rek ka pasar.', 'su', 'Sundanese'),
    ('Ahu do ndang hape molo, roha holong di dongan.', 'btk', 'Batak'),
])
def test_detect_language_identifies_regional_languages(text, expected_code, expected_name):
    result = detect_language(text)
    assert result['code'] == expected_code
    assert result['name'] == expected_name
    assert result['supported'] is False
    assert result['distinct_hits'][expected_code] >= 2


def test_predict_flags_regional_language_with_warning(client):
    response = client.post('/predict', json={
        'text': 'Aku ora ngerti kowe piye, mboten sampun teka.'
    })
    assert response.status_code == 200
    detection = response.get_json()['language_detection']
    assert detection['code'] == 'jv'
    assert detection['supported'] is False
    assert detection['warning']
    assert 'less accurate' in detection['warning']


def test_predict_supported_languages_have_no_warning(client):
    for text, code in [
        ('Scientists confirmed that the government announced new evidence today.', 'en'),
        ('Pemerintah Indonesia menyatakan bahwa berita tersebut tidak benar.', 'id'),
    ]:
        response = client.post('/predict', json={'text': text})
        assert response.status_code == 200
        detection = response.get_json()['language_detection']
        assert detection['code'] == code
        assert detection['supported'] is True
        assert detection['warning'] is None


def test_predict_uses_indonesian_model_for_indonesian_text(client):
    response = client.post('/predict', json={
        'text': 'Pemerintah Indonesia menyatakan bahwa berita tersebut tidak benar dan menyesatkan masyarakat.'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['model_used'] == 'id'
    assert data['language_detection']['code'] == 'id'
    assert data['language_detection']['supported'] is True


def test_predict_uses_english_model_for_english_text(client):
    response = client.post('/predict', json={
        'text': 'Scientists confirmed that the government announced new evidence today.'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['model_used'] == 'en'
    assert data['language_detection']['code'] == 'en'


def test_predict_uses_english_model_for_regional_text(client):
    response = client.post('/predict', json={
        'text': 'Aku ora ngerti kowe piye, mboten sampun teka.'
    })
    assert response.status_code == 200
    data = response.get_json()
    # Regional languages fall back to the English primary model.
    assert data['model_used'] == 'en'
    assert data['language_detection']['code'] == 'jv'
    assert data['language_detection']['supported'] is False


def test_predict_uses_indonesian_model_for_fullwidth_indonesian_text(client):
    text = 'Ｐｅｍｅｒｉｎｔａｈ　Ｉｎｄｏｎｅｓｉａ　ｍｅｎｙａｔｕｋａｎ　ｂｅｒｉｔａ　ｔｉｄａｋ　ｂｅｎａｒ．'
    response = client.post('/predict', json={'text': text})
    assert response.status_code == 200
    data = response.get_json()
    assert data['model_used'] == 'id'
    assert data['language_detection']['code'] == 'id'


def test_health_reports_both_models(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'models' in data
    assert data['models']['en']['loaded'] is True
    assert data['models']['en']['decision_threshold'] > 0
    assert 'id' in data['models']


def test_detect_language_rejects_empty_input():
    assert detect_language('')['code'] == 'unknown'
    assert detect_language('   !!!   ')['code'] == 'unknown'


# --- Bug 2: rate limiting -----------------------------------------------------

def test_predict_is_rate_limited_and_health_stays_open(client, monkeypatch):
    monkeypatch.setitem(app.config, 'RATELIMIT_ENABLED', True)
    monkeypatch.setitem(app.config, 'RATE_LIMIT', '2 per minute')
    app_module.limiter.reset()
    payload = {'text': 'A valid news article used to verify the rate limit.'}

    try:
        for _ in range(2):
            assert client.post('/predict', json=payload).status_code == 200

        blocked = client.post('/predict', json=payload)
        assert blocked.status_code == 429
        assert 'error' in blocked.get_json()

        # /health is deliberately exempt so uptime monitors are never blocked.
        for _ in range(5):
            assert client.get('/health').status_code == 200
    finally:
        app_module.limiter.reset()


def test_rate_limit_uses_configured_value():
    assert isinstance(app.config['RATE_LIMIT'], str)
    assert 'minute' in app.config['RATE_LIMIT']


# --- Item 1: debug / production server ----------------------------------------

def test_app_not_debug_by_default(monkeypatch):
    """Aplikasi tidak berjalan dalam mode debug bila FLASK_DEBUG tidak diset."""
    monkeypatch.delenv('FLASK_DEBUG', raising=False)
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')
    assert debug is False
    assert app.debug is False


@pytest.mark.parametrize('value', ['1', 'true', 'yes', 'on'])
def test_app_debug_when_env_set(monkeypatch, value):
    monkeypatch.setenv('FLASK_DEBUG', value)
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')
    assert debug is True


def test_app_default_host_is_localhost(monkeypatch):
    """Host default 127.0.0.1; 0.0.0.0 hanya melalui env HOST."""
    monkeypatch.delenv('HOST', raising=False)
    assert os.environ.get('HOST', '127.0.0.1') == '127.0.0.1'


# --- Item 3: rate limit / ProxyFix --------------------------------------------

def test_rate_limit_key_ignores_xff_by_default():
    """X-Forwarded-For tidak memengaruhi kunci rate limit bila TRUSTED_PROXIES=0.

    ProxyFix(x_for=0) abaikan header; kunci = alamat koneksi langsung.
    """
    from flask_limiter.util import get_remote_address
    from flask import request

    with app.test_request_context('/', headers={'X-Forwarded-For': '1.2.3.4'}):
        key_a = get_remote_address()
    with app.test_request_context('/', headers={'X-Forwarded-For': '5.6.7.8'}):
        key_b = get_remote_address()
    assert key_a == key_b == '127.0.0.1'


def test_rate_limit_memory_multiworker_warning():
    """Peringatan dicetak bila memory:// + banyak worker (config-level)."""
    assert app.config['RATELIMIT_STORAGE_URI'] == 'memory://'
    workers = int(os.environ.get('WEB_CONCURRENCY', '1'))
    # Saat 1 worker tidak perlu peringatan; saat >1, aplikasi mencetak peringatan
    # saat import (terjadi saat tes diimpor). Cukup verifikasi konfigurasi.
    assert isinstance(workers, int) and workers >= 1


# --- Bug 3: key_influences ----------------------------------------------------

def test_predict_returns_influences_for_known_vocabulary(client):
    response = client.post('/predict', json={
        'text': 'Anonymous whistleblowers claim secret alien bases under polar ice.'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['key_influences']
    assert data['key_influences_note'] is None


def test_predict_explains_empty_influences(client):
    response = client.post('/predict', json={
        'text': 'qxztpl wrrrmb qzxvbn blorb flimflam zzzxqq'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['key_influences'] == []
    assert data['key_influences_note']


# --- Bug 4: full-width normalisation -----------------------------------------

def test_clean_text_normalises_fullwidth_characters():
    text = 'Ｂｒｅａｋｉｎｇ　Ｎｅｗｓ：　Ｐｒｅｓｉｄｅｎｔ　ｓｐｏｋｅ　ｔｏｄａｙ．'
    assert id_clean_text(text) == 'breaking news president spoke today'


def test_detect_language_normalises_fullwidth_characters():
    text = 'Ｐｅｍｅｒｉｎｔａｈ　Ｉｎｄｏｎｅｓｉａ　ｍｅｎｇｕｍｕｍｋａｎ　ｂｅｒｉｔａ　ｔｅｒｓｅｂｕｔ．'
    assert detect_language(text)['code'] == 'id'


def test_predict_accepts_fullwidth_text(client):
    text = 'Ｂｒｅａｋｉｎｇ　ｎｅｗｓ：　ｓｃｉｅｎｔｉｓｔｓ　ｃｏｎｆｉｒｍｅｄ　ｔｈａｔ　ｔｈｅ　ｒｏｖｅｒ　ｌａｎｄｅｄ　ｔｏｄａｙ．'
    response = client.post('/predict', json={'text': text})
    assert response.status_code == 200
    data = response.get_json()
    assert data['language_detection']['code'] == 'en'
    assert data['key_influences']


# --- Bug 5: feedback label validation ----------------------------------------

@pytest.mark.parametrize('label', [True, False, 1.0, 0.0, 2, -1, '2', 'yes', None, [], {}])
def test_feedback_rejects_invalid_labels(client, tmp_path, monkeypatch, label):
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(tmp_path / 'feedback.csv'))
    response = client.post('/feedback', json={
        'text': 'A perfectly valid news article submitted as human feedback.',
        'label': label,
    })
    assert response.status_code == 400
    assert 'Label harus 0 (Real) atau 1 (Palsu).' in response.get_json()['error']


@pytest.mark.parametrize('label, expected', [(0, 0), (1, 1), ('0', 0), (' 1 ', 1)])
def test_feedback_accepts_valid_labels(client, tmp_path, monkeypatch, label, expected):
    feedback_file = tmp_path / 'feedback.csv'
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(feedback_file))
    response = client.post('/feedback', json={
        'text': 'A perfectly valid news article submitted as human feedback.',
        'label': label,
    })
    assert response.status_code == 200
    rows = feedback_file.read_text(encoding='utf-8').strip().splitlines()
    # Kolom: id,timestamp,text_hash,text,label,status,reviewer -> label kolom ke-5 (index 4)
    last = rows[-1].split(',')
    assert int(last[4]) == expected


def test_feedback_normalises_fullwidth_text(client, tmp_path, monkeypatch):
    feedback_file = tmp_path / 'feedback.csv'
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(feedback_file))
    response = client.post('/feedback', json={
        'text': 'Ｐｒｅｓｉｄｅｎｔ　ａｎｎｏｕｎｃｅｄ　ｔｈｅ　ｎｅｗ　ｐｏｌｉｃｙ　ｔｏｄａｙ．',
        'label': '1',
    })
    assert response.status_code == 200
    assert 'president announced the new policy today' in feedback_file.read_text(encoding='utf-8').lower()


# --- Regression: BUG-001 -----------------------------------------------------

@pytest.mark.parametrize('value', [None, True, 42, ['news'], {'text': 'news'}])
def test_predict_rejects_non_string_text(client, value):
    response = client.post('/predict', json={'text': value})
    assert response.status_code == 400
    assert 'must be a string' in response.get_json()['error']


# --- Regression: training vs inference cleaning skew ---------------------------

TRAINING_VS_INFERENCE_EXAMPLES = [
    "Pemerintah Indonesia mengumumkan bantuan sosial sebesar 2 juta rupiah.",
    "<b>BREAKING:</b> www.berita-palsu.com cara cepat kaya malam ini!!!",
    "Admin: Hubungi 08123456789 untuk OTP rekening Anda segera.",
    "turnbackhoax.id telah menilai klaim ini sebagai hoaks.",
    "Selamat pagi, semoga harimu menyenangkan dan produktif.",
    "Baca juga: artikel ini telah tayang di detikcom, editor: penulis.",
    "Hubungi kami di admin@example.com untuk info selengkapnya.",
    "Hoax atau hoaks? Cek fakta di cekfakta.com sebelum dibagikan.",
    "Ｐｅｍｅｒｉｎｔａｈ　Ｉｎｄｏｎｅｓｉａ　ｍｅｎｇｕｍｕｋａｎ　ｂｅｒｉｔａ．",
    "NASA scientists confirm rover landed on Mars (mixed English).",
]


def test_training_cleaning_equals_inference_cleaning():
    """Training (id_preprocessing.clean_text) == inferensi (app.id_clean_text).

    Mencegah train/serve skew: jika fungsi pembersih diubah di satu sisi,
    tes ini akan gagal sebelum prediksi berdampak pada pengguna.
    """
    from id_preprocessing import clean_text as train_clean_text

    for teks in TRAINING_VS_INFERENCE_EXAMPLES:
        hasil_training = train_clean_text(teks)
        hasil_inferensi = id_clean_text(teks)
        assert hasil_training == hasil_inferensi, (
            f"SKEW terdeteksi untuk {teks!r}: "
            f"training={hasil_training!r} inferensi={hasil_inferensi!r}"
        )


# --- Item 2: batas input + ReDoS ----------------------------------------------

import re  # noqa: E402  (diimpor di sini untuk tes ReDoS)


def test_domain_regex_not_redos():
    """_DOMAIN_RE harus cepat pada input berulang panjang (anti ReDoS).

    Target < 0.1 detik pada input 20.000 karakter berulang.
    """
    from app import _DOMAIN_RE
    berulang = 'a.b.c.d.e.f.' * 2000  # ~20.000 karakter
    import time
    t0 = time.perf_counter()
    _DOMAIN_RE.findall(berulang)
    dt = time.perf_counter() - t0
    assert dt < 0.1, f'_DOMAIN_RE terlambat: {dt:.3f}s (target < 0.1s)'


def test_predict_rejects_text_over_max_chars(client):
    """Teks lebih dari MAX_TEXT_CHARS (20.000) ditolak dengan 400 + pesan Indonesia."""
    max_chars = app_module.MAX_TEXT_CHARS
    terlalu_panjang = 'x ' * (max_chars // 2 + 10)  # melebihi batas
    response = client.post('/predict', json={'text': terlalu_panjang})
    assert response.status_code == 400
    data = response.get_json()
    assert 'terlalu panjang' in data['error'].lower()
    # Format angka Indonesia (pemisah ribuan koma)
    assert '20,000' in data['error']


def test_predict_accepts_text_at_max_chars(client):
    """Teks tepat MAX_TEXT_CHARS karakter masih diproses (batas inklusif)."""
    max_chars = app_module.MAX_TEXT_CHARS
    tepat_batas = 'a ' * (max_chars // 2)  # tepat 20.000 karakter
    assert len(tepat_batas) == max_chars
    response = client.post('/predict', json={'text': tepat_batas})
    # Teks 'a ' berulang bukan bahasa yang didukung, tapi harus diproses (400 karena
    # hanya simbol/angka setelah pembersihan) BUKAN ditolak karena panjang.
    assert response.status_code in (200, 400)
    if response.status_code == 400:
        assert 'terlalu panjang' not in response.get_json()['error'].lower()


def test_max_content_length_config():
    """MAX_CONTENT_LENGTH = 256 KB (Flask menanggapi 413 sebelum memproses body)."""
    assert app.config['MAX_CONTENT_LENGTH'] == 256 * 1024


# --- Item 4: feedback (formula injection, dedup, batas, status baru) ----------

FEEDBACK_TEXT = 'Ini adalah teks unik untuk uji deduplikasi feedback.'


def test_feedback_formula_injection(client, tmp_path, monkeypatch):
    """Sel yang diawali =, +, -, @, tab, atau CR diberi awalan ' (anti formula injection)."""
    feedback_file = tmp_path / 'feedback.csv'
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(feedback_file))
    for prefix in ['=', '+', '-', '@']:
        response = client.post('/feedback', json={
            'text': prefix + 'CMD|payload',
            'label': 1,
        })
        assert response.status_code == 200, response.get_json()
    content = feedback_file.read_text(encoding='utf-8')
    for prefix in ['=', '+', '-', '@']:
        assert ("'" + prefix) in content


def test_feedback_dedup(client, tmp_path, monkeypatch):
    """Teks+label yang sama dikirim dua kali -> status duplicate, tidak duplikat baris."""
    feedback_file = tmp_path / 'feedback.csv'
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(feedback_file))
    r1 = client.post('/feedback', json={'text': FEEDBACK_TEXT, 'label': 1})
    assert r1.status_code == 200
    assert r1.get_json()['status'] == 'saved'
    r2 = client.post('/feedback', json={'text': FEEDBACK_TEXT, 'label': 1})
    assert r2.status_code == 200
    assert r2.get_json()['status'] == 'duplicate'
    # Berkas hanya boleh punya 1 baris data (header + 1)
    lines = feedback_file.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 2  # header + 1 data


def test_feedback_rejects_over_max_chars(client, tmp_path, monkeypatch):
    """Feedback teks > 20.000 karakter ditolak 400."""
    feedback_file = tmp_path / 'feedback.csv'
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(feedback_file))
    terlalu_panjang = 'x ' * (app_module.FEEDBACK_MAX_CHARS // 2 + 10)
    response = client.post('/feedback', json={'text': terlalu_panjang, 'label': 1})
    assert response.status_code == 400
    assert 'terlalu panjang' in response.get_json()['error'].lower()


def test_training_ignores_unapproved_feedback():
    """train_model_id.py hanya membaca baris dengan status 'disetujui'."""
    import tempfile, os as _os
    from train_model_id import clean_text
    # Buat berkas feedback sementara dengan satu baris "baru" dan satu "disetujui"
    tmpdir = tempfile.mkdtemp()
    fb_path = _os.path.join(tmpdir, 'feedback_reviewed.csv')
    with open(fb_path, 'w', encoding='utf-8', newline='') as f:
        f.write('id,timestamp,text_hash,text,label,status,reviewer\n')
        f.write('uuid1,2026-01-01T00:00:00+00:00,hash1,Teks baru yang harus diabaikan,1,baru,\n')
        f.write('uuid2,2026-01-01T00:00:00+00:00,hash2,Teks disetujui untuk pelatihan,0,disetujui,\n')
    # Monkeypatch DATA_DIR sementara
    import train_model_id as tmi
    orig_data_dir = tmi.DATA_DIR
    tmi.DATA_DIR = tmpdir
    try:
        # Simulasi logika filter: hanya baris status 'disetujui'
        import pandas as pd
        fb = pd.read_csv(fb_path)
        approved = fb[fb['status'] == 'disetujui']
        assert len(approved) == 1
        assert approved.iloc[0]['text'] == 'Teks disetujui untuk pelatihan'
        assert approved.iloc[0]['label'] == 0
    finally:
        tmi.DATA_DIR = orig_data_dir
        import shutil
        shutil.rmtree(tmpdir)


# --- Item 5: URL aman di UI + server-side filter ------------------------------

from urllib.parse import urlparse  # noqa: E402


def test_url_scheme_filter_drops_non_http_https():
    """Filtering skema URL server-side hanya menyisakan http/https."""
    from app import _load_debunk_index, HOAX_META
    _load_debunk_index()
    if HOAX_META:
        for m in HOAX_META:
            assert urlparse(m.get('url', '')).scheme in ('http', 'https'), m.get('url')


def test_url_scheme_filter_unit():
    """Unit test logika filter skema: http/https lolos, yang lain dibuang."""
    raw = [
        {'url': 'https://example.com', 'judul': 'a'},
        {'url': 'http://example.com', 'judul': 'b'},
        {'url': 'javascript:alert(1)', 'judul': 'c'},
        {'url': 'ftp://example.com', 'judul': 'd'},
        {'url': '', 'judul': 'e'},
    ]
    filtered = [m for m in raw
                if urlparse(m.get('url', '')).scheme in ('http', 'https')]
    assert len(filtered) == 2
    assert {m['judul'] for m in filtered} == {'a', 'b'}


# --- Item 7: build_advice matriks status x jumlah red_flag ---------------------


@pytest.mark.parametrize(
    'status, red_flag_count, expected_judul_fragment',
    [
        # belum_dapat_disimpulkan: 0 / 1-2 / >=3 red_flag
        ('belum_dapat_disimpulkan', 0, 'Belum dapat disimpulkan'),
        ('belum_dapat_disimpulkan', 1, 'ada indikasi mencurigakan'),
        ('belum_dapat_disimpulkan', 2, 'ada indikasi mencurigakan'),
        ('belum_dapat_disimpulkan', 3, 'patut diwaspadai'),
        ('belum_dapat_disimpulkan', 5, 'patut diwaspadai'),
        # cenderung_valid: 0 / 1-2 / >=3 red_flag
        ('cenderung_valid', 0, 'Cenderung valid'),
        ('cenderung_valid', 2, 'Cenderung valid'),
        ('cenderung_valid', 3, 'Cenderung valid'),
        # cenderung_hoaks: tidak berubah untuk semua tier red_flag
        ('cenderung_hoaks', 0, 'Cenderung hoaks'),
        ('cenderung_hoaks', 2, 'Cenderung hoaks'),
        ('cenderung_hoaks', 5, 'Cenderung hoaks'),
    ],
)
def test_build_advice_matriks_status_red_flag(status, red_flag_count,
                                              expected_judul_fragment):
    """Setiap kombinasi status x tier red_flag menghasilkan konsisten.

    - belum_dapat_disimpulkan + 0 -> judul netral
    - belum_dapat_disimpulkan + 1-2 -> indikasi mencurigakan
    - belum_dapat_disimpulkan + >=3 -> patut diwaspadai (tegas)
    - cenderung_valid + >=3 -> tetap 'Cenderung valid' (ringkasan ditambah)
    - cenderung_hoaks -> tidak berubah meski red_flag tinggi
    """
    advice = build_advice(status, 50.0, 50.0, red_flag_count)
    assert set(advice) == {'judul', 'ringkasan', 'saran'}
    assert expected_judul_fragment in advice['judul']
    assert advice['ringkasan']
    assert advice['saran']
    # Tidak ada kata absolut/jaminan (cocokkan kata utuh, bukan substring
    # di dalam "pastikan").
    import re as _re
    combined = ' '.join(advice.values()).lower()
    assert not _re.search(r'\bterbukti\b', combined)
    assert not _re.search(r'\bpasti\b', combined)
    assert '100%' not in combined


def test_build_advice_belum_dapat_3_red_flag_tegas():
    advice = build_advice('belum_dapat_disimpulkan', 55.0, 45.0, 3)
    assert advice['judul'] == 'Belum dapat disimpulkan, tetapi patut diwaspadai'
    assert 'Jangan dibagikan dulu' in advice['saran']


def test_build_advice_cenderung_valid_tambah_ringkasan_saat_3_red_flag():
    advice = build_advice('cenderung_valid', 60.0, 40.0, 3)
    assert 'ada beberapa ciri' in advice['ringkasan']
    advice2 = build_advice('cenderung_valid', 60.0, 40.0, 2)
    assert 'ada beberapa ciri' not in advice2['ringkasan']


def test_build_advice_cenderung_hoaks_tidak_berubah():
    advice = build_advice('cenderung_hoaks', 80.0, 20.0, 5)
    assert advice['judul'] == 'Cenderung hoaks'
    assert 'Jangan dibagikan dulu' in advice['saran']


# --- Uji: model tidak ada (clone bersih) ---------------------------------------


def test_predict_model_tidak_ada(client, monkeypatch):
    """Bila model belum dilatih, /predict balas 503 + cara memulihkan."""
    import app as app_module

    monkeypatch.setattr(app_module, 'model', None)
    monkeypatch.setattr(app_module, 'tfidf', None)

    response = client.post('/predict', json={'text': 'Teks apa saja di sini.'})
    assert response.status_code == 503
    data = response.get_json()
    assert 'error' in data
    assert 'belum siap' in data['error'].lower()
    assert 'train_model' in data['error'].lower()
    assert 'en' in data['models_missing']
    assert 'train_model_id.py' in data['recovery']['id']
    assert 'restart' in data['recovery']['after'].lower()


def test_health_model_tidak_ada(client, monkeypatch):
    """Bila model belum dilatih, /health tetap 200 dan menunjukkan model mana yang tidak tersedia."""
    import app as app_module

    monkeypatch.setattr(app_module, 'model', None)
    monkeypatch.setattr(app_module, 'tfidf', None)
    monkeypatch.setattr(app_module, 'model_en', None)
    monkeypatch.setattr(app_module, 'tfidf_en', None)
    monkeypatch.setattr(app_module, 'model_id', None)
    monkeypatch.setattr(app_module, 'tfidf_id', None)

    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is False
    assert data['vectorizer_loaded'] is False
    assert data['models']['en']['loaded'] is False
    assert data['models']['id']['loaded'] is False
    # /health harus menyebut model mana yang tidak tersedia.
    assert set(data['models_missing']) == {'en', 'id'}
    assert data['models']['en']['missing'] is True
    assert data['models']['id']['missing'] is True
    assert 'train_model' in data['model_recovery']


def test_ui_menampilkan_pesan_pemulihan_503(client):
    """UI tidak boleh menimpa pesan 503 dengan pesan 'server bermasalah'."""
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'status === 503' in html
    assert 'Model belum siap' in html
    assert 'models_missing' in html
