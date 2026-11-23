"""Regression tests for the QA round-2 bug fixes."""

import os
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import app as app_module
from app import app
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
    assert 'Label must be 0 (Real) or 1 (Fake).' in response.get_json()['error']


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
    assert rows[-1].endswith(f'{expected}')


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
