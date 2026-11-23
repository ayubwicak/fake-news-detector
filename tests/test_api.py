import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import (
    app,
    build_advice,
    detect_red_flags,
    determine_status,
    find_similar_debunked,
    is_indonesian,
)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home_page_renders(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Fake News Detection System' in response.data


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True
    assert data['vectorizer_loaded'] is True


def test_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200
    data = response.get_json()
    assert 'accuracy' in data
    assert 'roc_auc' in data
    assert 0.0 <= data['accuracy'] <= 1.0


def test_predict_fake_news(client):
    payload = {
        'text': 'Anonymous whistleblowers claim secret alien bases are concealed beneath polar ice sheets.'
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['prediction'] in ['Fake', 'Real']
    assert 0 <= data['confidence'] <= 100
    assert 'key_influences' in data
    assert isinstance(data['key_influences'], list)


def test_predict_real_news(client):
    payload = {
        'text': 'NASA scientists confirmed today that the exploratory rover has landed on Mars.'
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['prediction'] in ['Fake', 'Real']
    assert data['probability_real'] + data['probability_fake'] == pytest.approx(100, abs=0.2)


def test_predict_missing_text(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_predict_empty_text(client):
    response = client.post('/predict', json={'text': ''})
    assert response.status_code == 400


def test_predict_short_text(client):
    response = client.post('/predict', json={'text': 'hi'})
    assert response.status_code == 400


def test_predict_symbols_only(client):
    response = client.post('/predict', json={'text': '!!!@@@###$$$%%%^^^&&&'})
    assert response.status_code == 400


def test_predict_no_json_body(client):
    response = client.post('/predict', data='not-json', content_type='text/plain')
    assert response.status_code == 400


def test_feedback_saved(client, tmp_path, monkeypatch):
    import app as app_module
    feedback_file = tmp_path / 'feedback.csv'
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(feedback_file))

    response = client.post('/feedback', json={
        'text': 'A perfectly valid news text submitted as human feedback.',
        'label': 1
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'saved'
    assert data['total_feedback_samples'] == 1
    assert feedback_file.exists()

    content = feedback_file.read_text(encoding='utf-8')
    assert content.startswith('text,label')
    assert 'perfectly valid news text' in content


def test_feedback_invalid_label(client, tmp_path, monkeypatch):
    import app as app_module
    monkeypatch.setattr(app_module, 'FEEDBACK_PATH', str(tmp_path / 'feedback.csv'))
    response = client.post('/feedback', json={
        'text': 'A perfectly valid news text submitted as human feedback.',
        'label': 7
    })
    assert response.status_code == 400


def test_feedback_missing_fields(client):
    response = client.post('/feedback', json={'text': 'only text, no label here'})
    assert response.status_code == 400


def test_feedback_short_text(client):
    response = client.post('/feedback', json={'text': 'hi', 'label': 1})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Honest reporting: three-zone status, red flags, language hint, advice.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('probability_fake,expected', [
    (0.2, 'cenderung_valid'),
    (0.5, 'belum_dapat_disimpulkan'),
    (0.8, 'cenderung_hoaks'),
    (0.30, 'cenderung_valid'),
    (0.70, 'cenderung_hoaks'),
])
def test_determine_status_zones(probability_fake, expected):
    assert determine_status(probability_fake) == expected


def test_detect_red_flags_chain():
    text = (
        'SEBARKAN pesan ini segera!! Katanya beredar kabar bahwa bantuan '
        'akan ditutup. Cek http://bit.ly/contoh'
    )
    flags = detect_red_flags(text)
    codes = {flag['kode'] for flag in flags}
    # tanpa_sumber terdeteksi karena ada "Katanya beredar kabar..."
    assert {'ajakan_menyebar', 'tekanan_waktu', 'tanpa_sumber',
            'huruf_berlebihan', 'tautan_mencurigakan'} <= codes
    assert len(flags) <= 5
    for flag in flags:
        assert set(flag) == {'kode', 'label'}


def test_detect_red_flags_neutral():
    text = 'Kementerian Kesehatan merilis laporan tahunan pada hari Senin.'
    assert detect_red_flags(text) == []


def test_detect_red_flags_accepts_go_id_links():
    text = 'Informasi resmi tersedia di https://kominfo.go.id/berita terbaru.'
    codes = {flag['kode'] for flag in detect_red_flags(text)}
    assert 'tautan_mencurigakan' not in codes


def test_detect_red_flags_bare_domain_without_scheme():
    codes = {flag['kode'] for flag in detect_red_flags('Baca selengkapnya di contoh.com')}
    assert 'tautan_mencurigakan' in codes


def test_is_indonesian_true_for_indonesian_sentence():
    text = 'Pemerintah dan masyarakat akan bekerja sama untuk menjaga kesehatan bersama.'
    assert is_indonesian(text) is True


def test_is_indonesian_false_for_english_sentence():
    text = 'The government announced a new policy for public health today.'
    assert is_indonesian(text) is False


def test_advice_never_uses_absolute_wording():
    for status in ('cenderung_hoaks', 'belum_dapat_disimpulkan',
                   'cenderung_valid'):
        advice = build_advice(status, 42.0, 58.0)
        combined = ' '.join(advice.values()).lower()
        assert set(advice) == {'judul', 'ringkasan', 'saran'}
        assert 'terbukti' not in combined
        assert 'pasti' not in combined
        assert '100%' not in combined


# --- 3 aturan baru berdasarkan red_flag_count ---

def test_advice_belum_dapat_dengan_3_red_flag():
    """status belum_dapat_disimpulan + 3+ red_flag → judul & saran lebih tegas."""
    advice = build_advice('belum_dapat_disimpulkan', 55.0, 45.0, 3)
    assert advice['judul'] == 'Belum dapat disimpulkan, tetapi patut diwaspadai'
    assert 'Jangan dibagikan dulu' in advice['saran']


def test_advice_cenderung_valid_dengan_3_red_flag():
    """status cenderung_valid + 3+ red_flag → kalimat tambahan di ringkasan."""
    advice = build_advice('cenderung_valid', 60.0, 40.0, 3)
    assert 'ada beberapa ciri' in advice['ringkasan']


def test_advice_cenderung_valid_dengan_0_2_red_flag():
    """status cenderung_valid + 0-2 red_flag → ringkasan tetap seperti biasa, tanpa kalimat tambahan."""
    advice = build_advice('cenderung_valid', 60.0, 40.0, 0)
    assert 'ada beberapa ciri' not in advice['ringkasan']
    advice2 = build_advice('cenderung_valid', 60.0, 40.0, 2)
    assert 'ada beberapa ciri' not in advice2['ringkasan']


def test_advice_cenderung_hoaks_tidak_berubah():
    """status cenderung_hoaks → tidak ada perubahan apapun."""
    advice = build_advice('cenderung_hoaks', 80.0, 20.0, 5)
    assert advice['judul'] == 'Cenderung hoaks'
    assert 'Jangan dibagikan dulu' in advice['saran']


def test_predict_keeps_old_fields_and_adds_new_ones(client):
    response = client.post('/predict', json={
        'text': 'Pemerintah dan masyarakat akan bekerja sama untuk menjaga '
                'kesehatan masyarakat pada musim hujan.'
    })
    assert response.status_code == 200
    data = response.get_json()

    old_fields = [
        'prediction', 'label_code', 'confidence', 'probability_real',
        'probability_fake', 'decision_threshold', 'word_count',
        'key_influences',
    ]
    for field in old_fields:
        assert field in data, f'field lama hilang: {field}'

    assert data['status'] in {
        'cenderung_hoaks', 'belum_dapat_disimpulkan', 'cenderung_valid'
    }
    assert data['status'] == determine_status(data['probability_fake'] / 100.0)
    assert isinstance(data['red_flags'], list)
    assert isinstance(data['language_warning'], bool)
    assert set(data['advice']) == {'judul', 'ringkasan', 'saran'}
    assert data['advice']['judul'] == {
        'cenderung_hoaks': 'Cenderung hoaks',
        'belum_dapat_disimpulkan': 'Belum dapat disimpulkan',
        'cenderung_valid': 'Cenderung valid',
    }[data['status']]


def test_find_similar_debunked_hoak_returns_nonempty():
    # Teks hampir identik dengan satu judul hoak di indeks -> tidak kosong.
    hoak = 'Hillary Clinton mengaku terinspirasi oleh Sri Mulyani dan Jokowi pada debate calon presiden Amerika Serikat.'
    hasil = find_similar_debunked(hoak, top_k=3)
    assert isinstance(hasil, list)
    assert len(hasil) > 0
    for item in hasil:
        assert set(item) == {'judul', 'url', 'similarity'}
        assert item['similarity'] >= 0


def test_find_similar_debunked_neutral_returns_empty():
    # Berita netral yang topiknya tidak berkaitan hoak -> kosong.
    netral = (
        'JAKARTA - Kementerian Pekerjaan Umum dan Perumahan Rakyat '
        'mengumumkan pembangunan jalan tol baru yang menghubungkan dua '
        'kota besar di Pulau Jawa.'
    )
    hasil = find_similar_debunked(netral, top_k=3)
    assert isinstance(hasil, list)
    assert len(hasil) == 0


def test_find_similar_debunked_fallback_without_library(monkeypatch):
    # Tanpa library sentence_transformers, fungsi tidak error dan
    # mengembalikan list kosong.
    import app as app_module

    def boom(*args, **kwargs):
        raise ImportError('No module named sentence_transformers')

    monkeypatch.setattr(app_module, '_embed_texts', boom)
    hasil = find_similar_debunked('Apa saja kabar hari ini?', top_k=3)
    assert hasil == []


def test_predict_includes_similar_debunked_field(client):
    response = client.post('/predict', json={
        'text': 'Hillary Clinton mengaku terinspirasi oleh Sri Mulyani dan Jokowi pada debate calon presiden Amerika Serikat.'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'similar_debunked' in data
    assert isinstance(data['similar_debunked'], list)
    # Field lama tetap ada (tidak dihapus).
    for field in ('prediction', 'probability_fake', 'model_reliable', 'red_flags', 'status'):
        assert field in data, f'field hilang: {field}'


def test_predict_includes_debunk_available_field(client):
    response = client.post('/predict', json={
        'text': 'Halo, apa kabar hari ini?'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert 'debunk_available' in data
    assert isinstance(data['debunk_available'], bool)


def test_debunk_available_false_when_library_missing(client, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, 'DEBUNK_AVAILABLE', False)

    def boom(*args, **kwargs):
        raise ImportError('No module named sentence_transformers')

    monkeypatch.setattr(app_module, '_embed_texts', boom)
    response = client.post('/predict', json={
        'text': 'Halo, apa kabar hari ini?'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['debunk_available'] is False
    assert data['similar_debunked'] == []


def test_predict_short_text_under_five_words(client):
    # E: teks < 5 kata -> jangan jalankan model, status teks_terlalu_pendek,
    # probability null, field lama tetap ada.
    response = client.post('/predict', json={
        'text': 'halo selamat pagi semua'
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'teks_terlalu_pendek'
    assert data['prediction'] == 'Tidak dapat dianalisis'
    assert data['label_code'] is None
    assert data['confidence'] is None
    assert data['probability_real'] is None
    assert data['probability_fake'] is None
    # Field lama tetap ada (tidak dihapus, boleh null).
    for field in ('prediction', 'probability_fake', 'model_reliable', 'red_flags', 'status', 'debunk_available'):
        assert field in data, f'field hilang: {field}'
    assert data['advice'] == 'Teks terlalu pendek untuk dianalisis. Tempelkan paragraf atau berita yang lebih lengkap.'
