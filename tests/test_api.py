import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app import app


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
