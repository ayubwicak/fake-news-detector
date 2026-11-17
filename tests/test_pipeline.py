import os
import pytest
import joblib
from train_model import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_clean_text_removes_urls():
    raw = "Breaking news visit https://example.com/clickbait today!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "example" not in cleaned
    assert "breaking news visit today" == cleaned


def test_clean_text_removes_punctuation():
    raw = "ALERT!!! Massive UFO spotted??? 100% real!!!"
    cleaned = clean_text(raw)
    assert "!" not in cleaned
    assert "?" not in cleaned
    assert "%" not in cleaned
    assert "alert massive ufo spotted real" == cleaned


def test_clean_text_handles_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_trained_artifacts_exist():
    model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    vec_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
    assert os.path.exists(model_path), "model.pkl should exist"
    assert os.path.exists(vec_path), "vectorizer.pkl should exist"


def test_model_inference_pipeline():
    model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    vec_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

    model = joblib.load(model_path)
    tfidf = joblib.load(vec_path)

    sample_real = "NASA scientists confirm successful landing of rover on Mars transmitting images."
    sample_fake = "Secret alien bases discovered by anonymous whistleblowers concealing covert technology."

    vec_real = tfidf.transform([clean_text(sample_real)])
    vec_fake = tfidf.transform([clean_text(sample_fake)])

    pred_real = model.predict(vec_real)[0]
    pred_fake = model.predict(vec_fake)[0]

    assert pred_real in [0, 1]
    assert pred_fake in [0, 1]
