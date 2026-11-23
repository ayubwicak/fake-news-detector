"""Tes gerbang keandalan otomatis (Butir 4) — hasil SINTETIS, bukan v3 asli."""

import json

import pytest

from reliability import compute_model_reliable

from app import app as flask_app  # noqa: E402


def _result(**overrides):
    result = {
        "generated_at": "2026-11-23T00:00:00+00:00",
        "validator": {"status": "pass"},
        "holdout_profile": {"n": 120, "n_sources": 10, "max_source_share": 0.12},
        "candidates": {
            "tfidf_lr_id": {
                "available": True,
                "metrics": {
                    "roc_auc": 0.88,
                    "accuracy_wilson_95": [0.82, 0.92],
                    "false_positive_rate_valid": 0.10,
                    "recall_hoax": 0.80,
                },
            },
            "sent_trans_lr": {"available": False, "reason": "artefak tidak ada"},
        },
    }
    for key, value in overrides.items():
        if key == "metrics":
            result["candidates"]["tfidf_lr_id"]["metrics"].update(value)
        else:
            result[key] = value
    return result


def _write(tmp_path, payload):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_tanpa_hasil_evaluasi_false(tmp_path):
    out = compute_model_reliable(str(tmp_path / "tidak_ada.json"))
    assert out["reliable"] is False
    assert any("belum ada" in reason for reason in out["reasons"])


def test_semua_gerbang_lulus_true(tmp_path):
    out = compute_model_reliable(_write(tmp_path, _result()))
    assert out["reliable"] is True, out["reasons"]
    assert out["n_candidates_available"] == 1


def test_satu_metrik_gagal_false(tmp_path):
    out = compute_model_reliable(_write(tmp_path, _result(metrics={"roc_auc": 0.84})))
    assert out["reliable"] is False
    assert any("roc_auc_ge_0.85" in reason for reason in out["reasons"])


def test_wilson_fpr_recall_gagal_false(tmp_path):
    out = compute_model_reliable(_write(tmp_path, _result(metrics={
        "accuracy_wilson_95": [0.78, 0.90],
        "false_positive_rate_valid": 0.20,
        "recall_hoax": 0.70,
    })))
    assert out["reliable"] is False
    joined = " ".join(out["reasons"])
    assert "wilson_low_ge_0.80" in joined
    assert "fpr_valid_le_0.15" in joined
    assert "recall_hoax_ge_0.75" in joined


def test_profil_sampel_gagal_false(tmp_path):
    out = compute_model_reliable(_write(tmp_path, _result(
        holdout_profile={"n": 80, "n_sources": 6, "max_source_share": 0.35})))
    assert out["reliable"] is False
    joined = " ".join(out["reasons"])
    assert "n_ge_100" in joined
    assert "n_sources_ge_8" in joined
    assert "max_source_share_le_20pct" in joined


def test_nan_tidak_lulus(tmp_path):
    out = compute_model_reliable(_write(tmp_path, _result(metrics={"roc_auc": float("nan")})))
    assert out["reliable"] is False


def test_validator_tidak_lulus_tetap_false(tmp_path):
    out = compute_model_reliable(_write(tmp_path, _result(validator={"status": "fail"})))
    assert out["reliable"] is False
    assert any("validator" in reason for reason in out["reasons"])


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


def test_ui_berganti_mode(client, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, 'MODEL_RELIABLE', False)
    html = client.get('/').get_data(as_text=True)
    assert 'data-mode="experimental"' in html
    assert 'Mode eksperimental' in html

    monkeypatch.setattr(app_module, 'MODEL_RELIABLE', True)
    html = client.get('/').get_data(as_text=True)
    assert 'data-mode="reliable"' in html
    assert 'Mode tervalidasi' in html


def test_predict_mengikuti_flag_gate(client, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, 'MODEL_RELIABLE', True)
    data = client.post('/predict', json={
        'text': 'Pemerintah mengumumkan kebijakan baru untuk masyarakat luas.'
    }).get_json()
    assert data['model_reliable'] is True


def test_health_menampilkan_alasan_gate(client):
    data = client.get('/health').get_json()
    assert data['model_reliability']['reliable'] is False
    assert isinstance(data['model_reliability']['reasons'], list)
    assert 'roc_auc' in data['model_reliability']['rules']
