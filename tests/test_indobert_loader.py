"""Tes loader IndoBERT: hanya memuat bila checksum cocok (Butir 6).

Tidak ada model asli yang dimuat; berkas di tes ini sintetis.
"""

import hashlib
import os

import pytest

import app as app_module


def _make_model_dir(tmp_path, files):
    """Buat folder model sintetis + checksums.txt dari dict {nama: bytes}."""
    out = tmp_path / "indobert"
    out.mkdir()
    lines = []
    for name, data in files.items():
        (out / name).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    (out / "checksums.txt").write_text("".join(lines), encoding="utf-8")
    return str(out)


def test_checksum_ok_semua_cocok(tmp_path):
    path = _make_model_dir(tmp_path, {
        "config.json": b'{"model_type":"bert"}',
        "tokenizer.json": b"{}",
        "model.safetensors": b"bobot",
    })
    ok, reason = app_module._indobert_checksums_ok(path)
    assert ok is True, reason
    assert "3 berkas terverifikasi" in reason


def test_checksum_gagal_tanpa_file(tmp_path):
    out = tmp_path / "indobert"
    out.mkdir()
    ok, reason = app_module._indobert_checksums_ok(str(out))
    assert ok is False
    assert "checksums.txt tidak ada" in reason


def test_checksum_gagal_hash_tidak_cocok(tmp_path):
    path = _make_model_dir(tmp_path, {
        "config.json": b"asli",
        "tokenizer.json": b"asli",
    })
    # Ubah salah satu berkas setelah checksum dibuat.
    with open(os.path.join(path, "config.json"), "wb") as fh:
        fh.write(b"diubah")
    ok, reason = app_module._indobert_checksums_ok(path)
    assert ok is False
    assert "config.json" in reason
    assert "tidak cocok" in reason


def test_checksum_gagal_berkas_hilang(tmp_path):
    path = _make_model_dir(tmp_path, {"config.json": b"a", "tokenizer.json": b"b"})
    os.remove(os.path.join(path, "tokenizer.json"))
    ok, reason = app_module._indobert_checksums_ok(path)
    assert ok is False
    assert "tokenizer.json tidak ada" in reason


def test_loader_tidak_memuat_bila_checksum_gagal(tmp_path, monkeypatch, capsys):
    """Checksum gagal -> model tetap None, tanpa mengimpor torch sama sekali."""
    out = tmp_path / "indobert"
    out.mkdir()  # tidak ada checksums.txt

    monkeypatch.setattr(app_module, "INDOBERT_DIR", str(out))
    monkeypatch.setattr(app_module, "_INDOBERT_READY", False)
    monkeypatch.setattr(app_module, "INDOBERT_MODEL", "SENTINEL")
    monkeypatch.setattr(app_module, "INDOBERT_TOKENIZER", "SENTINEL")

    app_module._load_indobert_optional()

    assert app_module.INDOBERT_MODEL == "SENTINEL"  # tidak diubah
    assert app_module.INDOBERT_TOKENIZER == "SENTINEL"
    assert "dilewati" in capsys.readouterr().out


def test_health_menampilkan_status_checksum():
    data = app_module.app.test_client().get("/health").get_json()
    assert "checksums_ok" in data["indobert"]
    assert isinstance(data["indobert"]["checksum_reason"], str)
