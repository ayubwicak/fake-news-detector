"""Tes untuk perangkat pelabelan hold-out v3 (Butir 2)."""

import csv
import os
import sys
from datetime import date, timedelta

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agreement as agreement_mod  # noqa: E402
import validate_holdout_v3 as validator  # noqa: E402


TOPICS = ["politik", "kesehatan", "ekonomi", "bencana", "penipuan_daring"]
FORMS = ["judul", "paragraf", "artikel", "pesan_berantai"]


def _alpha_token(i: int) -> str:
    """Token huruf unik (clean_text membuang digit, jadi angka tidak bisa dipakai)."""
    chars = []
    n = i + 1
    while n:
        chars.append(chr(97 + n % 26))
        n //= 26
    return "kata" + "".join(reversed(chars))


def make_rows(n=100):
    """Data sintetis VALID untuk validator (bukan data hold-out asli)."""
    sources = [f"sumber{i:02d}" for i in range(10)]
    start = date(2025, 1, 15)
    rows = []
    for i in range(n):
        label = i % 2
        # Sumber dipilih terpisah dari label agar tiap kelas mendapat 10 sumber.
        source = sources[(i // 2) % len(sources)]
        rows.append({
            "id": f"v3-{i:03d}",
            "text": (
                f"Ringkasan uji {_alpha_token(i)} dari {source} mengenai {TOPICS[i % 5]} "
                "yang dilaporkan secara berkala. "
                "Teks ini sengaja dibuat unik untuk keperluan pengujian validator."
            ),
            "label": label,
            "sumber_url": f"https://{source}.example/artikel/{i}",
            "nama_sumber": source,
            "tanggal_terbit": (start + timedelta(days=30 * i)).isoformat(),
            "topik": TOPICS[i % 5],
            "bentuk": FORMS[i % 4],
            "bukti_url": (
                f"https://cekfakta.example/bantahan/{i}" if label == 1
                else f"https://media.example/liputan/{i}"
            ),
            "catatan": "data sintetis untuk tes",
        })
    return rows


def test_validator_menerima_data_valid(tmp_path):
    report = validator.validate(make_rows(), base_dir=str(tmp_path))
    assert report["status"] == "pass", report["violations"]
    assert report["n_rows"] == 100
    assert all(check["ok"] for check in report["checks"].values())


def test_validator_mendeteksi_kebocoran_ke_data_latih(tmp_path):
    rows = make_rows()
    (tmp_path / "data").mkdir()
    with open(tmp_path / "data" / "train_id.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerow({"text": rows[7]["text"], "label": 0})

    report = validator.validate(rows, base_dir=str(tmp_path), check_external=True)
    check = report["checks"]["near_duplicate_external"]
    assert check["ok"] is False
    assert "train_id.csv" in check["detail"]
    assert report["status"] == "fail"


def test_validator_mendeteksi_sumber_dominan(tmp_path):
    rows = make_rows()
    for row in rows[:70]:
        row["nama_sumber"] = "sumber_dominan"
        row["sumber_url"] = "https://sumber_dominan.example/artikel"
    report = validator.validate(rows, base_dir=str(tmp_path), check_external=False)
    check = report["checks"]["source_share"]
    assert check["ok"] is False
    assert "70/100" in check["detail"]
    assert report["status"] == "fail"


def test_validator_menolak_kolom_wajib_hilang(tmp_path):
    rows = make_rows(4)
    for row in rows:
        row.pop("bukti_url")
    report = validator.validate(rows, base_dir=str(tmp_path), check_external=False)
    assert report["status"] == "fail"
    assert report["checks"]["columns"]["ok"] is False


def test_validator_menolak_label_invalid(tmp_path):
    rows = make_rows(4)
    rows[0]["label"] = "99"
    report = validator.validate(rows, base_dir=str(tmp_path), check_external=False)
    assert report["checks"]["labels"]["ok"] is False


def test_agreement_kappa_sempurna():
    rows = [
        {"id": f"v3-{i:03d}", "label_anotator1": i % 2, "label_anotator2": i % 2}
        for i in range(10)
    ]
    result = agreement_mod.compute_agreement(rows)
    assert result["status"] == "pass"
    assert result["cohens_kappa"] == pytest.approx(1.0)
    assert result["disagreements"] == []
    assert result["n_pairs"] == 10


def test_agreement_mencatat_baris_tidak_sepakat():
    rows = [
        {"id": f"v3-{i:03d}", "label_anotator1": i % 2,
         "label_anotator2": (i + 1) % 2 if i < 4 else i % 2}
        for i in range(10)
    ]
    result = agreement_mod.compute_agreement(rows)
    assert result["status"] == "pass"
    assert len(result["disagreements"]) == 4
    assert result["cohens_kappa"] < 1.0
    assert 0.0 <= result["agreement_rate_wilson_95"][0] <= 1.0
