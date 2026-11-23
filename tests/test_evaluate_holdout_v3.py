"""Tes gerbang satu pintu evaluator hold-out v3 (Butir 3).

Tes ini memakai data SINTETIS dan tidak menjalankan evaluasi pada hold-out v3.
"""

import hashlib
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import evaluate_on_holdout_v3 as evaluator  # noqa: E402


def _write_csv(path, n=4):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("id,text,label\n")
        for i in range(n):
            fh.write(f"v3-{i:03d},Teks sintetis untuk pengujian gerbang, {i % 2}\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _write_validation(path, hash_value, status="pass"):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"status": status, "csv_sha256": hash_value,
                   "generated_at": "2026-11-23T00:00:00+00:00"}, fh)


def _attempts(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_evaluator_menolak_tanpa_validator_lolos(tmp_path):
    csv_path = tmp_path / "holdout.csv"
    _write_csv(csv_path)
    result = tmp_path / "result.json"
    attempts = tmp_path / "attempts.jsonl"
    code = evaluator.main([
        "--csv", str(csv_path),
        "--validation-report", str(tmp_path / "validation.json"),
        "--result", str(result),
        "--attempts", str(attempts),
    ])
    assert code == 2
    assert not result.exists()
    log = _attempts(attempts)
    assert len(log) == 1 and log[0]["outcome"] == "blocked"


def test_evaluator_menolak_validator_status_fail(tmp_path):
    csv_path = tmp_path / "holdout.csv"
    digest = _write_csv(csv_path)
    validation = tmp_path / "validation.json"
    _write_validation(validation, digest, status="fail")
    result = tmp_path / "result.json"
    code = evaluator.main([
        "--csv", str(csv_path), "--validation-report", str(validation),
        "--result", str(result), "--attempts", str(tmp_path / "a.jsonl"),
    ])
    assert code == 2
    assert not result.exists()


def test_evaluator_menolak_hash_csv_berubah(tmp_path):
    csv_path = tmp_path / "holdout.csv"
    _write_csv(csv_path, n=4)
    validation = tmp_path / "validation.json"
    _write_validation(validation, "0" * 64)
    code = evaluator.main([
        "--csv", str(csv_path), "--validation-report", str(validation),
        "--result", str(tmp_path / "result.json"),
        "--attempts", str(tmp_path / "a.jsonl"),
    ])
    assert code == 2


def test_evaluator_menolak_run_kedua_hash_sama(tmp_path):
    csv_path = tmp_path / "holdout.csv"
    digest = _write_csv(csv_path)
    validation = tmp_path / "validation.json"
    _write_validation(validation, digest)
    result = tmp_path / "result.json"
    original = {"holdout_sha256": digest, "generated_at": "sebelumnya"}
    with open(result, "w", encoding="utf-8") as fh:
        json.dump(original, fh)

    passed, reason, _ = evaluator.check_guard(
        str(csv_path), str(validation), str(result), False)
    assert passed is False
    assert "sudah pernah dijalankan" in reason

    attempts = tmp_path / "attempts.jsonl"
    code = evaluator.main([
        "--csv", str(csv_path), "--validation-report", str(validation),
        "--result", str(result), "--attempts", str(attempts),
    ])
    assert code == 3
    assert json.load(open(result, encoding="utf-8")) == original
    assert len(_attempts(attempts)) == 1


def test_rerun_dengan_flag_diizinkan_oleh_guard(tmp_path):
    csv_path = tmp_path / "holdout.csv"
    digest = _write_csv(csv_path)
    validation = tmp_path / "validation.json"
    _write_validation(validation, digest)
    result = tmp_path / "result.json"
    with open(result, "w", encoding="utf-8") as fh:
        json.dump({"holdout_sha256": digest}, fh)
    passed, reason, info = evaluator.check_guard(
        str(csv_path), str(validation), str(result), True)
    assert passed is True, reason
    assert info["previous_result_sha256"] == digest


def test_metrik_klasifikasi_sintetis():
    y_true = [0, 0, 1, 1]
    prob = [0.10, 0.60, 0.40, 0.90]
    pred = [0, 1, 0, 1]
    metrics = evaluator.compute_classification_metrics(y_true, prob, pred)
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["false_positive_rate_valid"] == 0.5
    assert metrics["recall_hoax"] == 0.5
    assert metrics["gray_zone_pct"] == 0.5  # 0.60 dan 0.40
    lo, hi = metrics["accuracy_wilson_95"]
    assert lo <= metrics["accuracy"] <= hi
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_metrik_pencocokan_klaim_sintetis():
    hits = [1, 0, 1, 0]
    labels = [1, 1, 0, 0]
    m = evaluator.compute_claim_metrics(hits, labels)
    assert m["n_hoax"] == 2 and m["n_valid"] == 2
    assert m["recall_hoax"] == 0.5
    assert m["false_match_on_valid"] == 0.5
    assert m["precision_at_prevalence"]["10%"] == 0.1
    assert m["precision_at_prevalence"]["50%"] == 0.5
    lo, hi = m["recall_hoax_wilson_95"]
    assert lo <= 0.5 <= hi


def test_per_group_metrics_sumber():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    prob = [0.2, 0.8, 0.6, 0.9]
    groups = {"sumber": ["A", "A", "B", "B"]}
    out = evaluator.per_group_metrics(y_true, y_pred, prob, groups)
    assert out["sumber"]["A"]["n"] == 2
    assert out["sumber"]["A"]["accuracy"] == 1.0
    assert out["sumber"]["B"]["accuracy"] == 0.5
