"""Gerbang keandalan — MODEL_RELIABLE dihitung, tidak ditulis manual.

Sumber: ``logs/holdout_v3_result.json`` (hasil evaluasi satu pintu).
Aturan: lulus hanya jika SEMUA kandidat yang tersedia memenuhi SEMUA ambang,
dan profil hold-out v3 memenuhi syarat sampel. Tidak ada pemilihan model
berdasarkan angka: bila salah satu kandidat tersedia gagal, gerbang gagal.
"""

from __future__ import annotations

import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULT_PATH = os.path.join(BASE_DIR, "logs", "holdout_v3_result.json")

MIN_ROC_AUC = 0.85
MIN_WILSON_LOW = 0.80
MAX_FPR_VALID = 0.15
MIN_RECALL_HOAX = 0.75
MIN_SAMPLES = 100
MIN_SOURCES = 8
MAX_SOURCE_SHARE = 0.20

RULES = {
    "roc_auc": f">= {MIN_ROC_AUC}",
    "accuracy_wilson_95_lower": f">= {MIN_WILSON_LOW}",
    "false_positive_rate_valid": f"<= {MAX_FPR_VALID}",
    "recall_hoax": f">= {MIN_RECALL_HOAX}",
    "n": f">= {MIN_SAMPLES}",
    "n_sources": f">= {MIN_SOURCES}",
    "max_source_share": f"<= {MAX_SOURCE_SHARE:.2f}",
    "policy": "semua ambang harus dipenuhi oleh SEMUA kandidat yang tersedia",
}


def _number(value):
    """Angka finite atau None (NaN/inf tidak lulus gerbang)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _profile(result: dict) -> dict:
    profile = result.get("holdout_profile") or {}
    return {
        "n": _number(profile.get("n")),
        "n_sources": _number(profile.get("n_sources")),
        "max_source_share": _number(profile.get("max_source_share")),
    }


def compute_model_reliable(result_path: str = DEFAULT_RESULT_PATH) -> dict:
    """Kembalikan dict {reliable, reasons, gates, ...}.

    Tidak melempar exception untuk hasil yang belum ada/rusak — mengembalikan
    reliable=False beserta alasannya, supaya aplikasi tetap bisa startup.
    """
    reasons: list[str] = []
    gates: dict = {}
    if not os.path.exists(result_path):
        reasons.append("logs/holdout_v3_result.json belum ada — evaluasi v3 belum dijalankan")
        return {"reliable": False, "reasons": reasons, "gates": gates,
                "result_path": os.path.relpath(result_path, BASE_DIR), "rules": dict(RULES)}

    try:
        with open(result_path, encoding="utf-8") as fh:
            result = json.load(fh)
    except Exception as exc:
        reasons.append(f"hasil evaluasi v3 tidak terbaca: {exc}")
        return {"reliable": False, "reasons": reasons, "gates": gates,
                "result_path": os.path.relpath(result_path, BASE_DIR), "rules": dict(RULES)}

    validators = result.get("validator") or {}
    if validators.get("status") != "pass":
        reasons.append("validator hold-out v3 tidak berstatus pass")

    profile = _profile(result)
    dataset_gates = {
        "n_ge_100": (profile["n"] is not None and profile["n"] >= MIN_SAMPLES),
        "n_sources_ge_8": (profile["n_sources"] is not None
                           and profile["n_sources"] >= MIN_SOURCES),
        "max_source_share_le_20pct": (profile["max_source_share"] is not None
                                      and profile["max_source_share"] <= MAX_SOURCE_SHARE),
    }
    gates["holdout_profile"] = {
        "observed": profile,
        "thresholds": {
            "n": f">= {MIN_SAMPLES}",
            "n_sources": f">= {MIN_SOURCES}",
            "max_source_share": f"<= {MAX_SOURCE_SHARE:.2f}",
        },
        "checks": dataset_gates,
    }
    for name, ok in dataset_gates.items():
        if not ok:
            reasons.append(f"profil hold-out v3 gagal: {name} (nilai={profile})")

    candidates = result.get("candidates") or {}
    available = {name: data for name, data in candidates.items()
                 if isinstance(data, dict) and data.get("available")}
    if not available:
        reasons.append("tidak ada kandidat model yang tersedia di hasil evaluasi")

    gates["candidates"] = {}
    for name, data in available.items():
        metrics = data.get("metrics") or {}
        wilson = metrics.get("accuracy_wilson_95") or [None, None]
        values = {
            "roc_auc": _number(metrics.get("roc_auc")),
            "accuracy_wilson_low": _number(wilson[0] if wilson else None),
            "false_positive_rate_valid": _number(metrics.get("false_positive_rate_valid")),
            "recall_hoax": _number(metrics.get("recall_hoax")),
        }
        checks = {
            "roc_auc_ge_0.85": values["roc_auc"] is not None and values["roc_auc"] >= MIN_ROC_AUC,
            "wilson_low_ge_0.80": (values["accuracy_wilson_low"] is not None
                                   and values["accuracy_wilson_low"] >= MIN_WILSON_LOW),
            "fpr_valid_le_0.15": (values["false_positive_rate_valid"] is not None
                                  and values["false_positive_rate_valid"] <= MAX_FPR_VALID),
            "recall_hoax_ge_0.75": (values["recall_hoax"] is not None
                                    and values["recall_hoax"] >= MIN_RECALL_HOAX),
        }
        gates["candidates"][name] = {
            "observed": values,
            "thresholds": {
                "roc_auc": f">= {MIN_ROC_AUC}",
                "accuracy_wilson_low": f">= {MIN_WILSON_LOW}",
                "false_positive_rate_valid": f"<= {MAX_FPR_VALID}",
                "recall_hoax": f">= {MIN_RECALL_HOAX}",
            },
            "checks": checks,
        }
        for check, ok in checks.items():
            if not ok:
                reasons.append(f"{name} gagal: {check} (nilai={values})")

    return {
        "reliable": not reasons,
        "reasons": reasons,
        "gates": gates,
        "result_path": os.path.relpath(result_path, BASE_DIR),
        "result_generated_at": result.get("generated_at"),
        "n_candidates_available": len(available),
        "rules": dict(RULES),
    }
