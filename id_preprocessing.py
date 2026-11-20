"""Shared Indonesian text preprocessing and lightweight language detection.

Used by both the training pipeline and the Flask API so that inference
preprocessing is identical to training preprocessing (no train/serve skew).

Design notes
------------
- No stemming: Sastrawi stemming costs ~40 ms/word, which is impractical for
  training on hundreds of full articles and adds latency at inference time.
- Stopword removal keeps negation/modality words ("tidak", "bukan", "belum",
  "bisa", "harus", ...) because they carry important fake-news signal.
- Meta words that leak the label ("hoax", "hoaks", "fake news", "debunk")
  and news-agency markers are stripped during dataset preparation.  The
  cleaning function here also strips them at inference time for consistency.
"""

from __future__ import annotations

import re
from typing import Dict

try:  # Sastrawi is optional at import time so the module stays testable.
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
        StopWordRemoverFactory,
    )

    _SASTRAWI_STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
except Exception:  # pragma: no cover - fallback for minimal environments
    _SASTRAWI_STOPWORDS = set()

# Keep negation / modality / evidential words: they matter for factuality.
_KEEP_WORDS = {
    "tidak", "bukan", "belum", "jangan", "tanpa", "bisa", "dapat", "boleh",
    "harus", "akan", "mau", "ingin", "hendak", "sudah", "telah", "masih",
    "pernah", "sedang", "nanti", "pasti", "seharusnya", "setidaknya",
    "mungkin", "banyak", "sedikit", "semua", "setiap", "hanya", "juga",
    "lagi", "kembali", "lebih", "kurang", "sangat", "paling", "sama",
}

INDONESIAN_STOPWORDS = sorted(_SASTRAWI_STOPWORDS - _KEEP_WORDS)

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_EMAIL_RE = re.compile(r"\S+@\S+")
_WS_RE = re.compile(r"\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")

# Boilerplate sentences commonly copied between news portals.
_BOILERPLATE_RE = re.compile(
    r"(?:baca juga|simak juga|artikel ini telah tayang di|editor|penulis|"
    r"tags?|tag|sumber)\s*:[^.]{0,300}\.",
    re.IGNORECASE,
)

# Agency / portal markers.  Only unambiguous brand strings are removed:
# "detik" (second) and "tempo" (time) are ordinary Indonesian words.
_SOURCE_RE = re.compile(
    r"\b(?:detikcom|detik\.com|kompas\.com|kompas|tribunnews|tribun|"
    r"liputan6|kumparan|tempo\.co|cnnindonesia|cnn indonesia|republika|"
    r"antaranews|viva\.co\.id|okezone|sindonews|suara\.com|rmol|jawapos|"
    r"pikiranrakyat|pikiran rakyat|turnbackhoax\.id|turnbackhoax)\b",
    re.IGNORECASE,
)

# Label-leaking meta vocabulary: real fact-check articles talk *about* hoaxes.
_META_RE = re.compile(
    r"\b(?:hoax|hoaks|hoak|fake news|cek fakta|fact check|debunk|"
    r"disinformasi|misinformasi)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalise Indonesian text for TF-IDF and XAI.

    Removes URLs, HTML, e-mail addresses, portal boilerplate, agency markers,
    and label-leaking meta words; lowercases; keeps alphabetic tokens only.
    """
    if not isinstance(text, str):
        return ""

    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _SOURCE_RE.sub(" ", text)
    text = _META_RE.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize_label(value) -> int:
    """Map dataset label spellings to 0 = Real/Valid, 1 = Fake/Hoax."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1 if int(value) == 1 else 0
    text = str(value).strip().lower()
    if text in {"1", "fake", "hoax", "hoaks", "false", "palsu"}:
        return 1
    if text in {"0", "real", "valid", "true", "asli", "fakta"}:
        return 0
    raise ValueError(f"Unknown label: {value!r}")


# ---------------------------------------------------------------------------
# Lightweight language detection (heuristic, explainable)
# ---------------------------------------------------------------------------
# This is deliberately NOT a trained language ID model.  It is a transparent
# lexicon vote that distinguishes Indonesian from three major regional
# languages, so the API can warn users when a prediction is out of domain.

_LANGUAGE_NAMES = {
    "id": "Indonesian",
    "en": "English",
    "jv": "Javanese",
    "su": "Sundanese",
    "btk": "Batak",
    "unknown": "Unknown",
}

_LEXICONS: Dict[str, Dict[str, int]] = {
    "jv": {
        "ora": 3, "kowe": 3, "aku": 1, "piye": 3, "kepiye": 3, "nggih": 3,
        "meniko": 3, "puniko": 3, "sampun": 2, "dereng": 3, "mboten": 3,
        "wae": 2, "iku": 2, "iki": 2, "kui": 2, "sing": 2, "banget": 1,
        "tenan": 2, "yo": 1, "lah": 1, "ta": 1, "kok": 1, "gak": 1,
        "nggak": 1, "wis": 2, "durung": 3, "karo": 2, "kanggo": 2,
        "saka": 2, "menyang": 2, "ana": 1, "akeh": 2, "sithik": 2,
    },
    "su": {
        "kuring": 3, "abdi": 2, "naon": 3, "kumaha": 3, "saha": 1,
        "iraha": 3, "teh": 1, "euy": 3, "atuh": 2, "mah": 1, "pisan": 2,
        "henteu": 3, "moal": 3, "can": 1, "geus": 2, "rek": 2, "bade": 2,
        "ngan": 2, "ieu": 2, "eta": 1, "kitu": 2, "sabaraha": 3, "aya": 1,
        "teu": 2, "lain": 2, "deui": 2, "oge": 2, "pami": 2, "mangga": 2,
    },
    "btk": {
        "ahu": 2, "ho": 1, "do": 2, "ma": 1, "ndang": 3, "dang": 2,
        "hape": 2, "molo": 2, "asa": 1, "las": 2, "roha": 3, "tondi": 3,
        "sihol": 3, "holong": 3, "pariban": 3, "boru": 2, "dongan": 3,
        "sian": 3, "mang": 1, "nang": 1, "pe": 1, "hian": 3, "sude": 2,
        "dope": 2, "tu": 1, "di": 1,
    },
    "id": {
        "yang": 1, "dan": 1, "di": 1, "ke": 1, "dari": 1, "untuk": 1,
        "dengan": 1, "adalah": 2, "ini": 1, "itu": 1, "tidak": 2,
        "bukan": 2, "sudah": 1, "belum": 1, "akan": 1, "pada": 1,
        "dalam": 1, "karena": 1, "jika": 1, "juga": 1, "bisa": 1,
        "harus": 1, "mereka": 2, "kami": 2, "kita": 1, "saya": 1,
        "anda": 1, "tersebut": 2, "oleh": 1, "sebagai": 1, "menjadi": 1,
        "ada": 1, "kepada": 1, "terhadap": 1, "tentang": 1, "saat": 1,
        "setelah": 1, "sebelum": 1, "hingga": 1, "sampai": 1, "lebih": 1,
        "sangat": 1, "hanya": 1, "masih": 1, "telah": 1, "para": 1,
        "bagi": 1, "secara": 1, "namun": 1, "tetapi": 1, "sehingga": 1,
        "pemerintah": 1, "indonesia": 1, "kata": 1, "ujar": 1, "menurut": 1,
        # English news vocabulary (added for language detection)
        "the": 1, "and": 1, "a": 1, "an": 1, "of": 1, "to": 1, "in": 1, "is": 1,
        "for": 1, "with": 1, "on": 1, "at": 1, "by": 1, "from": 1, "about": 1,
        "as": 1, "was": 1, "were": 1, "are": 1, "been": 1, "had": 1, "has": 1,
        "have": 1, "but": 1, "not": 1, "you": 1, "your": 1, "we": 1, "our": 1,
        "they": 1, "them": 1, "his": 1, "her": 1, "it": 1, "its": 1, "i": 1,
        "me": 1, "my": 1, "this": 1, "that": 1, "these": 1, "those": 1,
        "news": 1, "breaking": 1, "report": 1, "article": 1, "story": 1,
        "journal": 1, "press": 1, "media": 1, "globe": 1, "world": 1,
        "can": 1, "will": 1, "should": 1, "would": 1, "could": 1, "may": 1,
        "might": 1, "must": 1, "do": 1, "does": 1, "did": 1,
        "that": 1, "what": 1, "when": 1, "where": 1, "why": 1, "how": 1,
        "there": 1, "here": 1, "then": 1, "than": 1, "some": 1, "any": 1,
        "no": 1, "all": 1, "only": 1, "own": 1, "same": 1, "so": 1,
    },
}

_WORD_RE = re.compile(r"[a-zA-Z]+")


def detect_language(text: str) -> dict:
    """Return a transparent lexicon-based language guess.

    The result includes per-language scores and a confidence flag.  Regional
    languages are reported honestly as unsupported by the Indonesian model.
    """
    tokens = _WORD_RE.findall((text or "").lower())
    if not tokens:
        return {
            "code": "unknown",
            "name": _LANGUAGE_NAMES["unknown"],
            "supported": False,
            "scores": {},
            "method": "lexicon-vote",
        }

    scores = {}
    for lang, lexicon in _LEXICONS.items():
        scores[lang] = sum(lexicon.get(tok, 0) for tok in tokens)

    # Regional language wins only with at least two distinct marker hits.
    regional = {k: v for k, v in scores.items() if k != "id"}
    best_regional, best_regional_score = max(regional.items(), key=lambda kv: kv[1])
    distinct_hits = len({tok for tok in tokens if tok in _LEXICONS[best_regional]})

    if distinct_hits >= 2 and best_regional_score >= scores["id"]:
        code = best_regional
    elif scores["id"] >= 2:
        code = "id"
    else:
        code = "unknown"

    return {
        "code": code,
        "name": _LANGUAGE_NAMES[code],
        "supported": code == "id",
        "scores": scores,
        "method": "lexicon-vote",
    }
