"""Persian text normalization and lightweight tokenization."""

from __future__ import annotations

import re
import unicodedata


_CHAR_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "ـ": "",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)

_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_SPACES = re.compile(r"[ \t\f\v]+")
_SEARCH_SEPARATORS = re.compile(r"[^0-9A-Za-z\u0600-\u06ff+\-.]+")
_TOKEN = re.compile(r"[0-9A-Za-z\u0600-\u06ff]+(?:[+\-.][0-9A-Za-z\u0600-\u06ff]+)*")


def normalize_display(text: str) -> str:
    """Normalize encoding noise while retaining readable punctuation and lines."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_CHAR_TRANSLATION)
    text = text.replace("\u200e", "").replace("\u200f", "")
    text = text.replace("\u200a", " ").replace("\u202f", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def normalize_search(text: str) -> str:
    """Create a conservative comparison form for Persian lexical search."""
    text = normalize_display(text)
    text = _DIACRITICS.sub("", text)
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = _SEARCH_SEPARATORS.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized Persian, Latin text, numbers, and formulas."""
    normalized = normalize_search(text)
    return [match.group(0) for match in _TOKEN.finditer(normalized)]


def token_count(text: str) -> int:
    return len(tokenize(text))
