"""Human-readable textbook names and citation labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_PATH = Path(__file__).with_name("book_catalog.json")
_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def load_book_catalog(path: Path | None = None) -> dict[str, str]:
    catalog_path = path or DEFAULT_CATALOG_PATH
    value: Any = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Book catalog must be a JSON object mapping book IDs to titles.")
    catalog: dict[str, str] = {}
    for book_id, title in value.items():
        if not isinstance(book_id, str) or not isinstance(title, str) or not title.strip():
            raise ValueError("Every book catalog entry must map a string ID to a non-empty title.")
        catalog[book_id] = title.strip()
    return catalog


def citation_label(book_title: str, book_pages: list[int]) -> str:
    pages = sorted({int(page) for page in book_pages})
    if not pages:
        return book_title
    formatted = "، ".join(str(page).translate(_PERSIAN_DIGITS) for page in pages)
    page_word = "صفحه" if len(pages) == 1 else "صفحه‌های"
    return f"{book_title}، {page_word} {formatted}"
