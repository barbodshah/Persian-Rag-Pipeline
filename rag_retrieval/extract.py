"""Extract logical Persian text and retain PDF layout provenance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import PageRecord
from .persian import normalize_display, normalize_search, token_count


CHAPTERS = (
    (1, 10, "front", "بخش آغازین"),
    (11, 54, "chapter-1", "کیهان، زادگاه الفبای هستی"),
    (55, 100, "chapter-2", "ردپای گازها در زندگی"),
    (101, 143, "chapter-3", "آب، آهنگ زندگی"),
    (144, 148, "glossary", "واژه نامه"),
    (149, 149, "references", "منابع"),
    (150, 150, "back", "پشت جلد"),
)

_STANDALONE_NUMBER = re.compile(r"^[0-9۰-۹٠-٩]+$")
_SENTENCE_END = re.compile(r"[.!؟?!؛:]$")


def chapter_for(pdf_page: int) -> tuple[str, str]:
    for start, end, chapter_id, title in CHAPTERS:
        if start <= pdf_page <= end:
            return chapter_id, title
    return "unknown", "نامشخص"


def book_page_for(pdf_page: int) -> int | None:
    if 11 <= pdf_page <= 149:
        return pdf_page - 10
    return None


def _clean_page_text(text: str, book_page: int | None) -> str:
    lines = normalize_display(text).splitlines()
    cleaned: list[str] = []
    for line in lines:
        if _STANDALONE_NUMBER.fullmatch(line):
            try:
                number = int(line.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")))
            except ValueError:
                number = -1
            if book_page is not None and number == book_page:
                continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _section_heading(display_text: str, chapter_title: str) -> str | None:
    for line in display_text.splitlines()[:12]:
        compact = line.strip(" .…")
        if not compact or len(compact) > 90 or _STANDALONE_NUMBER.fullmatch(compact):
            continue
        if "فصل" in compact or chapter_title.replace("،", "") in compact.replace("،", ""):
            return compact
        if len(compact.split()) <= 8 and not _SENTENCE_END.search(compact):
            if not re.search(r"[=+×÷<>]", compact):
                return compact
    return None


def _layout_spans(page: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []

    def visitor(text: str, _cm: Any, tm: Any, font: Any, font_size: float) -> None:
        cleaned = normalize_display(text)
        if not cleaned:
            return
        spans.append(
            {
                "text": cleaned,
                "x": round(float(tm[4]), 2),
                "y": round(float(tm[5]), 2),
                "font_size": round(float(font_size), 2),
                "font": (font or {}).get("/BaseFont") if isinstance(font, dict) else None,
            }
        )

    page.extract_text(visitor_text=visitor)
    return spans


def extract_book(
    pdf_path: Path,
    book_id: str | None = None,
    book_title: str | None = None,
    source_file: str | None = None,
) -> list[PageRecord]:
    book_id = book_id or pdf_path.stem
    book_title = book_title or pdf_path.stem
    source_file = source_file or pdf_path.name
    is_c110210 = book_id.casefold() == "c110210"
    reader = PdfReader(str(pdf_path))
    pages: list[PageRecord] = []
    for index, page in enumerate(reader.pages, start=1):
        if is_c110210:
            chapter_id, chapter_title = chapter_for(index)
            book_page = book_page_for(index)
        else:
            chapter_id, chapter_title = "document", book_title
            book_page = index
        raw_text = page.extract_text() or ""
        display_text = _clean_page_text(raw_text, book_page)
        spans = _layout_spans(page)
        pages.append(
            PageRecord(
                pdf_page=index,
                book_page=book_page,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                section_heading=_section_heading(display_text, chapter_title),
                raw_text=raw_text,
                display_text=display_text,
                search_text=normalize_search(display_text),
                book_id=book_id,
                book_title=book_title,
                source_file=source_file,
                spans=spans,
                quality={
                    "characters": len(display_text),
                    "tokens": token_count(display_text),
                    "span_count": len(spans),
                    "has_text": bool(display_text),
                    "needs_ocr_review": len(display_text) < 80 and index not in {1, 3, 4, 5, 6, 7, 8, 9, 10, 150},
                },
            )
        )
    return pages
