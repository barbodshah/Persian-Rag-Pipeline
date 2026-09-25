"""Shared serializable data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageRecord:
    pdf_page: int
    book_page: int | None
    chapter_id: str
    chapter_title: str
    section_heading: str | None
    raw_text: str
    display_text: str
    search_text: str
    book_id: str = "C110210"
    book_title: str = "C110210"
    source_file: str = "C110210.pdf"
    spans: list[dict[str, Any]] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    parent_id: str
    chapter_id: str
    chapter_title: str
    section_heading: str | None
    pdf_pages: list[int]
    book_pages: list[int]
    content_type: str
    text: str
    search_text: str
    token_count: int
    book_id: str = "C110210"
    book_title: str = "C110210"
    source_file: str = "C110210.pdf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
