"""Structure-aware parent and child chunk construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import ChunkRecord, PageRecord
from .persian import normalize_search, token_count


_SENTENCE_SPLIT = re.compile(r"(?<=[.!؟?؛])\s+")


@dataclass
class Segment:
    text: str
    pdf_page: int
    book_page: int | None
    chapter_id: str
    chapter_title: str
    heading: str | None
    book_id: str
    book_title: str
    source_file: str


def _classify(text: str) -> str:
    if any(label in text for label in ("خود را بیازمایید", "تمرین", "پرسش")):
        return "exercise"
    if any(label in text for label in ("جدول", "نمودار")):
        return "table_or_chart"
    if any(label in text for label in ("شکل", "تصویر")) and len(text) < 500:
        return "caption"
    return "prose"


def _split_long_paragraph(text: str, limit: int = 260) -> list[str]:
    sentences = [item.strip() for item in _SENTENCE_SPLIT.split(text) if item.strip()]
    if token_count(text) <= limit:
        return [text]
    if len(sentences) <= 1:
        pieces: list[str] = []
        words: list[str] = []
        size = 0
        for word in text.split():
            word_size = max(1, token_count(word))
            if words and size + word_size > limit:
                pieces.append(" ".join(words))
                words, size = [], 0
            words.append(word)
            size += word_size
        if words:
            pieces.append(" ".join(words))
        return pieces
    result: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        size = token_count(sentence)
        if current and current_tokens + size > limit:
            result.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sentence)
        current_tokens += size
    if current:
        result.append(" ".join(current))
    return result


def page_segments(pages: Iterable[PageRecord]) -> list[Segment]:
    segments: list[Segment] = []
    for page in pages:
        paragraphs = [line.strip() for line in page.display_text.splitlines() if line.strip()]
        for paragraph in paragraphs:
            for piece in _split_long_paragraph(paragraph):
                if token_count(piece) < 2:
                    continue
                segments.append(
                    Segment(
                        text=piece,
                        pdf_page=page.pdf_page,
                        book_page=page.book_page,
                        chapter_id=page.chapter_id,
                        chapter_title=page.chapter_title,
                        heading=page.section_heading,
                        book_id=page.book_id,
                        book_title=page.book_title,
                        source_file=page.source_file,
                    )
                )
    return segments


def _pack(segments: list[Segment], target: int, maximum: int) -> list[list[Segment]]:
    groups: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        segment_size = token_count(segment.text)
        chapter_changed = bool(current and current[-1].chapter_id != segment.chapter_id)
        target_reached = size >= target and size + segment_size > target
        maximum_exceeded = size + segment_size > maximum
        if current and (chapter_changed or target_reached or maximum_exceeded):
            groups.append(current)
            current, size = [], 0
        current.append(segment)
        size += segment_size
    if current:
        groups.append(current)
    return groups


def _overlap_tail(group: list[Segment], desired_tokens: int) -> list[Segment]:
    tail: list[Segment] = []
    size = 0
    for segment in reversed(group):
        segment_size = token_count(segment.text)
        if size + segment_size > desired_tokens:
            break
        tail.append(segment)
        size += segment_size
    return list(reversed(tail))


def build_chunks(
    pages: list[PageRecord],
    parent_target: int = 700,
    parent_max: int = 950,
    child_target: int = 240,
    child_max: int = 320,
    overlap: int = 40,
) -> tuple[list[dict], list[ChunkRecord]]:
    segments = page_segments(pages)
    parent_groups = _pack(segments, parent_target, parent_max)
    parents: list[dict] = []
    children: list[ChunkRecord] = []
    child_number = 0

    for parent_number, parent_group in enumerate(parent_groups, start=1):
        book_id = parent_group[0].book_id
        parent_id = f"{book_id}:parent-{parent_number:04d}"
        parent_text = "\n".join(segment.text for segment in parent_group)
        parent_pages = sorted({segment.pdf_page for segment in parent_group})
        parents.append(
            {
                "parent_id": parent_id,
                "book_id": book_id,
                "book_title": parent_group[0].book_title,
                "source_file": parent_group[0].source_file,
                "chapter_id": parent_group[0].chapter_id,
                "chapter_title": parent_group[0].chapter_title,
                "pdf_pages": parent_pages,
                "book_pages": sorted({s.book_page for s in parent_group if s.book_page is not None}),
                "text": parent_text,
                "search_text": normalize_search(parent_text),
                "token_count": token_count(parent_text),
            }
        )

        raw_child_groups = _pack(parent_group, child_target, child_max)
        with_overlap: list[list[Segment]] = []
        for index, group in enumerate(raw_child_groups):
            if index == 0:
                with_overlap.append(group)
            else:
                with_overlap.append(_overlap_tail(raw_child_groups[index - 1], overlap) + group)

        for group in with_overlap:
            child_number += 1
            text = "\n".join(segment.text for segment in group)
            headings = [segment.heading for segment in group if segment.heading]
            children.append(
                ChunkRecord(
                    chunk_id=f"{book_id}:chunk-{child_number:04d}",
                    parent_id=parent_id,
                    chapter_id=group[0].chapter_id,
                    chapter_title=group[0].chapter_title,
                    section_heading=headings[0] if headings else None,
                    pdf_pages=sorted({segment.pdf_page for segment in group}),
                    book_pages=sorted({s.book_page for s in group if s.book_page is not None}),
                    content_type=_classify(text),
                    text=text,
                    search_text=normalize_search(text),
                    token_count=token_count(text),
                    book_id=book_id,
                    book_title=group[0].book_title,
                    source_file=group[0].source_file,
                )
            )
    return parents, children
