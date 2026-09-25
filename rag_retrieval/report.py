"""Generate a concise extraction review for representative pages."""

from __future__ import annotations

from pathlib import Path

from .models import PageRecord


REPRESENTATIVE_PAGES = (11, 20, 49, 55, 66, 82, 93, 101, 124, 140)


def write_extraction_review(path: Path, pages: list[PageRecord]) -> None:
    by_number = {page.pdf_page: page for page in pages}
    lines = [
        "# Extraction review",
        "",
        "Ten pages were selected across chapter openings, prose, equations, tables, exercises, and water chemistry.",
        "The searchable source uses the PDF logical text layer; layout spans are retained separately for provenance.",
        "",
        "| PDF page | Book page | Chapter | Characters | Tokens | Spans | OCR review |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for number in REPRESENTATIVE_PAGES:
        page = by_number[number]
        quality = page.quality
        lines.append(
            f"| {number} | {page.book_page or '-'} | {page.chapter_title} | "
            f"{quality['characters']} | {quality['tokens']} | {quality['span_count']} | "
            f"{'yes' if quality['needs_ocr_review'] else 'no'} |"
        )
    lines.extend(["", "## Search-text excerpts", ""])
    for number in REPRESENTATIVE_PAGES:
        page = by_number[number]
        excerpt = page.search_text[:500].replace("\n", " ")
        lines.extend(
            [
                f"### PDF page {number} / book page {page.book_page or '-'}",
                "",
                excerpt,
                "",
            ]
        )
    lines.extend(
        [
            "## Known limitations",
            "",
            "- The logical layer is readable Persian, but some PDF spans split words or reorder formula fragments.",
            "- Tables and diagrams are indexed through extracted text and captions; exact spatial reconstruction is deferred.",
            "- Pages with fewer than 80 extracted characters are flagged for targeted OCR review.",
            "- Both PDF and printed book page numbers are retained because the book content starts at PDF page 11.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
