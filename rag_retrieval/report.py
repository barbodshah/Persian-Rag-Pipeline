"""Generate a concise extraction review for representative pages."""

from __future__ import annotations

from pathlib import Path

from .models import PageRecord


def write_extraction_review(path: Path, pages: list[PageRecord]) -> None:
    samples: list[PageRecord] = []
    by_book: dict[str, list[PageRecord]] = {}
    for page in pages:
        by_book.setdefault(page.book_id, []).append(page)
    for book_pages in by_book.values():
        if len(book_pages) <= 5:
            samples.extend(book_pages)
            continue
        indexes = {0, len(book_pages) // 4, len(book_pages) // 2, 3 * len(book_pages) // 4, len(book_pages) - 1}
        samples.extend(book_pages[index] for index in sorted(indexes))
    lines = [
        "# Extraction review",
        "",
        "Representative pages were selected across every book in the corpus.",
        "The searchable source uses the PDF logical text layer; layout spans are retained separately for provenance.",
        "",
        "| Book | PDF page | Book page | Chapter | Characters | Tokens | Spans | OCR review |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for page in samples:
        quality = page.quality
        lines.append(
            f"| {page.book_id} | {page.pdf_page} | {page.book_page or '-'} | {page.chapter_title} | "
            f"{quality['characters']} | {quality['tokens']} | {quality['span_count']} | "
            f"{'yes' if quality['needs_ocr_review'] else 'no'} |"
        )
    lines.extend(["", "## Search-text excerpts", ""])
    for page in samples:
        excerpt = page.search_text[:500].replace("\n", " ").strip()
        lines.extend(
            [
                f"### {page.book_id}: PDF page {page.pdf_page} / book page {page.book_page or '-'}",
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
            "- PDF and printed book page numbers are retained separately when a book-specific page mapping is known.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
