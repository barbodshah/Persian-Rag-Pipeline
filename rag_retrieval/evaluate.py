"""Evaluate whether BM25 retrieves a relevant page."""

from __future__ import annotations

from typing import Any

from .bm25 import BM25Index


def retrieval_metrics(ranks: list[int | None]) -> dict[str, Any]:
    total = len(ranks) or 1
    metrics: dict[str, Any] = {
        f"recall@{cutoff}": round(
            sum(rank is not None and rank <= cutoff for rank in ranks) / total,
            4,
        )
        for cutoff in (1, 3, 5, 10)
    }
    metrics["mrr@10"] = round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / total, 4)
    metrics["question_count"] = len(ranks)
    return metrics


def evaluate(
    index: BM25Index,
    chunks_by_id: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    available_books = {
        str(chunk.get("book_id", "C110210")) for chunk in chunks_by_id.values()
    }
    ranks: list[int | None] = []
    ranks_by_book: dict[str, list[int | None]] = {}
    details: list[dict[str, Any]] = []
    for question in questions:
        book_id = str(question.get("book_id", "C110210"))
        if book_id not in available_books:
            raise ValueError(
                f"Question {question.get('id', '(unknown)')} references unknown book_id "
                f"{book_id!r}. Available books: {', '.join(sorted(available_books))}"
            )
        allowed = {
            chunk_id
            for chunk_id, chunk in chunks_by_id.items()
            if str(chunk.get("book_id", "C110210")) == book_id
        }
        relevant = set(question["relevant_pdf_pages"])
        results = index.search(question["question"], top_k=10, allowed_doc_ids=allowed)
        rank = None
        for position, result in enumerate(results, start=1):
            pages = set(chunks_by_id[result.chunk_id]["pdf_pages"])
            if pages & relevant:
                rank = position
                break
        ranks.append(rank)
        ranks_by_book.setdefault(book_id, []).append(rank)
        details.append(
            {
                "id": question["id"],
                "book_id": book_id,
                "rank": rank,
                "top_chunk_ids": [result.chunk_id for result in results[:5]],
            }
        )

    return {
        "metrics": retrieval_metrics(ranks),
        "metrics_by_book": {
            book_id: retrieval_metrics(book_ranks)
            for book_id, book_ranks in sorted(ranks_by_book.items())
        },
        "details": details,
    }
