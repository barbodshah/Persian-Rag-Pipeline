"""Evaluation helpers for the stable retrieval interface."""

from __future__ import annotations

from typing import Any

from .evaluate import retrieval_metrics
from .retrieval import RetrievalEngine, RetrievalOptions


def evaluate_engine(
    engine: RetrievalEngine,
    questions: list[dict[str, Any]],
    query_vectors: list[list[float]] | None,
    mode: str = "hybrid",
) -> dict[str, Any]:
    if query_vectors is not None and len(query_vectors) != len(questions):
        raise ValueError("Question and query-vector counts differ.")
    ranks: list[int | None] = []
    ranks_by_book: dict[str, list[int | None]] = {}
    details: list[dict[str, Any]] = []
    options = RetrievalOptions(top_k=10, candidate_k=30, mode=mode)
    for index, question in enumerate(questions):
        vector = query_vectors[index] if query_vectors is not None else None
        book_id = str(question.get("book_id", "C110210"))
        response = engine.retrieve(
            question["question"],
            query_vector=vector,
            options=options,
            book_id=book_id,
        )
        relevant = set(question["relevant_pdf_pages"])
        rank = None
        for result in response["results"]:
            if relevant & set(result["source"]["pdf_pages"]):
                rank = int(result["rank"])
                break
        ranks.append(rank)
        ranks_by_book.setdefault(book_id, []).append(rank)
        details.append(
            {
                "id": question["id"],
                "book_id": book_id,
                "rank": rank,
                "top_chunk_ids": [item["chunk_id"] for item in response["results"][:5]],
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
