"""Evaluate whether BM25 retrieves a relevant page."""

from __future__ import annotations

from typing import Any

from .bm25 import BM25Index


def evaluate(
    index: BM25Index,
    chunks_by_id: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    ranks: list[int | None] = []
    details: list[dict[str, Any]] = []
    for question in questions:
        relevant = set(question["relevant_pdf_pages"])
        results = index.search(question["question"], top_k=10)
        rank = None
        for position, result in enumerate(results, start=1):
            pages = set(chunks_by_id[result.chunk_id]["pdf_pages"])
            if pages & relevant:
                rank = position
                break
        ranks.append(rank)
        details.append(
            {
                "id": question["id"],
                "rank": rank,
                "top_chunk_ids": [result.chunk_id for result in results[:5]],
            }
        )

    total = len(ranks) or 1
    metrics: dict[str, Any] = {
        f"recall@{cutoff}": round(sum(rank is not None and rank <= cutoff for rank in ranks) / total, 4)
        for cutoff in (1, 3, 5, 10)
    }
    metrics["mrr@10"] = round(sum((1.0 / rank) if rank else 0.0 for rank in ranks) / total, 4)
    metrics["question_count"] = len(ranks)
    return {"metrics": metrics, "details": details}
