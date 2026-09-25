"""Evaluation helpers for the stable retrieval interface."""

from __future__ import annotations

from typing import Any

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
    details: list[dict[str, Any]] = []
    options = RetrievalOptions(top_k=10, candidate_k=30, mode=mode)
    for index, question in enumerate(questions):
        vector = query_vectors[index] if query_vectors is not None else None
        response = engine.retrieve(question["question"], query_vector=vector, options=options)
        relevant = set(question["relevant_pdf_pages"])
        rank = None
        for result in response["results"]:
            if relevant & set(result["source"]["pdf_pages"]):
                rank = int(result["rank"])
                break
        ranks.append(rank)
        details.append(
            {
                "id": question["id"],
                "rank": rank,
                "top_chunk_ids": [item["chunk_id"] for item in response["results"][:5]],
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
