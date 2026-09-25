"""Stable hybrid retrieval interface and context contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Index
from .dense import DenseIndex
from .persian import normalize_search, tokenize


SCHEMA_VERSION = "1.1"
RERANKER_VERSION = "weighted-hybrid-v1"


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True)
class RetrievalOptions:
    top_k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    diversity_penalty: float = 0.12
    mode: str = "hybrid"


class RetrievalEngine:
    def __init__(
        self,
        bm25: BM25Index,
        chunks: dict[str, dict[str, Any]],
        parents: dict[str, dict[str, Any]],
        dense: DenseIndex | None = None,
    ) -> None:
        self.bm25 = bm25
        self.chunks = chunks
        self.parents = parents
        self.dense = dense

    @property
    def book_ids(self) -> list[str]:
        return sorted({str(chunk.get("book_id", "C110210")) for chunk in self.chunks.values()})

    def _allowed_chunk_ids(self, book_id: str | None) -> set[str] | None:
        if book_id is None:
            return None
        allowed = {
            chunk_id
            for chunk_id, chunk in self.chunks.items()
            if str(chunk.get("book_id", "C110210")) == book_id
        }
        if not allowed:
            available = ", ".join(self.book_ids) or "(none)"
            raise ValueError(f"Unknown book_id {book_id!r}. Available books: {available}")
        return allowed

    def retrieve(
        self,
        question: str,
        query_vector: list[float] | None = None,
        options: RetrievalOptions | None = None,
        book_id: str | None = None,
    ) -> dict[str, Any]:
        options = options or RetrievalOptions()
        if options.mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("mode must be bm25, dense, or hybrid")
        needs_dense = options.mode in {"dense", "hybrid"}
        if needs_dense and self.dense is None:
            raise ValueError("Dense or hybrid mode requires a loaded dense index.")
        if needs_dense and query_vector is None:
            raise ValueError("Dense or hybrid mode requires a query embedding.")

        allowed_chunk_ids = self._allowed_chunk_ids(book_id)
        bm25_results = (
            self.bm25.search(
                question,
                top_k=options.candidate_k,
                allowed_doc_ids=allowed_chunk_ids,
            )
            if options.mode != "dense"
            else []
        )
        dense_results = (
            self.dense.search(
                query_vector or [],
                top_k=options.candidate_k,
                allowed_doc_ids=allowed_chunk_ids,
            )
            if options.mode != "bm25" and self.dense is not None
            else []
        )
        bm25_scores = {item.chunk_id: item.score for item in bm25_results}
        dense_scores = {item.chunk_id: item.score for item in dense_results}
        fused: dict[str, float] = {}
        for weight, results in ((1.0, bm25_results), (1.0, dense_results)):
            for rank, result in enumerate(results, start=1):
                fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + weight / (options.rrf_k + rank)

        bm25_normalized = _minmax(bm25_scores)
        dense_normalized = _minmax(dense_scores)
        rrf_normalized = _minmax(fused)
        query_terms = set(tokenize(question))
        candidates: list[dict[str, Any]] = []
        for chunk_id in fused:
            chunk = self.chunks[chunk_id]
            chunk_terms = set(tokenize(chunk["search_text"]))
            coverage = len(query_terms & chunk_terms) / len(query_terms) if query_terms else 0.0
            if options.mode == "hybrid":
                rerank_score = (
                    0.36 * dense_normalized.get(chunk_id, 0.0)
                    + 0.32 * bm25_normalized.get(chunk_id, 0.0)
                    + 0.22 * rrf_normalized.get(chunk_id, 0.0)
                    + 0.10 * coverage
                )
            elif options.mode == "dense":
                rerank_score = 0.85 * dense_normalized.get(chunk_id, 0.0) + 0.15 * coverage
            else:
                rerank_score = 0.85 * bm25_normalized.get(chunk_id, 0.0) + 0.15 * coverage
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "base_score": rerank_score,
                    "terms": chunk_terms,
                    "scores": {
                        "bm25": round(bm25_scores.get(chunk_id, 0.0), 8),
                        "dense_cosine": round(dense_scores.get(chunk_id, 0.0), 8),
                        "rrf": round(fused.get(chunk_id, 0.0), 8),
                        "query_coverage": round(coverage, 8),
                    },
                }
            )
        candidates.sort(key=lambda item: (-item["base_score"], item["chunk_id"]))

        selected: list[dict[str, Any]] = []
        remaining = candidates[:]
        while remaining and len(selected) < options.top_k:
            best_item = None
            best_score = float("-inf")
            for item in remaining:
                redundancy = max(
                    (_jaccard(item["terms"], chosen["terms"]) for chosen in selected),
                    default=0.0,
                )
                diversified_score = item["base_score"] - options.diversity_penalty * redundancy
                if diversified_score > best_score:
                    best_item, best_score = item, diversified_score
            assert best_item is not None
            best_item["scores"]["rerank"] = round(best_item["base_score"], 8)
            best_item["scores"]["diversified"] = round(best_score, 8)
            selected.append(best_item)
            remaining.remove(best_item)

        results: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            chunk = self.chunks[item["chunk_id"]]
            result_book_id = str(chunk.get("book_id", "C110210"))
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "parent_id": chunk["parent_id"],
                    "score": item["scores"]["diversified"],
                    "scores": item["scores"],
                    "source": {
                        "book_id": result_book_id,
                        "book_title": chunk.get("book_title", result_book_id),
                        "source_file": chunk.get("source_file", f"{result_book_id}.pdf"),
                        "document_id": result_book_id,
                        "chapter_id": chunk["chapter_id"],
                        "chapter_title": chunk["chapter_title"],
                        "section_heading": chunk.get("section_heading"),
                        "pdf_pages": chunk["pdf_pages"],
                        "book_pages": chunk["book_pages"],
                        "content_type": chunk["content_type"],
                    },
                    "text": chunk["text"],
                }
            )

        response = {
            "schema_version": SCHEMA_VERSION,
            "query": {"original": question, "normalized": normalize_search(question)},
            "retrieval": {
                "mode": options.mode,
                "top_k": options.top_k,
                "candidate_k": options.candidate_k,
                "rrf_k": options.rrf_k,
                "embedding_model": self.dense.model if self.dense and needs_dense else None,
                "reranker": RERANKER_VERSION,
                "book_id": book_id,
                "available_books": self.book_ids,
            },
            "results": results,
        }
        response["llm_context"] = self.build_llm_context(results)
        return response

    def build_llm_context(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        passages: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        for result in results:
            parent_id = result["parent_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent = self.parents[parent_id]
            book_id = str(parent.get("book_id", "C110210"))
            citation_id = parent_id if parent_id.startswith(f"{book_id}:") else f"{book_id}:{parent_id}"
            passages.append(
                {
                    "citation_id": citation_id,
                    "parent_id": parent_id,
                    "book_id": book_id,
                    "book_title": parent.get("book_title", book_id),
                    "source_file": parent.get("source_file", f"{book_id}.pdf"),
                    "chapter_title": parent["chapter_title"],
                    "pdf_pages": parent["pdf_pages"],
                    "book_pages": parent["book_pages"],
                    "text": parent["text"],
                }
            )
        return {"format": "numbered-passages-v1", "passages": passages}
