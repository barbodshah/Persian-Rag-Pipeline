"""Chunk-size and embedding-model benchmark runner."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .bm25 import BM25Index
from .chunk import build_chunks
from .dense import DenseIndex
from .embeddings import EmbeddingConfig, EmbeddingProvider, MetisEmbeddingClient
from .extract import extract_book
from .hybrid_evaluate import evaluate_engine
from .models import PageRecord
from .retrieval import RetrievalEngine


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "model"


def run_benchmark(
    pdf_path: Path,
    questions: list[dict[str, Any]],
    targets: list[int],
    models: list[str],
    base_config: EmbeddingConfig,
    output_root: Path,
    batch_size: int = 32,
    provider_factory: Callable[[EmbeddingConfig], EmbeddingProvider] = MetisEmbeddingClient,
    pages: list[PageRecord] | None = None,
) -> dict[str, Any]:
    pages = pages if pages is not None else extract_book(pdf_path)
    report: dict[str, Any] = {
        "version": 1,
        "question_count": len(questions),
        "targets": targets,
        "models": models,
        "runs": [],
    }
    for model in models:
        provider = provider_factory(base_config.with_model(model))
        query_vectors = provider.embed([question["question"] for question in questions])
        model_root = output_root / _slug(model)
        cache_path = model_root / "embedding_cache.json.gz"
        for target in targets:
            parents, chunks = build_chunks(pages, child_target=target, child_max=320)
            chunk_dict = {chunk.chunk_id: chunk.to_dict() for chunk in chunks}
            parent_dict = {parent["parent_id"]: parent for parent in parents}
            bm25 = BM25Index()
            bm25.build([(chunk.chunk_id, chunk.search_text) for chunk in chunks])
            dense = DenseIndex(model)
            dense.build(
                [(chunk.chunk_id, chunk.text) for chunk in chunks],
                provider,
                cache_path=cache_path,
                batch_size=batch_size,
            )
            target_root = model_root / f"target-{target}"
            target_root.mkdir(parents=True, exist_ok=True)
            dense.save(target_root / "dense_index.json.gz")
            engine = RetrievalEngine(bm25, chunk_dict, parent_dict, dense=dense)
            evaluation = evaluate_engine(engine, questions, query_vectors, mode="hybrid")
            sizes = sorted(chunk.token_count for chunk in chunks)
            report["runs"].append(
                {
                    "model": model,
                    "child_target": target,
                    "chunk_count": len(chunks),
                    "median_chunk_tokens": sizes[len(sizes) // 2],
                    "max_chunk_tokens": max(sizes),
                    "metrics": evaluation["metrics"],
                }
            )
    report["runs"].sort(
        key=lambda run: (
            -run["metrics"]["recall@5"],
            -run["metrics"]["mrr@10"],
            run["median_chunk_tokens"],
        )
    )
    report["recommended"] = report["runs"][0] if report["runs"] else None
    return report
