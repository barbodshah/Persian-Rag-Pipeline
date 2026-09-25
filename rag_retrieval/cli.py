"""Command-line entry points for build, search, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark import run_benchmark
from .bm25 import BM25Index
from .chunk import build_chunks
from .dense import DenseIndex
from .embeddings import EmbeddingConfig, EmbeddingError, MetisEmbeddingClient
from .evaluate import evaluate
from .extract import extract_book
from .hybrid_evaluate import evaluate_engine
from .io_utils import read_jsonl, write_jsonl
from .report import write_extraction_review
from .retrieval import RetrievalEngine, RetrievalOptions


def _build(args: argparse.Namespace) -> None:
    output = Path(args.output)
    pages = extract_book(Path(args.pdf))
    parents, chunks = build_chunks(
        pages,
        parent_target=args.parent_target,
        parent_max=args.parent_max,
        child_target=args.child_target,
        child_max=args.child_max,
        overlap=args.overlap,
    )
    index = BM25Index(k1=args.k1, b=args.b)
    index.build([(chunk.chunk_id, chunk.search_text) for chunk in chunks])

    write_jsonl(output / "pages.jsonl", (page.to_dict() for page in pages))
    write_jsonl(output / "parents.jsonl", parents)
    write_jsonl(output / "chunks.jsonl", (chunk.to_dict() for chunk in chunks))
    index.save(output / "bm25_index.json.gz")
    write_extraction_review(Path(args.report), pages)

    summary = {
        "pdf_pages": len(pages),
        "parents": len(parents),
        "chunks": len(chunks),
        "bm25_terms": len(index.postings),
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_search_material(index_path: Path, chunks_path: Path) -> tuple[BM25Index, dict[str, dict[str, Any]]]:
    index = BM25Index.load(index_path)
    chunks = read_jsonl(chunks_path)
    return index, {chunk["chunk_id"]: chunk for chunk in chunks}


def _load_engine(
    index_path: Path,
    chunks_path: Path,
    parents_path: Path,
    dense_path: Path | None = None,
) -> RetrievalEngine:
    bm25, chunks = _load_search_material(index_path, chunks_path)
    parents = {parent["parent_id"]: parent for parent in read_jsonl(parents_path)}
    dense = DenseIndex.load(dense_path) if dense_path is not None else None
    if dense is not None:
        dense.validate_documents([(chunk_id, chunk["text"]) for chunk_id, chunk in chunks.items()])
    return RetrievalEngine(bm25, chunks, parents, dense=dense)


def _search(args: argparse.Namespace) -> None:
    index, chunks = _load_search_material(Path(args.index), Path(args.chunks))
    results = []
    for match in index.search(args.question, top_k=args.top_k):
        chunk = chunks[match.chunk_id]
        results.append(
            {
                "rank": len(results) + 1,
                "score": match.score,
                "chunk_id": match.chunk_id,
                "parent_id": chunk["parent_id"],
                "chapter": chunk["chapter_title"],
                "pdf_pages": chunk["pdf_pages"],
                "book_pages": chunk["book_pages"],
                "content_type": chunk["content_type"],
                "text": chunk["text"],
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    index, chunks = _load_search_material(Path(args.index), Path(args.chunks))
    questions = read_jsonl(Path(args.questions))
    result = evaluate(index, chunks, questions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _embed(args: argparse.Namespace) -> None:
    config = EmbeddingConfig.from_env(Path(args.env_file) if args.env_file else None)
    provider = MetisEmbeddingClient(config)
    chunks = read_jsonl(Path(args.chunks))
    dense = DenseIndex(config.model)
    dense.build(
        [(chunk["chunk_id"], chunk["text"]) for chunk in chunks],
        provider,
        cache_path=Path(args.cache),
        batch_size=args.batch_size,
    )
    dense.save(Path(args.output))
    print(
        json.dumps(
            {
                "model": dense.model,
                "documents": len(dense.doc_ids),
                "dimension": dense.dimension,
                "output": str(Path(args.output).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _retrieve(args: argparse.Namespace) -> None:
    dense_path = Path(args.dense) if args.mode in {"dense", "hybrid"} else None
    engine = _load_engine(Path(args.index), Path(args.chunks), Path(args.parents), dense_path)
    query_vector = None
    if dense_path is not None:
        config = EmbeddingConfig.from_env(Path(args.env_file) if args.env_file else None)
        assert engine.dense is not None
        provider = MetisEmbeddingClient(config.with_model(engine.dense.model))
        query_vector = provider.embed([args.question])[0]
    response = engine.retrieve(
        args.question,
        query_vector=query_vector,
        options=RetrievalOptions(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            diversity_penalty=args.diversity_penalty,
            mode=args.mode,
        ),
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


def _evaluate_hybrid(args: argparse.Namespace) -> None:
    dense_path = Path(args.dense) if args.mode in {"dense", "hybrid"} else None
    engine = _load_engine(Path(args.index), Path(args.chunks), Path(args.parents), dense_path)
    questions = read_jsonl(Path(args.questions))
    query_vectors = None
    if dense_path is not None:
        config = EmbeddingConfig.from_env(Path(args.env_file) if args.env_file else None)
        assert engine.dense is not None
        provider = MetisEmbeddingClient(config.with_model(engine.dense.model))
        query_vectors = provider.embed([question["question"] for question in questions])
    result = evaluate_engine(engine, questions, query_vectors, mode=args.mode)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _benchmark(args: argparse.Namespace) -> None:
    if not args.confirm_api_cost:
        raise EmbeddingError("Benchmark embeds multiple chunk variants. Re-run with --confirm-api-cost.")
    config = EmbeddingConfig.from_env(Path(args.env_file) if args.env_file else None)
    targets = [int(value.strip()) for value in args.chunk_targets.split(",") if value.strip()]
    models = [value.strip() for value in (args.models or config.model).split(",") if value.strip()]
    questions = read_jsonl(Path(args.questions))
    result = run_benchmark(
        Path(args.pdf),
        questions,
        targets,
        models,
        config,
        Path(args.artifact_root),
        batch_size=args.batch_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persian textbook hybrid retrieval pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Extract, chunk, and build the BM25 index")
    build.add_argument("--pdf", default="C110210.pdf")
    build.add_argument("--output", default="artifacts")
    build.add_argument("--report", default="reports/extraction_review.md")
    build.add_argument("--k1", type=float, default=1.5)
    build.add_argument("--b", type=float, default=0.75)
    build.add_argument("--parent-target", type=int, default=700)
    build.add_argument("--parent-max", type=int, default=950)
    build.add_argument("--child-target", type=int, default=240)
    build.add_argument("--child-max", type=int, default=320)
    build.add_argument("--overlap", type=int, default=40)
    build.set_defaults(handler=_build)

    search = subparsers.add_parser("search", help="Retrieve relevant chunks")
    search.add_argument("question")
    search.add_argument("--index", default="artifacts/bm25_index.json.gz")
    search.add_argument("--chunks", default="artifacts/chunks.jsonl")
    search.add_argument("--top-k", type=int, default=5)
    search.set_defaults(handler=_search)

    evaluation = subparsers.add_parser("evaluate", help="Evaluate page-level retrieval")
    evaluation.add_argument("--index", default="artifacts/bm25_index.json.gz")
    evaluation.add_argument("--chunks", default="artifacts/chunks.jsonl")
    evaluation.add_argument("--questions", default="data/evaluation/questions.jsonl")
    evaluation.add_argument("--output", default="reports/bm25_evaluation.json")
    evaluation.set_defaults(handler=_evaluate)

    embed = subparsers.add_parser("embed", help="Build the cached Metis dense index")
    embed.add_argument("--chunks", default="artifacts/chunks.jsonl")
    embed.add_argument("--cache", default="artifacts/embedding_cache.json.gz")
    embed.add_argument("--output", default="artifacts/dense_index.json.gz")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--env-file", default=".env")
    embed.set_defaults(handler=_embed)

    retrieve = subparsers.add_parser("retrieve", help="Use the stable BM25/dense/hybrid interface")
    retrieve.add_argument("question")
    retrieve.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    retrieve.add_argument("--index", default="artifacts/bm25_index.json.gz")
    retrieve.add_argument("--dense", default="artifacts/dense_index.json.gz")
    retrieve.add_argument("--chunks", default="artifacts/chunks.jsonl")
    retrieve.add_argument("--parents", default="artifacts/parents.jsonl")
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--candidate-k", type=int, default=30)
    retrieve.add_argument("--rrf-k", type=int, default=60)
    retrieve.add_argument("--diversity-penalty", type=float, default=0.12)
    retrieve.add_argument("--env-file", default=".env")
    retrieve.set_defaults(handler=_retrieve)

    hybrid_eval = subparsers.add_parser("evaluate-retrieval", help="Evaluate the stable retrieval interface")
    hybrid_eval.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    hybrid_eval.add_argument("--index", default="artifacts/bm25_index.json.gz")
    hybrid_eval.add_argument("--dense", default="artifacts/dense_index.json.gz")
    hybrid_eval.add_argument("--chunks", default="artifacts/chunks.jsonl")
    hybrid_eval.add_argument("--parents", default="artifacts/parents.jsonl")
    hybrid_eval.add_argument("--questions", default="data/evaluation/questions.jsonl")
    hybrid_eval.add_argument("--output", default="reports/retrieval_evaluation.json")
    hybrid_eval.add_argument("--env-file", default=".env")
    hybrid_eval.set_defaults(handler=_evaluate_hybrid)

    benchmark = subparsers.add_parser("benchmark", help="Compare chunk sizes and embedding models")
    benchmark.add_argument("--pdf", default="C110210.pdf")
    benchmark.add_argument("--questions", default="data/evaluation/questions.jsonl")
    benchmark.add_argument("--chunk-targets", default="200,240,280")
    benchmark.add_argument("--models", help="Comma-separated models; defaults to METIS_EMBEDDING_MODEL")
    benchmark.add_argument("--artifact-root", default="artifacts/benchmarks")
    benchmark.add_argument("--output", default="reports/embedding_benchmark.json")
    benchmark.add_argument("--batch-size", type=int, default=32)
    benchmark.add_argument("--env-file", default=".env")
    benchmark.add_argument("--confirm-api-cost", action="store_true")
    benchmark.set_defaults(handler=_benchmark)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except EmbeddingError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
