"""Local browser UI for testing the student answer pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .answering import (
    AnsweringConfig,
    AnsweringError,
    OpenAICompatibleChatClient,
    StudentAnswerPipeline,
)
from .bm25 import BM25Index
from .dense import DenseIndex
from .embeddings import EmbeddingConfig, EmbeddingError, MetisEmbeddingClient
from .io_utils import read_jsonl
from .retrieval import RetrievalEngine, RetrievalOptions


ASSET_ROOT = Path(__file__).with_name("web_assets")
MAX_REQUEST_BYTES = 16 * 1024 * 1024


def build_pipeline(
    *,
    index_path: Path,
    chunks_path: Path,
    parents_path: Path,
    dense_path: Path,
    env_file: Path,
) -> StudentAnswerPipeline:
    chunks = {item["chunk_id"]: item for item in read_jsonl(chunks_path)}
    parents = {item["parent_id"]: item for item in read_jsonl(parents_path)}
    bm25 = BM25Index.load(index_path)
    dense = DenseIndex.load(dense_path)
    dense.validate_documents([(chunk_id, chunk["text"]) for chunk_id, chunk in chunks.items()])
    engine = RetrievalEngine(bm25, chunks, parents, dense=dense)

    answer_config = AnsweringConfig.from_env(env_file)
    embedding_config = EmbeddingConfig.from_env(env_file).with_model(dense.model)
    return StudentAnswerPipeline(
        engine,
        MetisEmbeddingClient(embedding_config),
        OpenAICompatibleChatClient(answer_config),
        answer_config,
    )


def _clean_history(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value[-8:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()[:6000]})
    return cleaned


def answer_payload(pipeline: StudentAnswerPipeline, payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("message", "")
    image_data = payload.get("image_data")
    book_id = payload.get("book_id") or None
    mode = payload.get("mode", "hybrid")
    if not isinstance(text, str):
        raise AnsweringError("message must be text.")
    if image_data is not None and not isinstance(image_data, str):
        raise AnsweringError("image_data must be a data URL.")
    if book_id is not None and not isinstance(book_id, str):
        raise AnsweringError("book_id must be text.")
    if mode not in {"bm25", "dense", "hybrid"}:
        raise AnsweringError("mode must be bm25, dense, or hybrid.")
    if book_id is not None and book_id not in pipeline.engine.book_ids:
        raise AnsweringError(
            f"Unknown book_id {book_id!r}. Available books: "
            + ", ".join(pipeline.engine.book_ids)
        )
    return pipeline.answer(
        text=text,
        image_data=image_data,
        history=_clean_history(payload.get("history")),
        book_id=book_id,
        options=RetrievalOptions(mode=mode, top_k=3, candidate_k=30),
    )


def make_handler(pipeline: StudentAnswerPipeline) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        server_version = "RAGChat/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

        def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _asset(self, name: str, content_type: str) -> None:
            path = ASSET_ROOT / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._asset("index.html", "text/html; charset=utf-8")
            elif path == "/styles.css":
                self._asset("styles.css", "text/css; charset=utf-8")
            elif path == "/app.js":
                self._asset("app.js", "text/javascript; charset=utf-8")
            elif path == "/api/books":
                self._json(HTTPStatus.OK, {"books": pipeline.engine.book_ids})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/answer":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise AnsweringError("Request is empty or larger than 16 MB.")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise AnsweringError("Request body must be a JSON object.")
                self._json(HTTPStatus.OK, answer_payload(pipeline, payload))
            except (AnsweringError, EmbeddingError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # keep the local UI responsive while surfacing provider errors
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    return ChatHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local student RAG chat UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", default="artifacts/bm25_index.json.gz")
    parser.add_argument("--dense", default="artifacts/dense_index.json.gz")
    parser.add_argument("--chunks", default="artifacts/chunks.jsonl")
    parser.add_argument("--parents", default="artifacts/parents.jsonl")
    parser.add_argument("--env-file", default=".env")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = build_pipeline(
        index_path=Path(args.index),
        chunks_path=Path(args.chunks),
        parents_path=Path(args.parents),
        dense_path=Path(args.dense),
        env_file=Path(args.env_file),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline))
    print(f"Student RAG chat is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
