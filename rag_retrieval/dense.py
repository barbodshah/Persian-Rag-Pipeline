"""Cached dense-vector indexing and cosine search."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import EmbeddingProvider
from .io_utils import read_gzip_json, write_gzip_json


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return list(vector)
    return [value / magnitude for value in vector]


@dataclass
class DenseSearchResult:
    chunk_id: str
    score: float


class EmbeddingCache:
    def __init__(self, model: str, vectors: dict[str, list[float]] | None = None) -> None:
        self.model = model
        self.vectors = vectors or {}

    @classmethod
    def load(cls, path: Path, model: str) -> "EmbeddingCache":
        if not path.exists():
            return cls(model)
        value = read_gzip_json(path)
        if value.get("model") != model:
            return cls(model)
        return cls(model, {key: list(map(float, vector)) for key, vector in value["vectors"].items()})

    def save(self, path: Path) -> None:
        write_gzip_json(path, {"version": 1, "model": self.model, "vectors": self.vectors})


class DenseIndex:
    def __init__(self, model: str) -> None:
        self.model = model
        self.doc_ids: list[str] = []
        self.fingerprints: list[str] = []
        self.vectors: list[list[float]] = []
        self.dimension = 0

    def build(
        self,
        documents: list[tuple[str, str]],
        provider: EmbeddingProvider,
        cache_path: Path,
        batch_size: int = 32,
    ) -> None:
        if provider.model != self.model:
            raise ValueError(f"Provider model {provider.model!r} does not match index model {self.model!r}.")
        cache = EmbeddingCache.load(cache_path, self.model)
        fingerprints = [_fingerprint(text) for _, text in documents]
        missing = [(fingerprint, text) for fingerprint, (_, text) in zip(fingerprints, documents) if fingerprint not in cache.vectors]
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            vectors = provider.embed([text for _, text in batch])
            for (fingerprint, _), vector in zip(batch, vectors):
                cache.vectors[fingerprint] = _normalize(vector)
            cache.save(cache_path)

        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.fingerprints = fingerprints
        self.vectors = [cache.vectors[fingerprint] for fingerprint in fingerprints]
        self.dimension = len(self.vectors[0]) if self.vectors else 0
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("Cached embeddings contain inconsistent dimensions.")

    def search(self, query_vector: list[float], top_k: int = 10) -> list[DenseSearchResult]:
        if not self.vectors:
            return []
        query = _normalize(query_vector)
        if len(query) != self.dimension:
            raise ValueError(f"Query dimension {len(query)} does not match index dimension {self.dimension}.")
        scored = [
            (doc_id, sum(left * right for left, right in zip(query, vector)))
            for doc_id, vector in zip(self.doc_ids, self.vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [DenseSearchResult(chunk_id=doc_id, score=round(score, 8)) for doc_id, score in scored[:top_k]]

    def validate_documents(self, documents: list[tuple[str, str]]) -> None:
        doc_ids = [doc_id for doc_id, _ in documents]
        fingerprints = [_fingerprint(text) for _, text in documents]
        if doc_ids != self.doc_ids or fingerprints != self.fingerprints:
            raise ValueError(
                "Dense index does not match the current chunks. Re-run the embed command after rebuilding chunks."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "algorithm": "cosine-dense",
            "model": self.model,
            "dimension": self.dimension,
            "doc_ids": self.doc_ids,
            "fingerprints": self.fingerprints,
            "vectors": self.vectors,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DenseIndex":
        index = cls(str(value["model"]))
        index.dimension = int(value["dimension"])
        index.doc_ids = list(value["doc_ids"])
        index.fingerprints = list(value["fingerprints"])
        index.vectors = [[float(number) for number in vector] for vector in value["vectors"]]
        return index

    def save(self, path: Path) -> None:
        write_gzip_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        return cls.from_dict(read_gzip_json(path))
