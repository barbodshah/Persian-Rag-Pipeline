"""Dependency-free Okapi BM25 index for the lexical baseline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_gzip_json, write_gzip_json
from .persian import tokenize


@dataclass
class SearchResult:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.average_length = 0.0
        self.postings: dict[str, list[list[int]]] = {}

    def build(self, documents: list[tuple[str, str]]) -> None:
        postings: dict[str, list[list[int]]] = defaultdict(list)
        self.doc_ids = []
        self.doc_lengths = []
        for doc_index, (doc_id, text) in enumerate(documents):
            terms = tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings[term].append([doc_index, frequency])
        self.postings = dict(postings)
        self.average_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        if not self.doc_ids or not self.average_length:
            return []
        allowed_indexes = {
            index
            for index, doc_id in enumerate(self.doc_ids)
            if allowed_doc_ids is None or doc_id in allowed_doc_ids
        }
        if not allowed_indexes:
            return []
        query_terms = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        document_count = len(allowed_indexes)
        average_length = sum(self.doc_lengths[index] for index in allowed_indexes) / document_count
        for term, query_frequency in query_terms.items():
            entries = self.postings.get(term)
            if not entries:
                continue
            filtered_entries = [entry for entry in entries if entry[0] in allowed_indexes]
            if not filtered_entries:
                continue
            document_frequency = len(filtered_entries)
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for doc_index, term_frequency in filtered_entries:
                length_ratio = self.doc_lengths[doc_index] / average_length
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                scores[doc_index] += (
                    query_frequency
                    * inverse_document_frequency
                    * (term_frequency * (self.k1 + 1.0) / denominator)
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [
            SearchResult(chunk_id=self.doc_ids[index], score=round(score, 6))
            for index, score in ranked[:top_k]
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "algorithm": "okapi-bm25",
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "average_length": self.average_length,
            "postings": self.postings,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BM25Index":
        index = cls(k1=float(value["k1"]), b=float(value["b"]))
        index.doc_ids = list(value["doc_ids"])
        index.doc_lengths = [int(item) for item in value["doc_lengths"]]
        index.average_length = float(value["average_length"])
        index.postings = value["postings"]
        return index

    def save(self, path: Path) -> None:
        write_gzip_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls.from_dict(read_gzip_json(path))
