"""Cross-encoder reranking providers and persistent score caching."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .io_utils import read_gzip_json, write_gzip_json


class RerankerError(RuntimeError):
    pass


class Reranker(Protocol):
    model_name: str

    def score(self, question: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 512
    cache_dir: str | None = None

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        return cls(
            model_name=os.environ.get("RERANKER_MODEL", cls.model_name).strip(),
            device=os.environ.get("RERANKER_DEVICE", "auto").strip(),
            batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "8")),
            max_length=int(os.environ.get("RERANKER_MAX_LENGTH", "512")),
            cache_dir=os.environ.get("RERANKER_CACHE_DIR") or None,
        )


class LocalCrossEncoderReranker:
    def __init__(self, config: RerankerConfig) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerError(
                "Cross-encoder dependencies are missing. Install the project with the reranker extra: "
                "python -m pip install -e .[reranker]"
            ) from error

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        self.model_name = config.model_name
        self.device = device
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self._model = CrossEncoder(
            config.model_name,
            device=device,
            max_length=config.max_length,
            cache_dir=config.cache_dir,
            model_kwargs=model_kwargs,
        )

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(question, passage) for passage in passages]
        values = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        flattened = values.reshape(-1).tolist()
        if len(flattened) != len(passages):
            raise RerankerError(
                f"Cross-encoder returned {len(flattened)} scores for {len(passages)} passages."
            )
        return [float(value) for value in flattened]


def _cache_key(model_name: str, question: str, passage: str) -> str:
    payload = "\x1f".join((model_name, question, passage))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedReranker:
    """Persist query-passage scores so evaluation ablations do not repeat inference."""

    def __init__(self, provider: Reranker, cache_path: Path) -> None:
        self.provider = provider
        self.model_name = provider.model_name
        self.cache_path = cache_path
        self.values: dict[str, float] = {}
        if cache_path.exists():
            value = read_gzip_json(cache_path)
            if value.get("model_name") == self.model_name:
                self.values = {key: float(score) for key, score in value.get("values", {}).items()}

    def score(self, question: str, passages: list[str]) -> list[float]:
        keys = [_cache_key(self.model_name, question, passage) for passage in passages]
        missing_indices = [index for index, key in enumerate(keys) if key not in self.values]
        if missing_indices:
            missing_passages = [passages[index] for index in missing_indices]
            missing_scores = self.provider.score(question, missing_passages)
            for index, score in zip(missing_indices, missing_scores):
                self.values[keys[index]] = float(score)
            write_gzip_json(
                self.cache_path,
                {"version": 1, "model_name": self.model_name, "values": self.values},
            )
        return [self.values[key] for key in keys]
