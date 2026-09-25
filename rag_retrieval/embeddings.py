"""OpenAI-compatible embedding client configured for Metis."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding exported variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env")) -> "EmbeddingConfig":
        if env_file is not None:
            load_env_file(env_file)
        api_key = os.environ.get("METIS_API_KEY", "").strip()
        base_url = os.environ.get("METIS_BASE_URL", "").strip()
        model = os.environ.get("METIS_EMBEDDING_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("METIS_API_KEY", api_key),
                ("METIS_BASE_URL", base_url),
                ("METIS_EMBEDDING_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise EmbeddingError(
                "Missing embedding configuration: " + ", ".join(missing) + ". "
                "Export the variables or copy .env.example to .env."
            )
        if api_key.upper() in {"MY_API_KEY", "YOUR_API_KEY", "REPLACE_ME"}:
            raise EmbeddingError("METIS_API_KEY still contains a placeholder value.")
        return cls(api_key=api_key, base_url=base_url.rstrip("/"), model=model)

    def with_model(self, model: str) -> "EmbeddingConfig":
        return replace(self, model=model)


class MetisEmbeddingClient:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model = config.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps(
            {"model": self.model, "input": texts, "encoding_format": "float"},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.base_url + "/embeddings",
            data=payload,
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                data = sorted(body.get("data", []), key=lambda item: int(item["index"]))
                vectors = [item["embedding"] for item in data]
                if len(vectors) != len(texts):
                    raise EmbeddingError(
                        f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs."
                    )
                if not vectors or any(not isinstance(vector, list) or not vector for vector in vectors):
                    raise EmbeddingError("Embedding API returned an empty or malformed vector.")
                dimension = len(vectors[0])
                if any(len(vector) != dimension for vector in vectors):
                    raise EmbeddingError("Embedding API returned inconsistent dimensions.")
                return [[float(value) for value in vector] for vector in vectors]
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                last_error = EmbeddingError(f"Embedding API HTTP {error.code}: {detail}")
                if error.code < 500 and error.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
                last_error = error
            if attempt + 1 < self.config.max_retries:
                time.sleep(min(2**attempt, 4))
        raise EmbeddingError(f"Embedding request failed after retries: {last_error}")
