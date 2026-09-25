# Retrieval contract v1.0

`python -m rag_retrieval.cli retrieve` emits the frozen JSON interface used by downstream answer generation.

The top-level fields are:

- `schema_version`: currently `1.0`.
- `query`: original and normalized question text.
- `retrieval`: mode, limits, embedding model, fusion constant, and reranker version.
- `results`: ranked child chunks with component scores and source metadata.
- `llm_context`: de-duplicated parent passages ready to supply to an LLM.

Consumers should use `llm_context.passages` for answer generation and `results` for diagnostics. Each context passage has a stable `citation_id`, PDF pages, printed book pages, chapter, and text. The response is validated structurally by `schemas/retrieval_response.schema.json`.

## Retrieval stages

1. BM25 retrieves exact lexical candidates.
2. Dense cosine search retrieves semantic candidates.
3. Reciprocal-rank fusion combines the two ranked lists.
4. `weighted-hybrid-v1` reranks candidates using normalized lexical score, dense similarity, RRF score, and query-token coverage.
5. A token-overlap diversity penalty reduces near-duplicate results.
6. Child results are expanded to unique parent passages for the LLM context.

The current reranker is deterministic and local. It is not a cross-encoder. A cross-encoder can be introduced later under a new reranker version without changing the v1.0 response fields.
