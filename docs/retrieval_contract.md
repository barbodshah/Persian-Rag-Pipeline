# Retrieval contract v1.1

`python -m rag_retrieval.cli retrieve` emits the frozen JSON interface used by downstream answer generation.

The top-level fields are:

- `schema_version`: currently `1.1`.
- `query`: original and normalized question text.
- `retrieval`: mode, limits, embedding model, fusion constant, and reranker version.
- `results`: ranked child chunks with component scores and source metadata.
- `llm_context`: de-duplicated parent passages ready to supply to an LLM.

Consumers should use `llm_context.passages` for answer generation and `results` for diagnostics. Each result and context passage identifies its `book_id`, `book_title`, and `source_file`, as well as PDF pages, printed book pages, chapter, and text. Citation and chunk IDs are namespaced by book.

`retrieval.book_id` is `null` when all corpus books were searched. When a CLI `--book` filter is supplied, it contains that book ID. `retrieval.available_books` lists the IDs present in the loaded artifacts.

## Retrieval stages

1. BM25 retrieves exact lexical candidates.
2. Dense cosine search retrieves semantic candidates.
3. Reciprocal-rank fusion combines the two ranked lists.
4. `weighted-hybrid-v1` deterministically combines normalized lexical score, dense similarity, RRF score, and query-token coverage.
5. A token-overlap diversity penalty reduces near-duplicate results.
6. Child results are expanded to unique parent passages for the LLM context.

The `retrieval.reranker` field names the final ranking strategy, not necessarily a learned model. In version 0.3.0, `weighted-hybrid-v1` is deterministic and is not a cross-encoder. The existing experimental cross-encoder provider scaffold is not connected to this contract or the active retrieval path. A learned reranker can later use a new strategy name while preserving the v1.1 response fields.
