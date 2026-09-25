# Persian textbook retrieval baseline

This repository implements a retrieval-only pipeline for `C110210.pdf`:

1. Inspect representative pages.
2. Extract and normalize Persian text while retaining layout provenance.
3. Maintain a question-to-relevant-page evaluation set.
4. Build hierarchical parent and retrieval chunks.
5. Search the chunks with an Okapi BM25 baseline.
6. Build dense embeddings through the Metis OpenAI-compatible API.
7. Fuse BM25 and dense rankings with reciprocal-rank fusion.
8. Rerank and diversify the fused candidates.
9. Benchmark chunk sizes and embedding models.
10. Emit a versioned retrieval contract ready for later LLM answer generation.

No LLM or answer generation is used yet.

## Setup

Create a Python 3.10+ environment and install the single runtime dependency:

```powershell
python -m pip install -r requirements.txt
```

## Build the artifacts

```powershell
python -m rag_retrieval.cli build --pdf C110210.pdf
```

Chunk parameters can be changed explicitly when applying a benchmark winner, for example `--child-target 240 --child-max 320 --overlap 40`. Re-run `embed` after any chunk rebuild; stale dense indexes are rejected by content fingerprint.

The build creates:

- `artifacts/pages.jsonl`: logical page text, normalized text, coordinates, fonts, and page metadata.
- `artifacts/parents.jsonl`: larger context windows.
- `artifacts/chunks.jsonl`: smaller retrieval chunks.
- `artifacts/bm25_index.json.gz`: local lexical index.
- `reports/extraction_review.md`: ten-page extraction review.

The source PDF is never modified.

## Ask a question

```powershell
python -m rag_retrieval.cli search "پیوند هیدروژنی در آب چگونه تشکیل می شود؟" --top-k 5
```

Each result includes its score, chunk and parent IDs, chapter, PDF page, printed book page, content type, and original display text.

## Configure Metis embeddings

Keep the key outside source control. Either export the variables or copy `.env.example` to the ignored `.env` file:

```powershell
$env:METIS_API_KEY="your-real-key"
$env:METIS_BASE_URL="https://api.metisai.ir/openai/v1"
$env:METIS_EMBEDDING_MODEL="text-embedding-3-small"
```

Build the cached dense index:

```powershell
python -m rag_retrieval.cli embed
```

The embedding cache is saved after every successful API batch, so interrupted builds resume without paying to embed completed chunks again.

Run hybrid retrieval:

```powershell
python -m rag_retrieval.cli retrieve "پیوند هیدروژنی در آب چگونه تشکیل می‌شود؟"
```

The versioned output contains diagnostic child results and de-duplicated parent passages under `llm_context`. The frozen interface is documented in `docs/retrieval_contract.md` and `schemas/retrieval_response.schema.json`.

## Evaluate the baseline

```powershell
python -m rag_retrieval.cli evaluate
```

The evaluation uses `data/evaluation/questions.jsonl` and reports page-level Recall@1/3/5/10 and MRR@10. The labels are intentionally stored as pages rather than generated chunk IDs so the evaluation set remains valid when chunking parameters change.
The complete evaluation output is also written to `reports/bm25_evaluation.json`.

After building the dense index, evaluate the complete hybrid pipeline:

```powershell
python -m rag_retrieval.cli evaluate-retrieval
```

Compare multiple child chunk targets or embedding models:

```powershell
python -m rag_retrieval.cli benchmark --chunk-targets "200,240,280" --confirm-api-cost
```

Benchmarking makes additional embedding calls and therefore requires the explicit cost flag. Cached text embeddings are reused across runs.

## Design decisions

- The normal PDF extraction mode is the searchable source because layout-mode extraction reverses Persian text in this book.
- Layout spans and coordinates are still retained for provenance and later reconstruction of tables and figures.
- Display text and normalized search text are stored separately.
- Printed book pages are retained alongside PDF pages; book page 1 begins at PDF page 11.
- Full-page OCR is not used. Pages with suspiciously little text are flagged for targeted review.
- The index is dependency-free apart from PDF extraction, making the baseline transparent and reproducible.

## Current limitations

- Some formulas and table cells are fragmented by the source PDF text layer.
- Headings are detected heuristically.
- The current reranker is a deterministic weighted reranker, not a cross-encoder.
- The evaluation labels should be manually reviewed and expanded as real user questions become available.
