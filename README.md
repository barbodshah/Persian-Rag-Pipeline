# Persian textbook hybrid retrieval

Version 0.2.0 implements retrieval from the Persian chemistry textbook `C110210.pdf`. It extracts and normalizes the book, creates hierarchical chunks, builds lexical and dense indexes, combines their candidates, and returns cited parent passages that can later be sent to an LLM.

Answer generation is not implemented. A true cross-encoder is also not active: the current `weighted-hybrid-v1` stage is a deterministic score combiner followed by overlap-based diversity filtering. An experimental provider scaffold exists in `rag_retrieval/reranker.py`, but it is not imported by the CLI or retrieval engine and does not affect the reported results.

## Current pipeline

```text
PDF
 → logical Persian text extraction + layout provenance
 → Persian normalization
 → parent contexts and child retrieval chunks
 → BM25 index
 → Metis dense embeddings

Question
 → BM25 candidates + dense cosine candidates
 → reciprocal-rank fusion
 → weighted score combination
 → overlap-based diversity selection
 → ranked child chunks
 → de-duplicated parent passages for future LLM context
```

The active weighted score combines normalized dense similarity, BM25, reciprocal-rank-fusion score, and query-token coverage. It is a heuristic reranker, not a learned cross-encoder.

## Current artifacts and results

The current workspace artifacts contain:

- 150 PDF pages.
- 59 parent contexts.
- 168 child chunks.
- Child chunk median of 252.5 tokens and maximum of 291 tokens.
- A 1,536-dimensional dense index for all 168 chunks.
- Embedding model: `text-embedding-3-small`.

On the 30-question evaluation set, the active hybrid pipeline achieved:

| Metric | Score | Questions found |
|---|---:|---:|
| Recall@1 | 0.7333 | 22/30 |
| Recall@3 | 0.9000 | 27/30 |
| Recall@5 | 1.0000 | 30/30 |
| Recall@10 | 1.0000 | 30/30 |
| MRR@10 | 0.8417 | - |

Here, `Recall@k` is a page-level hit rate: a question succeeds when at least one of the first `k` chunks overlaps a manually labeled relevant PDF page. It is not classical recall over every relevant chunk. The questions are synthetic and manually labeled, so these values are useful for regression testing but are not yet an independent estimate of production quality.

See [reports/current_system.md](reports/current_system.md) for the full status and distinctions between the evaluation reports.

## Setup

Use Python 3.10 or newer and install the extraction dependency:

```powershell
python -m pip install -r requirements.txt
```

No cross-encoder packages are required or used by version 0.2.0.

## Build extraction, chunks, and BM25

```powershell
python -m rag_retrieval.cli build --pdf C110210.pdf
```

The command creates:

- `artifacts/pages.jsonl`: display/search text, layout spans, coordinates, fonts, and page metadata.
- `artifacts/parents.jsonl`: larger contexts returned for later answer generation.
- `artifacts/chunks.jsonl`: child chunks used for retrieval.
- `artifacts/bm25_index.json.gz`: local lexical index.
- `reports/extraction_review.md`: representative extraction review.

The default child target is 240 tokens, with a 320-token maximum and 40-token overlap. Parameters can be changed with `--child-target`, `--child-max`, and `--overlap`. Re-run `embed` after rebuilding chunks; the loader rejects a stale dense index by content fingerprint.

The source PDF is never modified.

## Configure and build dense embeddings

Copy `.env.example` to the ignored `.env` file or export the variables directly:

```powershell
$env:METIS_API_KEY="your-real-key"
$env:METIS_BASE_URL="https://api.metisai.ir/openai/v1"
$env:METIS_EMBEDDING_MODEL="text-embedding-3-small"
```

Build the dense index:

```powershell
python -m rag_retrieval.cli embed
```

The embedding cache is written after each successful batch so interrupted builds can reuse completed vectors. API keys remain environment-only and `.env` is excluded from source control.

## Retrieve passages

Use the versioned retrieval interface:

```powershell
python -m rag_retrieval.cli retrieve "پیوند هیدروژنی در آب چگونه تشکیل می‌شود؟"
```

The default mode is `hybrid`. Explicit alternatives are available:

```powershell
python -m rag_retrieval.cli retrieve "قانون پایستگی جرم چیست؟" --mode bm25
python -m rag_retrieval.cli retrieve "قانون پایستگی جرم چیست؟" --mode dense
```

`retrieve` returns diagnostic child results plus unique parent passages under `llm_context`. The response contract is documented in [docs/retrieval_contract.md](docs/retrieval_contract.md) and [schemas/retrieval_response.schema.json](schemas/retrieval_response.schema.json).

The older `search` command performs direct BM25 lookup only and does not emit the full retrieval contract.

## Evaluation

Evaluate direct BM25:

```powershell
python -m rag_retrieval.cli evaluate
```

Evaluate the versioned retrieval interface:

```powershell
python -m rag_retrieval.cli evaluate-retrieval --mode hybrid
```

The ground truth is [data/evaluation/questions.jsonl](data/evaluation/questions.jsonl). Labels are PDF pages, so they remain usable when chunk boundaries change.

Current reports:

- `reports/bm25_evaluation.json`: direct BM25 ranking.
- `reports/retrieval_bm25_evaluation.json`: BM25 passed through the weighted/diversified retrieval interface.
- `reports/retrieval_evaluation.json`: active hybrid retrieval.

## Benchmarking

Compare child chunk sizes or embedding models:

```powershell
python -m rag_retrieval.cli benchmark --chunk-targets "200,240,280" --confirm-api-cost
```

Benchmarking requires the explicit cost flag because it may make additional embedding calls. Cached text embeddings are reused.

## Design decisions

- Logical PDF extraction is used for search because layout-mode extraction reverses Persian text in this book.
- Coordinates and font spans are retained separately for provenance and future table reconstruction.
- Display text and normalized search text are stored separately.
- Both PDF and printed book page numbers are retained; printed page 1 starts at PDF page 11.
- Full-page OCR is not used. Suspiciously sparse pages are flagged for targeted review.
- Small child chunks are ranked; larger parent contexts are returned only after selection.
- Dense query embeddings require the configured Metis-compatible endpoint; BM25 remains fully local.

## Current limitations

- Some formulas and table cells are fragmented by the PDF text layer.
- Headings are detected heuristically.
- The weighted reranker is heuristic and not a cross-encoder.
- The 30 evaluation questions were created from the textbook and need independent review and expansion.
- Table- and image-dependent questions are represented only by extracted text and captions.
