# Current retrieval system report

## Release

- Project version: 0.3.0
- Corpus: PDFs under `Books/` (`C110210` and `C110216` in the current build)
- Scope: retrieval only; no LLM answer generation
- Active learned reranker: none
- Active final ranking strategy: `weighted-hybrid-v1`

An experimental local cross-encoder provider scaffold exists in `rag_retrieval/reranker.py`. It is not imported by the CLI or retrieval engine, is not represented in the current project dependencies, and did not produce any metric in this report.

## Active processing path

1. Discover every PDF in the corpus and assign its filename stem as `book_id`.
2. Extract logical Persian text while retaining coordinate/font spans and book provenance.
3. Normalize Persian characters, spacing, digits, and searchable symbols.
4. Build book-namespaced parent contexts and smaller child chunks.
5. Retrieve lexical candidates with Okapi BM25 and semantic candidates with cosine similarity.
6. Optionally restrict both candidate searches to one `book_id`.
7. Combine rankings with reciprocal-rank fusion and weighted scoring.
8. Apply token-overlap diversity selection.
9. Return ranked child chunks and source-aware parent passages in `llm_context`.

## Artifact inventory

| Item | Current value |
|---|---:|
| Books | 2 |
| PDF pages | 270 |
| Parent contexts | 117 |
| Child chunks / BM25 documents | 341 |
| BM25 terms | 10,116 |
| Dense-index documents | 341 |
| Dense dimensions | 1,536 |
| Embedding model | `text-embedding-3-small` |

The current dense index matches all 341 ordered chunk IDs and content fingerprints. It must be regenerated with `python -m rag_retrieval.cli embed` after a future corpus rebuild; the loader rejects stale indexes.

## Evaluation definition

The dataset contains 68 synthetic Persian questions in `data/evaluation/questions.jsonl`: 30 for `C110210` and 38 for `C110216`. Every question has a `book_id` and one or more relevant PDF pages. Evaluation restricts candidate retrieval to that book before ranking and reports aggregate and per-book metrics.

The report calls the metric `Recall@k`, but its exact definition is page-level hit rate:

```text
successful question = any top-k chunk overlaps any labeled relevant PDF page
Recall@k = successful questions / evaluated questions
```

MRR@10 uses the reciprocal rank of the first chunk whose PDF pages overlap the labels. Answer correctness, completeness, and faithfulness are not measured.

## Current book-aware BM25 results

| Scope | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| All books | 68 | 0.7794 | 0.9118 | 0.9559 | 1.0000 | 0.8530 |
| `C110210` | 30 | 0.7333 | 0.8667 | 0.9333 | 1.0000 | 0.8148 |
| `C110216` | 38 | 0.8158 | 0.9474 | 0.9737 | 1.0000 | 0.8831 |

Source: `reports/bm25_evaluation.json`.

## Current book-aware hybrid results

Source: `reports/retrieval_evaluation.json`

| Scope | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| All books | 68 | 0.7941 | 0.9412 | 0.9853 | 1.0000 | 0.8738 |
| `C110210` | 30 | 0.7333 | 0.9000 | 1.0000 | 1.0000 | 0.8361 |
| `C110216` | 38 | 0.8421 | 0.9737 | 0.9737 | 1.0000 | 0.9035 |

These results use book-scoped BM25 and dense candidates, `text-embedding-3-small`, reciprocal-rank fusion, `weighted-hybrid-v1`, and diversity selection. They are not cross-encoder results.

## Report distinctions

| Report | Meaning | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---:|---:|---:|---:|
| `bm25_evaluation.json` | Current 68-question direct BM25 | 0.7794 | 0.9118 | 0.9559 | 0.8530 |
| `retrieval_bm25_evaluation.json` | Current 68-question BM25 with weighted scoring/diversity | 0.7647 | 0.9118 | 0.9559 | 0.8481 |
| `retrieval_evaluation.json` | Current 68-question hybrid pipeline | 0.7941 | 0.9412 | 0.9853 | 0.8738 |

All three current reports use the same 68 book-labeled questions and enforce each question's `book_id` before ranking.

## Known limitations

- The test set is synthetic and not independently labeled; per-book results should be treated as regression signals rather than production-quality estimates.
- Exact-text extraction fragments some Persian words, formulas, tables, and captions.
- Table and image semantics are not reconstructed.
- Heading detection and content-type classification are heuristic.
- The current score combination is hand weighted rather than learned.
- No latency or resource-usage measurements are recorded in the current evaluation JSON.

## Next planned stage

The next planned change is an actual multilingual cross-encoder applied only to the fused candidate pool. It must be evaluated against this frozen hybrid baseline and must not be described as active until it is integrated into `RetrievalEngine`, exposed by the CLI, covered by tests, and measured in a distinct report.
