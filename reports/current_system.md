# Current retrieval system report

## Release

- Project version: 0.2.0
- Source document: `C110210.pdf`
- Scope: retrieval only; no LLM answer generation
- Active learned reranker: none
- Active final ranking strategy: `weighted-hybrid-v1`

An experimental local cross-encoder provider scaffold exists in `rag_retrieval/reranker.py`. It is not imported by the CLI or retrieval engine, is not represented in the current project dependencies, and did not produce any metric in this report.

## Active processing path

1. Extract logical Persian text while retaining coordinate/font spans.
2. Normalize Persian characters, spacing, digits, and searchable symbols.
3. Build parent contexts and smaller child chunks.
4. Retrieve lexical candidates with Okapi BM25.
5. Retrieve semantic candidates with cosine similarity over Metis embeddings.
6. Combine rankings with reciprocal-rank fusion.
7. Calculate a deterministic weighted score from dense, BM25, RRF, and token-coverage signals.
8. Apply token-overlap diversity selection.
9. Return ranked child chunks and expand unique parents into `llm_context`.

## Artifact inventory

| Item | Current value |
|---|---:|
| PDF pages | 150 |
| Parent contexts | 59 |
| Child chunks | 168 |
| Minimum child tokens | 10 |
| Median child tokens | 252.5 |
| Maximum child tokens | 291 |
| Dense-index documents | 168 |
| Dense dimensions | 1,536 |
| Embedding model | `text-embedding-3-small` |

The dense index contains the same 168 ordered chunk IDs and content fingerprints as the current chunk artifact. The loader rejects stale dense indexes after a chunk rebuild.

## Evaluation definition

The dataset contains 30 synthetic Persian questions in `data/evaluation/questions.jsonl`. Each question is labeled with one or more relevant PDF pages.

The report calls the metric `Recall@k`, but its exact definition is page-level hit rate:

```text
successful question = any top-k chunk overlaps any labeled relevant PDF page
Recall@k = successful questions / 30
```

MRR@10 uses the reciprocal rank of the first chunk whose PDF pages overlap the labels. Answer correctness, completeness, and faithfulness are not measured.

## Current hybrid results

Source: `reports/retrieval_evaluation.json`

| Metric | Score | Count where applicable |
|---|---:|---:|
| Recall@1 | 0.7333 | 22/30 |
| Recall@3 | 0.9000 | 27/30 |
| Recall@5 | 1.0000 | 30/30 |
| Recall@10 | 1.0000 | 30/30 |
| MRR@10 | 0.8417 | - |

These are hybrid results using BM25, `text-embedding-3-small`, RRF, `weighted-hybrid-v1`, and diversity selection. They are not cross-encoder results.

## Report distinctions

| Report | Meaning | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---:|---:|---:|---:|
| `bm25_evaluation.json` | Direct BM25 order | 0.7333 | 0.8667 | 0.9333 | 0.8148 |
| `retrieval_bm25_evaluation.json` | BM25 candidates passed through weighted scoring/diversity | 0.7000 | 0.8667 | 0.9333 | 0.8037 |
| `retrieval_evaluation.json` | Active hybrid pipeline | 0.7333 | 0.9000 | 1.0000 | 0.8417 |

All three use the same 30 page-labeled questions, but they exercise different ranking paths and should not be treated as interchangeable runs.

## Known limitations

- The test set is small, synthetic, and not independently labeled; one question changes aggregate recall by 3.33 percentage points.
- Exact-text extraction fragments some Persian words, formulas, tables, and captions.
- Table and image semantics are not reconstructed.
- Heading detection and content-type classification are heuristic.
- The current score combination is hand weighted rather than learned.
- No latency or resource-usage measurements are recorded in the current evaluation JSON.

## Next planned stage

The next planned change is an actual multilingual cross-encoder applied only to the fused candidate pool. It must be evaluated against this frozen hybrid baseline and must not be described as active until it is integrated into `RetrievalEngine`, exposed by the CLI, covered by tests, and measured in a distinct report.
