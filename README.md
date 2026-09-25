# Persian textbook hybrid retrieval

The pipeline indexes every PDF in a corpus folder. It extracts and normalizes each book, creates source-aware hierarchical chunks, builds shared lexical and dense indexes, combines their candidates, and returns cited parent passages that identify the originating book.

Answer generation is not implemented. A true cross-encoder is also not active: the current `weighted-hybrid-v1` stage is a deterministic score combiner followed by overlap-based diversity filtering. An experimental provider scaffold exists in `rag_retrieval/reranker.py`, but it is not imported by the CLI or retrieval engine and does not affect the reported results.

## Current pipeline

```text
Books/*.pdf
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

## Current corpus and baseline results

The current corpus build contains:

- 2 books (`C110210` and `C110216`).
- 270 PDF pages.
- 117 parent contexts.
- 341 child chunks in the shared BM25 index.

The dense index currently covers all 341 chunks using `text-embedding-3-small`. On the 68-question, book-aware evaluation set, the active hybrid pipeline achieved:

| Scope | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| All books | 68 | 0.7941 | 0.9412 | 0.9853 | 1.0000 | 0.8738 |
| `C110210` | 30 | 0.7333 | 0.9000 | 1.0000 | 1.0000 | 0.8361 |
| `C110216` | 38 | 0.8421 | 0.9737 | 0.9737 | 1.0000 | 0.9035 |

Here, `Recall@k` is a book-scoped page-level hit rate: a question succeeds when at least one of the first `k` chunks from its specified book overlaps a manually labeled relevant PDF page. It is not classical recall over every relevant chunk. The questions are synthetic and manually labeled, so these values are useful for regression testing but are not yet an independent estimate of production quality.

See [reports/current_system.md](reports/current_system.md) for the full status and distinctions between the evaluation reports.

## Setup

Use Python 3.10 or newer and install the extraction dependency:

```powershell
python -m pip install -r requirements.txt
```

No cross-encoder packages are required or used by version 0.3.0.

## Add books and build the corpus

Place every knowledge-base PDF under `Books/`. Nested folders are supported, but PDF filename stems must be unique because the filename without `.pdf` is the `book_id` used by the CLI and citations.

```powershell
python -m rag_retrieval.cli build
```

The command creates:

- `artifacts/books.jsonl`: corpus manifest and per-book counts.
- `artifacts/pages.jsonl`: display/search text, layout spans, coordinates, fonts, page metadata, and book identity.
- `artifacts/parents.jsonl`: larger contexts returned for later answer generation.
- `artifacts/chunks.jsonl`: child chunks used for retrieval.
- `artifacts/bm25_index.json.gz`: local lexical index.
- `reports/extraction_review.md`: representative extraction review.

The default child target is 240 tokens, with a 320-token maximum and 40-token overlap. Parameters can be changed with `--child-target`, `--child-max`, and `--overlap`. Re-run `embed` after rebuilding chunks; the loader rejects a stale dense index by content fingerprint.

The source PDFs are never modified. `--corpus PATH` selects a different corpus. `--pdf PATH` remains available for a one-book build.

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

Omitting `--book` searches all indexed books and ranks the most relevant passages together. To limit retrieval to one book, pass its `book_id`:

```powershell
python -m rag_retrieval.cli retrieve "قانون پایستگی جرم چیست؟" --book C110210
python -m rag_retrieval.cli search "قانون پایستگی جرم چیست؟" --book C110210
```

Every result includes `book_id`, `book_title`, `source_file`, chunk/parent IDs, and PDF/printed page numbers. If a book ID is misspelled, the CLI reports the available IDs.

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

The ground truth is [data/evaluation/questions.jsonl](data/evaluation/questions.jsonl). Every question specifies a `book_id`, and labels are PDF pages within that book. Evaluation restricts candidate retrieval to that book, preventing an identically numbered page in another book from counting as a hit. Reports include both aggregate `metrics` and a `metrics_by_book` breakdown.

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
- Both PDF and printed book page numbers are retained. The known `C110210` mapping is preserved; other PDFs default their book page to the PDF page.
- Full-page OCR is not used. Suspiciously sparse pages are flagged for targeted review.
- Small child chunks are ranked; larger parent contexts are returned only after selection.
- Dense query embeddings require the configured Metis-compatible endpoint; BM25 remains fully local.

## Current limitations

- Some formulas and table cells are fragmented by the PDF text layer.
- Headings are detected heuristically.
- The weighted reranker is heuristic and not a cross-encoder.
- The 30 evaluation questions were created from the textbook and need independent review and expansion.
- Table- and image-dependent questions are represented only by extracted text and captions.
