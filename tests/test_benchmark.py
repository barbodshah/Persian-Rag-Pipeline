import tempfile
import unittest
from pathlib import Path

from rag_retrieval.benchmark import run_benchmark
from rag_retrieval.embeddings import EmbeddingConfig
from rag_retrieval.models import PageRecord
from rag_retrieval.persian import normalize_search


class FakeProvider:
    def __init__(self, config):
        self.model = config.model

    def embed(self, texts):
        return [
            [float(text.count("آب")), float(text.count("اتم")), 1.0]
            for text in texts
        ]


class BenchmarkTests(unittest.TestCase):
    def test_compares_targets_without_network(self):
        page_text = "آب دارای پیوند هیدروژنی است. " * 30
        page = PageRecord(
            pdf_page=10,
            book_page=1,
            chapter_id="chapter-1",
            chapter_title="آب",
            section_heading=None,
            raw_text=page_text,
            display_text=page_text,
            search_text=normalize_search(page_text),
        )
        config = EmbeddingConfig("fake-key", "https://example.invalid/v1", "fake-model")
        questions = [
            {
                "id": "q1",
                "question": "پیوند آب چیست؟",
                "relevant_pdf_pages": [10],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = run_benchmark(
                Path("unused.pdf"),
                questions,
                [50, 80],
                ["fake-model"],
                config,
                Path(directory),
                provider_factory=FakeProvider,
                pages=[page],
            )
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(result["recommended"]["metrics"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
