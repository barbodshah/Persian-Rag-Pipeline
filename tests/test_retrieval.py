import unittest

from rag_retrieval.bm25 import BM25Index
from rag_retrieval.dense import DenseIndex
from rag_retrieval.retrieval import RetrievalEngine, RetrievalOptions


class RetrievalContractTests(unittest.TestCase):
    def setUp(self):
        self.chunks = {
            "c1": {
                "chunk_id": "c1",
                "parent_id": "p1",
                "chapter_id": "ch1",
                "chapter_title": "آب",
                "section_heading": None,
                "pdf_pages": [10],
                "book_pages": [1],
                "content_type": "prose",
                "text": "پیوند هیدروژنی در آب",
                "search_text": "پیوند هیدروژنی در آب",
                "book_id": "water-book",
                "book_title": "Water Book",
                "source_file": "water-book.pdf",
            },
            "c2": {
                "chunk_id": "c2",
                "parent_id": "p2",
                "chapter_id": "ch2",
                "chapter_title": "اتم",
                "section_heading": None,
                "pdf_pages": [20],
                "book_pages": [11],
                "content_type": "prose",
                "text": "عدد اتمی عنصر",
                "search_text": "عدد اتمی عنصر",
                "book_id": "atom-book",
                "book_title": "Atom Book",
                "source_file": "atom-book.pdf",
            },
        }
        self.parents = {
            "p1": {
                "parent_id": "p1",
                "chapter_title": "آب",
                "pdf_pages": [10],
                "book_pages": [1],
                "text": "متن کامل درباره پیوند هیدروژنی در آب",
                "book_id": "water-book",
                "book_title": "Water Book",
                "source_file": "water-book.pdf",
            },
            "p2": {
                "parent_id": "p2",
                "chapter_title": "اتم",
                "pdf_pages": [20],
                "book_pages": [11],
                "text": "متن کامل درباره عدد اتمی",
                "book_id": "atom-book",
                "book_title": "Atom Book",
                "source_file": "atom-book.pdf",
            },
        }

    def test_hybrid_response_has_frozen_contract(self):
        bm25 = BM25Index()
        bm25.build([(key, value["search_text"]) for key, value in self.chunks.items()])
        dense = DenseIndex("fake-model")
        dense.doc_ids = ["c1", "c2"]
        dense.fingerprints = ["one", "two"]
        dense.vectors = [[1.0, 0.0], [0.0, 1.0]]
        dense.dimension = 2
        engine = RetrievalEngine(bm25, self.chunks, self.parents, dense=dense)
        response = engine.retrieve(
            "پیوند آب",
            query_vector=[1.0, 0.0],
            options=RetrievalOptions(top_k=2, candidate_k=2, mode="hybrid"),
        )
        self.assertEqual(response["schema_version"], "1.1")
        self.assertEqual(response["results"][0]["chunk_id"], "c1")
        self.assertEqual(response["retrieval"]["embedding_model"], "fake-model")
        self.assertEqual(response["llm_context"]["format"], "numbered-passages-v1")
        self.assertEqual(response["results"][0]["source"]["book_id"], "water-book")
        self.assertEqual(response["llm_context"]["passages"][0]["citation_id"], "water-book:p1")

    def test_bm25_mode_does_not_require_dense_index(self):
        bm25 = BM25Index()
        bm25.build([(key, value["search_text"]) for key, value in self.chunks.items()])
        engine = RetrievalEngine(bm25, self.chunks, self.parents)
        response = engine.retrieve("عدد اتمی", options=RetrievalOptions(mode="bm25"))
        self.assertEqual(response["results"][0]["chunk_id"], "c2")
        self.assertIsNone(response["retrieval"]["embedding_model"])

    def test_book_filter_is_applied_before_ranking(self):
        bm25 = BM25Index()
        bm25.build([(key, value["search_text"]) for key, value in self.chunks.items()])
        engine = RetrievalEngine(bm25, self.chunks, self.parents)
        response = engine.retrieve(
            "پیوند آب عدد اتمی",
            options=RetrievalOptions(mode="bm25"),
            book_id="atom-book",
        )
        self.assertEqual([item["chunk_id"] for item in response["results"]], ["c2"])
        self.assertTrue(all(item["source"]["book_id"] == "atom-book" for item in response["results"]))
        self.assertEqual(response["retrieval"]["book_id"], "atom-book")

    def test_unknown_book_has_actionable_error(self):
        bm25 = BM25Index()
        bm25.build([(key, value["search_text"]) for key, value in self.chunks.items()])
        engine = RetrievalEngine(bm25, self.chunks, self.parents)
        with self.assertRaisesRegex(ValueError, "Available books"):
            engine.retrieve("آب", options=RetrievalOptions(mode="bm25"), book_id="missing")


if __name__ == "__main__":
    unittest.main()
