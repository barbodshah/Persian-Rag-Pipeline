import unittest

from rag_retrieval.bm25 import BM25Index
from rag_retrieval.evaluate import evaluate


class BookAwareEvaluationTests(unittest.TestCase):
    def test_page_labels_are_scoped_to_the_question_book(self):
        chunks = {
            "book-a:c1": {
                "chunk_id": "book-a:c1",
                "book_id": "book-a",
                "pdf_pages": [10],
                "search_text": "موضوع نامرتبط",
            },
            "book-b:c1": {
                "chunk_id": "book-b:c1",
                "book_id": "book-b",
                "pdf_pages": [10],
                "search_text": "پرسش هدف",
            },
        }
        index = BM25Index()
        index.build([(chunk_id, chunk["search_text"]) for chunk_id, chunk in chunks.items()])
        result = evaluate(
            index,
            chunks,
            [
                {
                    "id": "q1",
                    "book_id": "book-a",
                    "question": "پرسش هدف",
                    "relevant_pdf_pages": [10],
                },
                {
                    "id": "q2",
                    "book_id": "book-b",
                    "question": "پرسش هدف",
                    "relevant_pdf_pages": [10],
                },
            ],
        )
        self.assertIsNone(result["details"][0]["rank"])
        self.assertEqual(result["details"][1]["rank"], 1)
        self.assertEqual(result["metrics_by_book"]["book-a"]["recall@1"], 0.0)
        self.assertEqual(result["metrics_by_book"]["book-b"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
