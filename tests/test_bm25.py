import unittest

from rag_retrieval.bm25 import BM25Index


class BM25Tests(unittest.TestCase):
    def test_exact_persian_term_ranks_matching_document_first(self):
        index = BM25Index()
        index.build(
            [
                ("water", "پیوند هیدروژنی میان مولکول های آب"),
                ("atom", "عدد اتمی و جدول دوره ای عنصرها"),
            ]
        )
        results = index.search("پیوند هیدروژنی آب")
        self.assertEqual(results[0].chunk_id, "water")


if __name__ == "__main__":
    unittest.main()
