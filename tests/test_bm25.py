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

    def test_search_can_be_limited_to_allowed_documents(self):
        index = BM25Index()
        index.build([("water-a", "آب"), ("water-b", "آب دریا")])
        results = index.search("آب", allowed_doc_ids={"water-b"})
        self.assertEqual([result.chunk_id for result in results], ["water-b"])


if __name__ == "__main__":
    unittest.main()
