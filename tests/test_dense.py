import tempfile
import unittest
from pathlib import Path

from rag_retrieval.dense import DenseIndex


class FakeEmbeddingProvider:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            vectors.append([float(text.count("آب")), float(text.count("اتم")), 1.0])
        return vectors


class DenseIndexTests(unittest.TestCase):
    def test_build_search_and_cache_reuse(self):
        documents = [("water", "آب آب مولکول"), ("atom", "اتم عنصر")]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache.json.gz"
            first_provider = FakeEmbeddingProvider()
            index = DenseIndex(first_provider.model)
            index.build(documents, first_provider, cache_path, batch_size=1)
            self.assertEqual(index.search([2.0, 0.0, 1.0], top_k=1)[0].chunk_id, "water")
            self.assertEqual(first_provider.calls, 2)

            second_provider = FakeEmbeddingProvider()
            rebuilt = DenseIndex(second_provider.model)
            rebuilt.build(documents, second_provider, cache_path, batch_size=1)
            self.assertEqual(second_provider.calls, 0)
            self.assertEqual(rebuilt.dimension, 3)
            rebuilt.validate_documents(documents)
            with self.assertRaisesRegex(ValueError, "does not match"):
                rebuilt.validate_documents([("water", "changed"), ("atom", "اتم عنصر")])
            filtered = rebuilt.search([2.0, 0.0, 1.0], allowed_doc_ids={"atom"})
            self.assertEqual([result.chunk_id for result in filtered], ["atom"])


if __name__ == "__main__":
    unittest.main()
