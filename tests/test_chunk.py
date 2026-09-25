import unittest

from rag_retrieval.chunk import build_chunks
from rag_retrieval.models import PageRecord
from rag_retrieval.persian import normalize_search


class ChunkingTests(unittest.TestCase):
    def test_child_chunks_respect_configured_maximum(self):
        paragraph = " ".join(["مولکول آب و پیوند هیدروژنی"] * 70)
        page = PageRecord(
            pdf_page=1,
            book_page=1,
            chapter_id="chapter-test",
            chapter_title="آزمون",
            section_heading="عنوان",
            raw_text=paragraph,
            display_text=paragraph,
            search_text=normalize_search(paragraph),
        )
        _, chunks = build_chunks([page], child_target=240, child_max=320, overlap=40)
        self.assertTrue(chunks)
        self.assertLessEqual(max(chunk.token_count for chunk in chunks), 320)
        self.assertTrue(all(chunk.chunk_id.startswith("C110210:") for chunk in chunks))
        self.assertTrue(all(chunk.book_id == "C110210" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
