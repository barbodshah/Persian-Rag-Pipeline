import unittest

from rag_retrieval.book_catalog import citation_label, load_book_catalog


class BookCatalogTests(unittest.TestCase):
    def test_default_catalog_has_persian_titles(self):
        catalog = load_book_catalog()
        self.assertEqual(catalog["C110210"], "کتاب شیمی دهم")
        self.assertEqual(catalog["C110216"], "کتاب زیست شناسی دهم")

    def test_citation_uses_persian_printed_book_pages(self):
        self.assertEqual(
            citation_label("کتاب شیمی دهم", [57, 56, 56]),
            "کتاب شیمی دهم، صفحه‌های ۵۶، ۵۷",
        )


if __name__ == "__main__":
    unittest.main()
