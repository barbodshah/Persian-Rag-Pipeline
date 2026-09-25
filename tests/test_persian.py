import unittest

from rag_retrieval.persian import normalize_search, tokenize


class PersianNormalizationTests(unittest.TestCase):
    def test_arabic_and_persian_letters_match(self):
        self.assertEqual(normalize_search("كيمي"), normalize_search("کیمی"))

    def test_digits_are_ascii_in_search_form(self):
        self.assertEqual(normalize_search("۱۲٣"), "123")

    def test_half_space_and_space_match(self):
        self.assertEqual(normalize_search("می\u200cشود"), normalize_search("می شود"))

    def test_formula_is_tokenized(self):
        self.assertIn("co2", tokenize("گاز CO2"))


if __name__ == "__main__":
    unittest.main()
