import unittest

from rag_retrieval.answering import AnsweringError
from rag_retrieval.web import answer_payload


class FakeEngine:
    book_ids = ["book-a", "book-b"]


class FakePipeline:
    def __init__(self):
        self.engine = FakeEngine()
        self.arguments = None

    def answer(self, **kwargs):
        self.arguments = kwargs
        return {"answer": "ok"}


class WebPayloadTests(unittest.TestCase):
    def test_payload_is_forwarded_with_chat_history(self):
        pipeline = FakePipeline()
        result = answer_payload(
            pipeline,
            {
                "message": "توضیح بده",
                "book_id": "book-a",
                "mode": "bm25",
                "history": [
                    {"role": "user", "content": "پرسش اصلی"},
                    {"role": "assistant", "content": "پاسخ قبلی"},
                ],
            },
        )

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(pipeline.arguments["book_id"], "book-a")
        self.assertEqual(pipeline.arguments["options"].mode, "bm25")
        self.assertEqual(len(pipeline.arguments["history"]), 2)

    def test_unknown_book_is_rejected(self):
        with self.assertRaisesRegex(AnsweringError, "Available books"):
            answer_payload(FakePipeline(), {"message": "question", "book_id": "missing"})


if __name__ == "__main__":
    unittest.main()
