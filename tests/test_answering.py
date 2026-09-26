import tempfile
import unittest
import base64
from pathlib import Path

from rag_retrieval.answering import (
    AnsweringConfig,
    AnsweringError,
    StudentAnswerPipeline,
    validate_image_data_url,
    validate_structure,
)
from rag_retrieval.retrieval import RetrievalOptions


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, model, messages, *, response_format=None, temperature=0.0):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "temperature": temperature,
            }
        )
        return self.responses.pop(0)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def retrieve(self, question, query_vector=None, options=None, book_id=None):
        self.calls.append(
            {
                "question": question,
                "query_vector": query_vector,
                "options": options,
                "book_id": book_id,
            }
        )
        number = len(self.calls)
        return {
            "llm_context": {
                "passages": [
                    {
                        "citation_id": f"book:p{number}",
                        "parent_id": f"p{number}",
                        "book_id": "book",
                        "book_title": "Test Book",
                        "source_file": "book.pdf",
                        "chapter_title": f"Chapter {number}",
                        "pdf_pages": [number],
                        "book_pages": [number],
                        "text": f"Evidence {number}",
                    }
                ]
            }
        }


def config():
    return AnsweringConfig(
        api_key="test",
        base_url="https://example.invalid/v1",
        ocr_model="ocr-model",
        weak_model="weak-model",
        capable_model="capable-model",
    )


class StructureValidationTests(unittest.TestCase):
    def test_invalid_inline_image_is_rejected(self):
        with self.assertRaisesRegex(AnsweringError, "base64"):
            validate_image_data_url("data:image/png;base64,not-valid-@@")

    def test_single_question_discards_statements(self):
        value = validate_structure(
            {
                "type": "single",
                "question_text": "پیوند یونی چیست؟",
                "statements": [{"id": "S1", "text": "ignored"}],
            },
            "fallback",
        )
        self.assertEqual(value["statements"], [])

    def test_multiple_question_requires_at_least_two_statements(self):
        with self.assertRaisesRegex(AnsweringError, "at least two"):
            validate_structure(
                {
                    "type": "multiple_statements",
                    "question_text": "چند عبارت درست است؟",
                    "statements": [{"id": "S1", "text": "عبارت اول"}],
                },
                "fallback",
            )


class StudentAnswerPipelineTests(unittest.TestCase):
    def test_text_question_is_retrieved_once_and_answered(self):
        chat = FakeChat(
            [
                '{"type":"single","question_text":"آب چیست؟","statements":[]}',
                "آب یک ماده است. [book:p1]",
            ]
        )
        engine = FakeEngine()
        pipeline = StudentAnswerPipeline(engine, None, chat, config())

        result = pipeline.answer(
            text="آب چیست؟",
            book_id="book",
            options=RetrievalOptions(mode="bm25", top_k=3),
        )

        self.assertEqual([call["question"] for call in engine.calls], ["آب چیست؟"])
        self.assertEqual(engine.calls[0]["book_id"], "book")
        self.assertEqual(result["models"]["ocr"], None)
        self.assertIn("book:p1", result["answer"])
        self.assertEqual([call["model"] for call in chat.calls], ["weak-model", "capable-model"])
        final_prompt = chat.calls[-1]["messages"][-1]["content"]
        self.assertIn("[Test Book، صفحه ۱]", final_prompt)
        self.assertNotIn("book:p1", final_prompt)

    def test_screenshot_ocr_and_multiple_statements_are_retrieved_separately(self):
        chat = FakeChat(
            [
                "چند عبارت درست است؟ الف) آب مایع است. ب) آهن گاز است.",
                (
                    '{"type":"multiple_statements","question_text":"چند عبارت درست است؟",'
                    '"statements":[{"id":"S1","text":"آب مایع است."},'
                    '{"id":"S2","text":"آهن گاز است."}]}'
                ),
                "یک عبارت درست است. [book:p1] [book:p2]",
            ]
        )
        engine = FakeEngine()
        pipeline = StudentAnswerPipeline(engine, None, chat, config())

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "question.png"
            image.write_bytes(b"not-a-real-image-needed-for-unit-test")
            result = pipeline.answer(
                image_path=image,
                options=RetrievalOptions(mode="bm25", top_k=3),
            )

        self.assertEqual(
            [call["question"] for call in engine.calls],
            ["آب مایع است.", "آهن گاز است."],
        )
        self.assertEqual(result["structure"]["type"], "multiple_statements")
        self.assertEqual([call["model"] for call in chat.calls], [
            "ocr-model",
            "weak-model",
            "capable-model",
        ])
        final_content = chat.calls[-1]["messages"][-1]["content"]
        self.assertIsInstance(final_content, list)
        self.assertEqual(final_content[-1]["type"], "image_url")

    def test_inline_screenshot_is_sent_to_ocr_and_final_model(self):
        chat = FakeChat(
            [
                "آب چیست؟",
                '{"type":"single","question_text":"آب چیست؟","statements":[]}',
                "پاسخ",
            ]
        )
        engine = FakeEngine()
        pipeline = StudentAnswerPipeline(engine, None, chat, config())
        encoded = base64.b64encode(b"test-image").decode("ascii")

        result = pipeline.answer(
            image_data=f"data:image/png;base64,{encoded}",
            options=RetrievalOptions(mode="bm25"),
        )

        self.assertEqual(result["input"]["image"], "inline-upload")
        self.assertEqual(result["models"]["ocr"], "ocr-model")
        self.assertEqual(chat.calls[0]["messages"][-1]["content"][-1]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
