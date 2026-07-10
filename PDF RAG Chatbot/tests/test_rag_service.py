import unittest
from unittest.mock import patch

from services.rag_service import answer_question, build_prompt_template


class DummyClient:
    def __init__(self, response: str = "Answer") -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


class RagServiceTests(unittest.TestCase):
    def test_answer_question_returns_answer_and_sources_for_relevant_chunks(self) -> None:
        with patch("services.rag_service.query_index", return_value=[("Relevant context", 0.91)]):
            result = answer_question(
                user_question="What is AI?",
                index=object(),
                chunks=["Relevant context"],
                embeddings=object(),
                ollama_client=DummyClient("AI is machine intelligence."),
                top_k=3,
            )

        self.assertEqual(result["answer"], "AI is machine intelligence.")
        self.assertEqual(result["sources"][0]["score"], 0.91)
        self.assertEqual(result["sources"][0]["chunk"], 1)

    def test_answer_question_returns_fallback_when_no_chunks_are_retrieved(self) -> None:
        with patch("services.rag_service.query_index", return_value=[]):
            result = answer_question(
                user_question="What is AI?",
                index=object(),
                chunks=["Some content"],
                embeddings=object(),
                ollama_client=DummyClient("Should not be used"),
                top_k=3,
            )

        self.assertIn("couldn't find this information", result["answer"].lower())
        self.assertEqual(result["sources"], [])

    def test_empty_question_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            answer_question(
                user_question="   ",
                index=object(),
                chunks=["Some content"],
                embeddings=object(),
                ollama_client=DummyClient(),
                top_k=3,
            )

    def test_prompt_template_contains_context_and_fallback_instruction(self) -> None:
        prompt = build_prompt_template(["Context A", "Context B"], "What is AI?")
        self.assertIn("Context:", prompt)
        self.assertIn("Context A", prompt)
        self.assertIn("I couldn't find this information", prompt)
        self.assertIn("What is AI?", prompt)


if __name__ == "__main__":
    unittest.main()
