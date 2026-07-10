import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app


class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]

    def embed_query(self, query):
        return [float(len(query))]


class UploadPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_upload_rebuilds_knowledge_base(self) -> None:
        pdf_path = Path(__file__).resolve().parents[1] / "uploads" / "rag_testing_document.pdf"
        with pdf_path.open("rb") as handle:
            pdf_bytes = handle.read()

        with patch("app.load_embedding_model", return_value=DummyEmbeddings()), \
             patch("app.build_faiss_index", return_value=object()), \
             patch("app.save_faiss", return_value=None), \
             patch("app.load_faiss", return_value=(object(), ["sample chunk"])):
            response = self.client.post(
                "/upload",
                files={"upload_file": ("test.pdf", pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(app.faiss_index)
        self.assertEqual(app.chunks, ["sample chunk"])


if __name__ == "__main__":
    unittest.main()
