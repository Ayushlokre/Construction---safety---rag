import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api as api


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        api.UPLOAD_SESSIONS.clear()
        self.client = TestClient(api.app)

    def tearDown(self):
        api.UPLOAD_SESSIONS.clear()

    @patch.object(api, "get_collection")
    @patch.object(api.rag, "retrieve_documents", return_value="[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 1]\nA JHA is...")
    @patch.object(api.rag, "generate_answer", return_value="A JHA is a process. [Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 1]")
    def test_original_ask_returns_sources(self, _generate, _retrieve, get_collection):
        get_collection.return_value = MagicMock()
        response = self.client.post("/ask", json={"question": "What is a JHA?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"][0]["pdf_page"], 1)

    def test_blank_question_rejected(self):
        response = self.client.post("/ask", json={"question": "  "})
        self.assertEqual(response.status_code, 422)

    @patch.object(api.upload_rag, "create_session_collection")
    def test_upload_creates_temporary_session(self, create_collection):
        fake_client = MagicMock()
        fake_collection = MagicMock()
        fake_collection.name = "upload_test"
        create_collection.return_value = (fake_client, fake_collection, 3, "Construction-related content detected.")
        response = self.client.post(
            "/uploads",
            files={"file": ("site.pdf", b"%PDF-test", "application/pdf")},
        )
        self.assertEqual(response.status_code, 201)
        upload_id = response.json()["upload_id"]
        self.assertIn(upload_id, api.UPLOAD_SESSIONS)

    def test_unknown_upload_returns_404(self):
        response = self.client.post(
            "/uploads/not-a-real-id/ask",
            json={"question": "What hazards are described?"},
        )
        self.assertEqual(response.status_code, 404)

    def test_remove_upload_deletes_collection(self):
        fake_client = MagicMock()
        fake_collection = MagicMock()
        fake_collection.name = "upload_test"
        api.UPLOAD_SESSIONS["abc123"] = api.UploadSession(
            fake_client, fake_collection, "site.pdf", api.time.monotonic()
        )
        response = self.client.delete("/uploads/abc123")
        self.assertEqual(response.status_code, 200)
        fake_client.delete_collection.assert_called_once_with(name="upload_test")
        self.assertNotIn("abc123", api.UPLOAD_SESSIONS)


if __name__ == "__main__":
    unittest.main()
