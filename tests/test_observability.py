import unittest

from fastapi.testclient import TestClient

from main import app
from observability import normalize_request_id, scrub_event


class ObservabilityTests(unittest.TestCase):
    def test_request_id_propagates_valid_and_replaces_invalid(self):
        self.assertEqual(normalize_request_id("client_req-12345"), "client_req-12345")
        generated = normalize_request_id("person@example.com")
        self.assertNotEqual(generated, "person@example.com")
        self.assertRegex(generated, r"^[A-Za-z0-9-]{36}$")

    def test_scrubber_removes_sensitive_context_and_preserves_safe_tags(self):
        event = {
            "request": {"cookies": {"accessToken": "secret"}, "data": "raw body"},
            "user": {"email": "person@example.com"},
            "breadcrumbs": [{"message": "AI prompt"}],
            "extra": {"phone": "9999999999"},
            "contexts": {"response": "AI reply"},
            "message": "private failure detail",
            "exception": {"values": [{"type": "RuntimeError", "value": "private detail"}]},
            "tags": {"service": "ai", "request_id": "client_req-12345", "studentId": "S1"},
        }

        scrubbed = scrub_event(event, {})

        for key in ("request", "user", "breadcrumbs", "extra", "contexts"):
            self.assertNotIn(key, scrubbed)
        self.assertEqual(scrubbed["message"], "Unexpected application error")
        self.assertEqual(scrubbed["exception"]["values"][0]["value"], "Unexpected application error")
        self.assertEqual(scrubbed["tags"], {"service": "ai", "request_id": "client_req-12345"})

    def test_health_response_propagates_request_id(self):
        response = TestClient(app).get("/health", headers={"X-Request-ID": "client_req-12345"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Request-ID"], "client_req-12345")


if __name__ == "__main__":
    unittest.main()
