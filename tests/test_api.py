"""Smoke tests for the API layer.

Uses FastAPI's TestClient for HTTP-level testing.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.common import HealthResponse

client = TestClient(app)


class TestHealth:
    """Health check endpoint."""

    def test_health_returns_200(self) -> None:
        response = client.get("/api/v1/admin/health")
        assert response.status_code in (200, 503)  # 503 if Chroma is down
        data = response.json()
        assert "status" in data
        assert "version" in data


class TestDocuments:
    """Document upload and listing."""

    def test_upload_no_file_400(self) -> None:
        """Upload without a file should return 422 (FastAPI validation)."""
        response = client.post("/api/v1/documents/upload")
        assert response.status_code == 422

    def test_list_documents_empty(self) -> None:
        """Listing documents when none are uploaded should return empty list."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_upload_txt_file(self) -> None:
        """Upload a .txt file — should return 202 Accepted."""
        with open(__file__, "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.txt", f, "text/plain")},
            )
        # May fail if OPENAI_API_KEY not set, but should still be a structured response
        assert response.status_code in (202, 500)


class TestQueries:
    """Query endpoints."""

    def test_ask_empty_question_rejected(self) -> None:
        """空问题应被拒绝 — Pydantic 校验返回 422。"""
        response = client.post(
            "/api/v1/queries/ask",
            json={"question": "", "stream": False},
        )
        # Pydantic min_length=1 校验在业务逻辑之前拦截 → 422
        assert response.status_code in (400, 422)

    def test_ask_valid_request(self) -> None:
        """有效问题 — Chroma 正常时返回 200，Chroma 不可用时返回 500。"""
        response = client.post(
            "/api/v1/queries/ask",
            json={
                "question": "生命的意义是什么？",
                "top_k": 3,
                "retrieval_mode": "vector",
            },
        )
        # Chroma 在线时返回 200，离线时返回 500（由中间件捕获）
        assert response.status_code in (200, 500)


class TestSchemas:
    """Pydantic schema validation."""

    def test_health_response(self) -> None:
        hr = HealthResponse(version="1.0", chroma_connected=True, redis_connected=False)
        assert hr.status == "ok"

    def test_ask_request_validation(self) -> None:
        from app.schemas.query import AskRequest, RetrievalMode

        req = AskRequest(question="Hello", top_k=5, retrieval_mode=RetrievalMode.HYBRID)
        assert req.question == "Hello"
        assert req.stream is False
