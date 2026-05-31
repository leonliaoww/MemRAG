"""Smoke tests for the ingestion pipeline.

These tests require OPENAI_API_KEY to be set in the environment
or in a .env file.
"""

import os
import tempfile

import pytest

from app.core.config import settings
from app.engine.ingestion.loader import load_document
from app.engine.ingestion.splitter import split_documents


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — set it to run integration tests",
)
class TestIngestion:
    """Integration tests for document loading and splitting."""

    def test_load_txt(self) -> None:
        """Load a plain text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\n\nThird paragraph here.")
            tmp_path = f.name

        try:
            docs = load_document(tmp_path)
            assert len(docs) >= 1
            assert "test document" in docs[0].page_content.lower()
            assert docs[0].metadata["file_type"] == "txt"
        finally:
            os.unlink(tmp_path)

    def test_split_preserves_metadata(self) -> None:
        """Split should carry forward file metadata to every chunk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("A" * 5000)  # Large enough to split
            tmp_path = f.name

        try:
            docs = load_document(tmp_path)
            chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
            assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
            for chunk in chunks:
                assert "source" in chunk.metadata
                assert "chunk_index" in chunk.metadata
                assert chunk.metadata["file_type"] == "txt"
        finally:
            os.unlink(tmp_path)


class TestUnit:
    """Unit tests that don't require external services."""

    def test_config_defaults(self) -> None:
        """Verify default settings are sane."""
        assert settings.CHUNK_SIZE > 0
        assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE
        assert len(settings.SUPPORTED_EXTENSIONS) > 0
        assert settings.RETRIEVAL_TOP_K > 0

    def test_supported_extensions(self) -> None:
        """Verify supported extensions are standard formats."""
        assert ".pdf" in settings.SUPPORTED_EXTENSIONS
        assert ".txt" in settings.SUPPORTED_EXTENSIONS
        assert ".md" in settings.SUPPORTED_EXTENSIONS
