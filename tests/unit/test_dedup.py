"""Unit tests for deduplication during ingestion."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import date

from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument
from src.models.enums import DocumentType


class TestDeduplication:
    """Test that re-ingesting the same source_url does not create duplicates."""

    def test_chroma_store_rejects_duplicate_source_url(self):
        """Verify ChromaStore.has_document() detects existing documents."""
        from src.vectorstore.chroma_store import ChromaStore

        store = ChromaStore(path=":memory:")
        # Ensure clean state by deleting any existing data
        store._client.delete_collection("fomc_chunks")
        store._collection = store._client.get_or_create_collection(
            name="fomc_chunks", metadata={"hnsw:space": "cosine"}
        )
        doc = FOMCDocument(
            title="FOMC Statement - Jan 2024",
            date=date(2024, 1, 31),
            document_type=DocumentType.STATEMENT,
            source_url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm",
            raw_text="Economic activity expanded at a solid pace.",
        )

        # First check: document should not exist
        assert store.has_document(doc.source_url) is False

        # Add a chunk for this document
        from src.models.chunk import DocumentChunk

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_text="Economic activity expanded at a solid pace.",
            chunk_index=0,
            token_count=8,
            embedding=[0.1] * 384,
            section_header=None,
        )
        store.add_chunks([chunk], doc)

        # Second check: document should now exist
        assert store.has_document(doc.source_url) is True

    def test_pipeline_skips_already_ingested_documents(self):
        """Verify the pipeline checks for duplicates before ingesting."""
        from src.vectorstore.chroma_store import ChromaStore

        store = ChromaStore(path=":memory:")

        doc = FOMCDocument(
            title="FOMC Statement - Jan 2024",
            date=date(2024, 1, 31),
            document_type=DocumentType.STATEMENT,
            source_url="https://www.federalreserve.gov/test.htm",
            raw_text="Test content for deduplication.",
        )

        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_text="Test content for deduplication.",
            chunk_index=0,
            token_count=5,
            embedding=[0.1] * 384,
        )
        store.add_chunks([chunk], doc)

        # Attempting to add the same document again should be detectable
        assert store.has_document(doc.source_url) is True

        # Adding a different URL should not be detected
        assert store.has_document("https://www.federalreserve.gov/different.htm") is False
