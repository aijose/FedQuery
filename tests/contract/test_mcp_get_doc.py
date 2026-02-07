"""Contract tests for get_document MCP tool per contracts/mcp-tools.md."""

import pytest
from datetime import date

from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument
from src.models.enums import DocumentType
from src.vectorstore.chroma_store import ChromaStore


@pytest.fixture
def store_with_document():
    """Create a store with a known document."""
    store = ChromaStore(path=":memory:")
    store._client.delete_collection("fomc_chunks")
    store._collection = store._client.get_or_create_collection(
        name="fomc_chunks", metadata={"hnsw:space": "cosine"}
    )

    doc = FOMCDocument(
        title="FOMC Minutes - January 2024",
        date=date(2024, 1, 31),
        document_type=DocumentType.MINUTES,
        source_url="https://www.federalreserve.gov/test.htm",
        raw_text="Full document text about monetary policy.",
    )

    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_text="Monetary policy text chunk.",
        chunk_index=0,
        token_count=5,
        embedding=[0.1] * 384,
        section_header="Policy Action",
    )
    store.add_chunks([chunk], doc)

    return store, doc


class TestGetDocumentContract:
    """Verify get_document response schema matches contracts/mcp-tools.md."""

    def test_returns_chunks_for_valid_document_id(self, store_with_document):
        """get_document should return chunks for a valid document_id."""
        store, doc = store_with_document
        results = store.get_document_chunks(doc.id)
        assert len(results) > 0

    def test_response_has_required_fields(self, store_with_document):
        """Response must have: id, text, metadata."""
        store, doc = store_with_document
        results = store.get_document_chunks(doc.id)
        assert len(results) > 0
        result = results[0]
        assert "id" in result
        assert "text" in result
        assert "metadata" in result
        metadata = result["metadata"]
        assert "document_id" in metadata
        assert "document_title" in metadata
        assert "document_date" in metadata
        assert "document_type" in metadata

    def test_not_found_returns_empty(self, store_with_document):
        """Nonexistent document_id should return empty results."""
        store, _ = store_with_document
        results = store.get_document_chunks("nonexistent-id-12345")
        assert results == []
