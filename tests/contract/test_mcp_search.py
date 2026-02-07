"""Contract tests for search_fomc MCP tool per contracts/mcp-tools.md."""

import pytest
from datetime import date

from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument
from src.models.enums import DocumentType
from src.vectorstore.chroma_store import ChromaStore
from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider


@pytest.fixture(scope="module")
def embedding_provider():
    return SentenceTransformerEmbeddingProvider()


@pytest.fixture
def populated_store(embedding_provider):
    """Create a ChromaDB store with test data."""
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
        raw_text="Test document about inflation and federal funds rate.",
    )

    texts = [
        "Inflation has eased over the past year but remains somewhat elevated.",
        "The Committee decided to maintain the federal funds rate at 5-1/4 to 5-1/2 percent.",
        "Consumer spending remained resilient, supported by a strong labor market.",
    ]
    embeddings = embedding_provider.embed(texts)

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_text=text,
            chunk_index=i,
            token_count=len(text) // 4,
            embedding=emb,
            section_header="Economic Outlook" if i < 2 else "Committee Policy Action",
        )
        store.add_chunks([chunk], doc)

    return store


class TestSearchFomcContract:
    """Verify search_fomc response schema matches contracts/mcp-tools.md."""

    def test_response_has_required_fields(self, populated_store, embedding_provider):
        """Each result must have: chunk_id, document_name, document_date,
        document_id, section_header, chunk_text, relevance_score."""
        query_emb = embedding_provider.embed(["inflation"])[0]
        results = populated_store.query(query_embedding=query_emb, top_k=3)

        assert len(results) > 0
        for result in results:
            # Map from ChromaStore output to contract schema
            assert "id" in result  # chunk_id
            assert "text" in result  # chunk_text
            assert "metadata" in result
            assert "distance" in result  # relevance_score (as distance)
            metadata = result["metadata"]
            assert "document_id" in metadata
            assert "document_title" in metadata  # document_name
            assert "document_date" in metadata
            assert "section_header" in metadata

    def test_relevance_score_is_valid_range(self, populated_store, embedding_provider):
        """relevance_score should be between 0 and 1 (cosine distance)."""
        query_emb = embedding_provider.embed(["federal funds rate"])[0]
        results = populated_store.query(query_embedding=query_emb, top_k=3)

        for result in results:
            # ChromaDB cosine distance is [0, 2], convert to similarity [0, 1]
            distance = result["distance"]
            assert 0 <= distance <= 2

    def test_empty_corpus_returns_empty(self, embedding_provider):
        """Empty corpus should return no results."""
        store = ChromaStore(path=":memory:")
        store._client.delete_collection("fomc_chunks")
        store._collection = store._client.get_or_create_collection(
            name="fomc_chunks", metadata={"hnsw:space": "cosine"}
        )

        query_emb = embedding_provider.embed(["inflation"])[0]
        results = store.query(query_embedding=query_emb, top_k=5)
        assert results == []

    def test_invalid_query_raises_error(self, populated_store):
        """Must provide either query_embedding or query_text."""
        with pytest.raises(ValueError):
            populated_store.query(top_k=5)

    def test_top_k_limits_results(self, populated_store, embedding_provider):
        """top_k parameter should limit the number of results."""
        query_emb = embedding_provider.embed(["economic activity"])[0]
        results = populated_store.query(query_embedding=query_emb, top_k=1)
        assert len(results) <= 1
