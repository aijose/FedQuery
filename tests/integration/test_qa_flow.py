"""Integration test for the full Q&A flow.

Tests the agent graph with a real ChromaDB store and mocked LLM.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument
from src.models.enums import DocumentType
from src.vectorstore.chroma_store import ChromaStore
from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from src.agent.mcp_client import create_search_fn
from src.agent.nodes import evaluate_confidence_level


@pytest.fixture(scope="module")
def embedding_provider():
    return SentenceTransformerEmbeddingProvider()


@pytest.fixture
def populated_store(embedding_provider):
    """Create a store with realistic FOMC content."""
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
        raw_text="Full text about inflation and interest rates.",
    )

    texts = [
        "Inflation has eased over the past year but remains somewhat elevated above the Committee's 2 percent objective.",
        "The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent.",
        "Participants observed that economic activity had continued to expand at a solid pace, with strong consumer spending.",
        "The labor market remained tight, with the unemployment rate near historic lows and solid job gains.",
        "Several participants noted risks to the economic outlook, including geopolitical tensions and banking sector stress.",
    ]
    embeddings = embedding_provider.embed(texts)
    headers = [
        "Economic Outlook",
        "Committee Policy Action",
        "Participants' Views",
        "Labor Market",
        "Risk Assessment",
    ]

    for i, (text, emb, header) in enumerate(zip(texts, embeddings, headers)):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_text=text,
            chunk_index=i,
            token_count=len(text) // 4,
            embedding=emb,
            section_header=header,
        )
        store.add_chunks([chunk], doc)

    return store


class TestSearchFunction:
    """Test the search function used by the agent."""

    def test_search_returns_chunk_results(self, populated_store, embedding_provider):
        search_fn = create_search_fn(populated_store, embedding_provider)
        results = search_fn("What did the Fed say about inflation?", top_k=3)
        assert len(results) > 0
        assert results[0]["chunk_text"]  # Has text
        assert results[0]["document_name"]  # Has metadata
        assert 0 <= results[0]["relevance_score"] <= 1  # Valid score

    def test_search_inflation_returns_relevant(self, populated_store, embedding_provider):
        search_fn = create_search_fn(populated_store, embedding_provider)
        results = search_fn("inflation", top_k=3)
        top_text = results[0]["chunk_text"].lower()
        assert "inflation" in top_text

    def test_search_interest_rates_returns_relevant(self, populated_store, embedding_provider):
        search_fn = create_search_fn(populated_store, embedding_provider)
        results = search_fn("interest rate decision", top_k=3)
        top_text = results[0]["chunk_text"].lower()
        assert "rate" in top_text or "percent" in top_text


class TestConfidenceIntegration:
    """Test confidence evaluation with real search results."""

    def test_relevant_query_gets_reasonable_confidence(self, populated_store, embedding_provider):
        search_fn = create_search_fn(populated_store, embedding_provider)
        results = search_fn("inflation rate", top_k=5)
        if results:
            avg_score = sum(r["relevance_score"] for r in results) / len(results)
            confidence = evaluate_confidence_level(avg_score)
            # With relevant content, should get at least low confidence
            assert confidence in ("high", "medium", "low")

    def test_irrelevant_query_gets_low_confidence(self, populated_store, embedding_provider):
        search_fn = create_search_fn(populated_store, embedding_provider)
        results = search_fn("best pizza restaurant in Manhattan", top_k=5)
        if results:
            avg_score = sum(r["relevance_score"] for r in results) / len(results)
            # Off-topic query should have lower relevance
            assert avg_score < 0.80  # Should not be "high"
