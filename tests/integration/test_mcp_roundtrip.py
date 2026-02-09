"""Integration test for full MCP stdio round-trip.

Spawns the MCP server as a subprocess, connects via the MCP client SDK,
calls search_fomc and get_document, and verifies the response schema
matches ChunkResult.

Requires ingested data in ChromaDB (run `fedquery ingest --year 2024` first).
"""

import pytest

from src.agent.mcp_client import MCPSearchClient


@pytest.fixture(scope="module")
def mcp_client():
    """Connect to MCP server subprocess once for the module."""
    client = MCPSearchClient()
    client.connect()
    yield client
    client.close()


class TestMCPRoundTrip:
    """Test the full MCP stdio protocol round-trip."""

    def test_search_returns_results(self, mcp_client):
        results = mcp_client.search("inflation", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_schema(self, mcp_client):
        results = mcp_client.search("federal funds rate", top_k=2)
        assert len(results) > 0

        r = results[0]
        assert "chunk_id" in r
        assert "document_name" in r
        assert "document_date" in r
        assert "document_id" in r
        assert "section_header" in r
        assert "chunk_text" in r
        assert "relevance_score" in r
        assert isinstance(r["relevance_score"], float)
        assert 0.0 <= r["relevance_score"] <= 1.0

    def test_search_respects_top_k(self, mcp_client):
        results = mcp_client.search("economic outlook", top_k=3)
        assert len(results) <= 3

    def test_search_empty_query_returns_error(self, mcp_client):
        results = mcp_client.search("", top_k=3)
        assert results == []

    def test_get_document(self, mcp_client):
        # First search to get a valid document_id
        search_results = mcp_client.search("inflation", top_k=1)
        assert len(search_results) > 0

        doc_id = search_results[0]["document_id"]
        doc = mcp_client.get_document(doc_id)
        assert isinstance(doc, dict)
        assert doc.get("id") == doc_id
        assert "title" in doc
        assert "full_text" in doc
        assert "chunk_count" in doc
        assert doc["chunk_count"] > 0

    def test_search_with_where_filter(self, mcp_client):
        """search_fomc with a date range where filter should return matching results."""
        where = {"$and": [
            {"document_date": {"$gte": "2024-12-01"}},
            {"document_date": {"$lte": "2024-12-31"}},
        ]}
        results = mcp_client.search("FOMC meeting", top_k=5, where=where)
        assert isinstance(results, list)
        # All results should be from December 2024
        for r in results:
            assert r["document_date"] >= "2024-12-01"
            assert r["document_date"] <= "2024-12-31"

    def test_search_with_where_filter_no_match(self, mcp_client):
        """where filter with future dates should return empty list."""
        where = {"$and": [
            {"document_date": {"$gte": "2099-01-01"}},
            {"document_date": {"$lte": "2099-12-31"}},
        ]}
        results = mcp_client.search("inflation", top_k=5, where=where)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_get_document_not_found(self, mcp_client):
        doc = mcp_client.get_document("nonexistent-doc-id")
        assert doc.get("error") == "not_found"
