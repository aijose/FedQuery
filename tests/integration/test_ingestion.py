"""Integration tests for the full ingestion pipeline.

These tests verify the complete flow: scrape → clean → chunk → embed → store.
Uses mocked HTTP responses to avoid hitting the real Federal Reserve website.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.models.enums import DocumentType


SAMPLE_STATEMENT_HTML = """
<html>
<body>
<div id="article">
    <p>Recent indicators suggest that economic activity has been expanding at a solid pace.
    Job gains have been strong, and the unemployment rate has remained low. Inflation has eased
    over the past year but remains somewhat elevated. The Committee decided to maintain the
    target range for the federal funds rate at 5-1/4 to 5-1/2 percent.</p>
</div>
</body>
</html>
"""

SAMPLE_MINUTES_HTML = """
<html>
<body>
<div id="article">
    <h3>Participants' Views on Current Conditions and the Economic Outlook</h3>
    <p>Participants observed that economic activity had continued to expand at a solid pace.
    Consumer spending remained resilient, supported by a strong labor market and real income gains.
    The unemployment rate remained low by historical standards.</p>

    <h3>Committee Policy Action</h3>
    <p>Members agreed that recent indicators suggest that economic activity has been expanding
    at a solid pace. The Committee decided to maintain the target range for the federal funds
    rate at 5-1/4 to 5-1/2 percent.</p>
</div>
</body>
</html>
"""

SAMPLE_CALENDAR_HTML = """
<html>
<body>
<div class="panel panel-default">
    <div class="panel-heading">January 28-29, 2024</div>
    <div class="panel-body">
        <div class="fomc-meeting--month">
            <a href="/newsevents/pressreleases/monetary20240131a.htm">HTML</a>
            <a href="/monetarypolicy/fomcminutes20240131.htm">HTML</a>
        </div>
    </div>
</div>
</body>
</html>
"""


class TestFullIngestionPipeline:
    """End-to-end integration test for the ingestion pipeline."""

    @patch("src.ingestion.scraper.requests.get")
    def test_pipeline_ingests_documents_into_chromadb(self, mock_get):
        """Full pipeline: scrape → clean → chunk → embed → store in ChromaDB."""
        from src.ingestion.pipeline import run_ingestion_pipeline
        from src.vectorstore.chroma_store import ChromaStore

        # Mock HTTP responses for calendar + documents
        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "fomccalendars" in url or "fomchistorical" in url:
                resp.text = SAMPLE_CALENDAR_HTML
            elif "pressreleases" in url:
                resp.text = SAMPLE_STATEMENT_HTML
            elif "fomcminutes" in url:
                resp.text = SAMPLE_MINUTES_HTML
            else:
                resp.text = SAMPLE_STATEMENT_HTML
            return resp

        mock_get.side_effect = side_effect

        store = ChromaStore(path=":memory:")
        result = run_ingestion_pipeline(years=[2024], store=store)

        assert result["documents_ingested"] > 0
        assert result["chunks_stored"] > 0

    @patch("src.ingestion.scraper.requests.get")
    def test_stored_chunks_have_correct_metadata(self, mock_get):
        """Verify ChromaDB collection has correct metadata fields."""
        from src.ingestion.pipeline import run_ingestion_pipeline
        from src.vectorstore.chroma_store import ChromaStore

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "fomccalendars" in url or "fomchistorical" in url:
                resp.text = SAMPLE_CALENDAR_HTML
            elif "pressreleases" in url:
                resp.text = SAMPLE_STATEMENT_HTML
            elif "fomcminutes" in url:
                resp.text = SAMPLE_MINUTES_HTML
            else:
                resp.text = SAMPLE_STATEMENT_HTML
            return resp

        mock_get.side_effect = side_effect

        store = ChromaStore(path=":memory:")
        run_ingestion_pipeline(years=[2024], store=store)

        # Query the store to verify metadata using embedding
        from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
        provider = SentenceTransformerEmbeddingProvider()
        query_emb = provider.embed(["federal funds rate"])[0]
        results = store.query(query_embedding=query_emb, top_k=1)
        assert len(results) > 0
        result = results[0]
        assert "document_id" in result["metadata"]
        assert "document_title" in result["metadata"]
        assert "document_date" in result["metadata"]
        assert "document_type" in result["metadata"]
        assert "chunk_index" in result["metadata"]

    @patch("src.ingestion.scraper.requests.get")
    def test_re_ingestion_does_not_duplicate(self, mock_get):
        """Re-ingesting the same year does not create duplicate chunks."""
        from src.ingestion.pipeline import run_ingestion_pipeline
        from src.vectorstore.chroma_store import ChromaStore

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "fomccalendars" in url or "fomchistorical" in url:
                resp.text = SAMPLE_CALENDAR_HTML
            elif "pressreleases" in url:
                resp.text = SAMPLE_STATEMENT_HTML
            elif "fomcminutes" in url:
                resp.text = SAMPLE_MINUTES_HTML
            else:
                resp.text = SAMPLE_STATEMENT_HTML
            return resp

        mock_get.side_effect = side_effect

        store = ChromaStore(path=":memory:")
        result1 = run_ingestion_pipeline(years=[2024], store=store)
        result2 = run_ingestion_pipeline(years=[2024], store=store)

        # Second run should skip already-ingested documents
        assert result2["documents_skipped"] > 0
        assert result2["chunks_stored"] == 0 or result2["documents_skipped"] == result1["documents_ingested"]
