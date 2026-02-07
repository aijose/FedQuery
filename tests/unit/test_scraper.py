"""Unit tests for FOMC document scraper."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date

from src.ingestion.scraper import (
    build_calendar_url,
    parse_calendar_page,
    fetch_document,
    scrape_fomc_documents,
)
from src.models.enums import DocumentType


class TestBuildCalendarUrl:
    """Test URL pattern generation for FOMC calendar pages."""

    def test_builds_url_for_recent_year(self):
        url = build_calendar_url(2024)
        assert url == "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def test_builds_url_for_2021_boundary(self):
        url = build_calendar_url(2021)
        assert url == "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def test_builds_url_for_pre_2021_year(self):
        url = build_calendar_url(2020)
        assert url == "https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm"


class TestParseCalendarPage:
    """Test HTML parsing to extract statement and minutes URLs."""

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
    <div class="panel panel-default">
        <div class="panel-heading">March 19-20, 2024</div>
        <div class="panel-body">
            <div class="fomc-meeting--month">
                <a href="/newsevents/pressreleases/monetary20240320a.htm">HTML</a>
                <a href="/monetarypolicy/fomcminutes20240320.htm">HTML</a>
            </div>
        </div>
    </div>
    </body>
    </html>
    """

    def test_extracts_statement_urls(self):
        docs = parse_calendar_page(self.SAMPLE_CALENDAR_HTML, 2024)
        statement_urls = [d["url"] for d in docs if d["type"] == DocumentType.STATEMENT]
        assert len(statement_urls) == 2
        assert any("monetary20240131a" in u for u in statement_urls)
        assert any("monetary20240320a" in u for u in statement_urls)

    def test_extracts_minutes_urls(self):
        docs = parse_calendar_page(self.SAMPLE_CALENDAR_HTML, 2024)
        minutes_urls = [d["url"] for d in docs if d["type"] == DocumentType.MINUTES]
        assert len(minutes_urls) == 2
        assert any("fomcminutes20240131" in u for u in minutes_urls)
        assert any("fomcminutes20240320" in u for u in minutes_urls)

    def test_returns_date_for_each_document(self):
        docs = parse_calendar_page(self.SAMPLE_CALENDAR_HTML, 2024)
        for doc in docs:
            assert isinstance(doc["date"], date)

    def test_handles_empty_html(self):
        docs = parse_calendar_page("<html><body></body></html>", 2024)
        assert docs == []


class TestFetchDocument:
    """Test fetching and parsing individual FOMC document pages."""

    SAMPLE_STATEMENT_HTML = """
    <html>
    <body>
    <div id="article">
        <p>Recent indicators suggest that economic activity has been expanding at a solid pace.</p>
        <p>The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent.</p>
    </div>
    </body>
    </html>
    """

    @patch("src.ingestion.scraper.requests.get")
    def test_fetches_and_parses_statement(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = self.SAMPLE_STATEMENT_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        doc = fetch_document(
            url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm",
            doc_type=DocumentType.STATEMENT,
            doc_date=date(2024, 1, 31),
        )
        assert doc is not None
        assert doc.document_type == DocumentType.STATEMENT
        assert "economic activity" in doc.raw_text
        assert doc.date == date(2024, 1, 31)

    @patch("src.ingestion.scraper.requests.get")
    def test_returns_none_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_resp

        doc = fetch_document(
            url="https://example.com/bad",
            doc_type=DocumentType.STATEMENT,
            doc_date=date(2024, 1, 31),
        )
        assert doc is None


class TestScrapeDocuments:
    """Test the top-level scrape orchestration."""

    @patch("src.ingestion.scraper.fetch_document")
    @patch("src.ingestion.scraper.requests.get")
    def test_scrape_returns_fomc_documents(self, mock_get, mock_fetch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = TestParseCalendarPage.SAMPLE_CALENDAR_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mock_doc = MagicMock()
        mock_fetch.return_value = mock_doc

        docs = scrape_fomc_documents(years=[2024])
        assert len(docs) > 0
        assert mock_fetch.call_count > 0
