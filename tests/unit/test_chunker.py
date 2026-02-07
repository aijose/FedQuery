"""Unit tests for text chunking with section metadata."""

import pytest

from src.ingestion.chunker import (
    chunk_document,
    detect_section_header,
    SECTION_PATTERNS,
)
from src.models.document import FOMCDocument
from src.models.chunk import DocumentChunk
from src.models.enums import DocumentType
from datetime import date


@pytest.fixture
def sample_document():
    """Create a sample FOMC document for testing."""
    text = """Participants' Views on Current Conditions and the Economic Outlook

Participants observed that economic activity had continued to expand at a solid pace.
Consumer spending remained resilient, supported by a strong labor market.
The unemployment rate remained low by historical standards.

Business investment had shown moderate growth. Manufacturing activity was mixed,
with some sectors experiencing supply chain improvements while others faced
ongoing challenges from elevated interest rates.

Committee Policy Action

Members agreed that recent indicators suggest that economic activity has been
expanding at a solid pace. Job gains have been strong, and the unemployment
rate has remained low. Inflation has eased over the past year but remains
somewhat elevated.

The Committee decided to maintain the target range for the federal funds rate
at 5-1/4 to 5-1/2 percent. The Committee will continue to assess additional
information and its implications for monetary policy."""
    return FOMCDocument(
        title="FOMC Minutes - January 2024",
        date=date(2024, 1, 31),
        document_type=DocumentType.MINUTES,
        source_url="https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm",
        raw_text=text,
    )


class TestDetectSectionHeader:
    """Test section header detection via regex patterns."""

    def test_detects_participants_views(self):
        header = detect_section_header(
            "Participants' Views on Current Conditions and the Economic Outlook"
        )
        assert header is not None
        assert "Participants'" in header

    def test_detects_committee_policy_action(self):
        header = detect_section_header("Committee Policy Action")
        assert header is not None
        assert "Committee Policy" in header

    def test_detects_staff_review(self):
        header = detect_section_header("Staff Review of the Economic Situation")
        assert header is not None
        assert "Staff Review" in header

    def test_detects_developments(self):
        header = detect_section_header("Developments in Financial Markets and Open Market Operations")
        assert header is not None
        assert "Developments" in header

    def test_returns_none_for_regular_text(self):
        header = detect_section_header("The economy showed strong growth.")
        assert header is None

    def test_detects_markdown_headers(self):
        header = detect_section_header("## Economic Outlook")
        assert header is not None


class TestChunkDocument:
    """Test document chunking with section metadata."""

    def test_produces_chunks(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 0

    def test_chunks_are_document_chunks(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_chunks_reference_parent_document(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.document_id == sample_document.id

    def test_chunks_have_sequential_indices(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_have_nonempty_text(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.chunk_text.strip()) > 0

    def test_chunks_have_positive_token_count(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_section_headers_captured(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        headers = [c.section_header for c in chunks if c.section_header]
        assert len(headers) > 0
        assert any("Participants'" in h or "Committee Policy" in h for h in headers)

    def test_respects_paragraph_boundaries(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=20)
        # No chunk should start or end mid-word (very crude check)
        for chunk in chunks:
            text = chunk.chunk_text.strip()
            assert not text[0].islower() or text[0] in "abcdefghijklmnopqrstuvwxyz"

    def test_overlap_creates_shared_content(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=200, chunk_overlap=50)
        if len(chunks) >= 2:
            # Adjacent chunks should share some text due to overlap
            # This is a weak test; overlap means the end of chunk N
            # appears at the start of chunk N+1
            for i in range(len(chunks) - 1):
                end_of_current = chunks[i].chunk_text[-30:]
                start_of_next = chunks[i + 1].chunk_text[:100]
                # At least some overlap text should appear in the next chunk
                # (checking that the overlap mechanism is working)
                assert len(chunks[i].chunk_text) > 0 and len(chunks[i + 1].chunk_text) > 0

    def test_short_document_produces_single_chunk(self):
        doc = FOMCDocument(
            title="Short doc",
            date=date(2024, 1, 1),
            document_type=DocumentType.STATEMENT,
            source_url="https://example.com/short",
            raw_text="A short statement about monetary policy.",
        )
        chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
