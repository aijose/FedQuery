"""Unit tests for the text_writer module."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.text_writer import (
    build_chunks_path,
    build_html_path,
    build_text_path,
    save_document_chunks,
    save_document_html,
    save_document_text,
)
from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument
from src.models.enums import DocumentType


@pytest.fixture
def statement_doc():
    return FOMCDocument(
        title="FOMC Statement",
        date=date(2024, 1, 31),
        document_type=DocumentType.STATEMENT,
        source_url="https://example.com/statement",
        raw_text="The Federal Open Market Committee decided to maintain...",
    )


@pytest.fixture
def minutes_doc():
    return FOMCDocument(
        title="FOMC Minutes",
        date=date(2024, 1, 31),
        document_type=DocumentType.MINUTES,
        source_url="https://example.com/minutes",
        raw_text="Minutes of the Federal Open Market Committee January 30-31, 2024...",
    )


# --- T005: build_text_path produces correct paths ---

class TestBuildTextPath:
    def test_statement_path(self, statement_doc):
        """T005/T011: build_text_path produces correct path for statement."""
        base = Path("data/texts")
        result = build_text_path(statement_doc, base)
        assert result == Path("data/texts/2024/statement_2024-01-31.txt")

    def test_minutes_path(self, minutes_doc):
        """T005/T012: build_text_path produces correct path for minutes."""
        base = Path("data/texts")
        result = build_text_path(minutes_doc, base)
        assert result == Path("data/texts/2024/minutes_2024-01-31.txt")

    def test_year_directory_organization(self, statement_doc):
        """T013: build_text_path organizes files by year directory."""
        base = Path("data/texts")
        result = build_text_path(statement_doc, base)
        # Year should be the parent directory
        assert result.parent.name == "2024"
        assert result.parent.parent == base

    def test_different_year(self):
        """build_text_path uses the document's year for directory."""
        doc = FOMCDocument(
            title="FOMC Statement 2023",
            date=date(2023, 6, 14),
            document_type=DocumentType.STATEMENT,
            source_url="https://example.com/2023-statement",
            raw_text="Some FOMC text content.",
        )
        base = Path("data/texts")
        result = build_text_path(doc, base)
        assert result == Path("data/texts/2023/statement_2023-06-14.txt")


# --- T006: save_document_text writes file with correct content ---

class TestSaveDocumentText:
    def test_writes_file_with_correct_content(self, tmp_path, statement_doc):
        """T006: save_document_text writes file with correct content."""
        result = save_document_text(statement_doc, tmp_path)

        assert result is not None
        assert result.exists()
        assert result.read_text(encoding="utf-8") == statement_doc.raw_text
        assert result.name == "statement_2024-01-31.txt"

    def test_creates_year_directory(self, tmp_path, statement_doc):
        """save_document_text creates parent directories as needed."""
        result = save_document_text(statement_doc, tmp_path)

        assert result is not None
        assert (tmp_path / "2024").is_dir()

    # --- T007: save_document_text skips when file exists (no-overwrite) ---

    def test_no_overwrite_existing_file(self, tmp_path, statement_doc):
        """T007: save_document_text skips writing when file already exists."""
        # Write first time
        first_result = save_document_text(statement_doc, tmp_path)
        assert first_result is not None

        # Modify the file content to detect overwrite
        first_result.write_text("original content", encoding="utf-8")

        # Try writing again — should return None and NOT overwrite
        second_result = save_document_text(statement_doc, tmp_path)
        assert second_result is None

        # Verify original content is preserved
        assert first_result.read_text(encoding="utf-8") == "original content"

    # --- T008: save_document_text returns None and logs on OSError ---

    def test_oserror_returns_none_and_logs(self, tmp_path, statement_doc):
        """T008: save_document_text returns None and logs warning on OSError."""
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = save_document_text(statement_doc, tmp_path)
            assert result is None


# --- HTML path and save tests ---

class TestBuildHtmlPath:
    def test_statement_html_path(self, statement_doc):
        base = Path("data/html")
        result = build_html_path(statement_doc, base)
        assert result == Path("data/html/2024/statement_2024-01-31.html")

    def test_minutes_html_path(self, minutes_doc):
        base = Path("data/html")
        result = build_html_path(minutes_doc, base)
        assert result == Path("data/html/2024/minutes_2024-01-31.html")


class TestSaveDocumentHtml:
    def test_writes_html_file(self, tmp_path, statement_doc):
        statement_doc.raw_html = "<html><body>FOMC content</body></html>"
        result = save_document_html(statement_doc, tmp_path)

        assert result is not None
        assert result.exists()
        assert result.read_text(encoding="utf-8") == statement_doc.raw_html
        assert result.name == "statement_2024-01-31.html"

    def test_returns_none_when_no_raw_html(self, tmp_path, statement_doc):
        assert statement_doc.raw_html == ""
        result = save_document_html(statement_doc, tmp_path)
        assert result is None

    def test_no_overwrite_existing_html(self, tmp_path, statement_doc):
        statement_doc.raw_html = "<html>new</html>"
        first = save_document_html(statement_doc, tmp_path)
        assert first is not None

        first.write_text("original", encoding="utf-8")

        second = save_document_html(statement_doc, tmp_path)
        assert second is None
        assert first.read_text(encoding="utf-8") == "original"

    def test_oserror_returns_none(self, tmp_path, statement_doc):
        statement_doc.raw_html = "<html>content</html>"
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = save_document_html(statement_doc, tmp_path)
            assert result is None


# --- Chunks path and save tests ---

@pytest.fixture
def sample_chunks(statement_doc):
    return [
        DocumentChunk(
            document_id=statement_doc.id,
            chunk_text="The Committee decided to maintain the target range.",
            chunk_index=0,
            token_count=10,
            section_header="Federal Funds Rate Decision",
        ),
        DocumentChunk(
            document_id=statement_doc.id,
            chunk_text="Inflation has eased over the past year but remains elevated.",
            chunk_index=1,
            token_count=12,
            section_header="Economic Outlook",
        ),
    ]


class TestBuildChunksPath:
    def test_statement_chunks_path(self, statement_doc):
        base = Path("data/texts")
        result = build_chunks_path(statement_doc, base)
        assert result == Path("data/texts/2024/statement_2024-01-31.chunks.txt")

    def test_minutes_chunks_path(self, minutes_doc):
        base = Path("data/texts")
        result = build_chunks_path(minutes_doc, base)
        assert result == Path("data/texts/2024/minutes_2024-01-31.chunks.txt")


class TestSaveDocumentChunks:
    def test_writes_chunks_file(self, tmp_path, statement_doc, sample_chunks):
        result = save_document_chunks(sample_chunks, statement_doc, tmp_path)

        assert result is not None
        assert result.exists()
        assert result.name == "statement_2024-01-31.chunks.txt"

        content = result.read_text(encoding="utf-8")
        assert "CHUNK 0" in content
        assert "CHUNK 1" in content
        assert "Federal Funds Rate Decision" in content
        assert "Economic Outlook" in content
        assert "The Committee decided" in content
        assert "Inflation has eased" in content
        assert "Total chunks: 2" in content

    def test_returns_none_for_empty_chunks(self, tmp_path, statement_doc):
        result = save_document_chunks([], statement_doc, tmp_path)
        assert result is None

    def test_no_overwrite_existing_chunks(self, tmp_path, statement_doc, sample_chunks):
        first = save_document_chunks(sample_chunks, statement_doc, tmp_path)
        assert first is not None

        first.write_text("original", encoding="utf-8")

        second = save_document_chunks(sample_chunks, statement_doc, tmp_path)
        assert second is None
        assert first.read_text(encoding="utf-8") == "original"

    def test_oserror_returns_none(self, tmp_path, statement_doc, sample_chunks):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = save_document_chunks(sample_chunks, statement_doc, tmp_path)
            assert result is None
