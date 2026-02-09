"""Tests for chunking grid evaluation."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.chunking_eval import (
    load_documents_from_text_dir,
    evaluate_chunking_grid,
)
from src.models.document import FOMCDocument
from src.models.enums import DocumentType


@pytest.fixture
def text_dir(tmp_path):
    """Create a temp directory with sample text files."""
    year_dir = tmp_path / "2024"
    year_dir.mkdir()

    (year_dir / "statement_2024-01-31.txt").write_text(
        "The Committee decided to maintain the target range for the "
        "federal funds rate at 5-1/4 to 5-1/2 percent. "
        "Inflation has eased over the past year but remains elevated."
    )
    (year_dir / "minutes_2024-01-31.txt").write_text(
        "Minutes of the Federal Open Market Committee January 30-31, 2024. "
        "Participants discussed the economic outlook and risks."
    )
    # Should be skipped (chunks file)
    (year_dir / "statement_2024-01-31.chunks.txt").write_text("chunk data")
    # Should be skipped (not matching pattern)
    (year_dir / "notes.txt").write_text("random notes")

    return tmp_path


@pytest.fixture
def golden_qa_path(tmp_path):
    """Create a minimal golden QA dataset."""
    qa_data = [
        {
            "id": "test-01",
            "question": "What was the rate?",
            "category": "factual",
            "expected_answer_keywords": ["5-1/4"],
            "relevant_documents": [{"type": "statement", "date": "2024-01-31"}],
            "relevant_sections": [],
            "relevant_text_fragments": ["5-1/4 to 5-1/2"],
            "difficulty": "easy",
        },
    ]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(qa_data))
    return path


class TestLoadDocumentsFromTextDir:
    def test_loads_matching_files(self, text_dir):
        docs = load_documents_from_text_dir(text_dir)
        assert len(docs) == 2

    def test_correct_types(self, text_dir):
        docs = load_documents_from_text_dir(text_dir)
        types = {d.document_type for d in docs}
        assert types == {DocumentType.STATEMENT, DocumentType.MINUTES}

    def test_correct_dates(self, text_dir):
        docs = load_documents_from_text_dir(text_dir)
        dates = {d.date for d in docs}
        assert date(2024, 1, 31) in dates

    def test_skips_chunks_files(self, text_dir):
        docs = load_documents_from_text_dir(text_dir)
        # Should not include .chunks.txt
        assert all("chunk data" not in d.raw_text for d in docs)

    def test_empty_dir(self, tmp_path):
        docs = load_documents_from_text_dir(tmp_path / "nonexistent")
        assert docs == []


class TestEvaluateChunkingGrid:
    def test_runs_grid(self, text_dir, golden_qa_path):
        docs = load_documents_from_text_dir(text_dir)

        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384] * 100  # enough for any number of chunks
        mock_provider.embed_query.return_value = [[0.1] * 384]

        grid = [
            {"chunk_size": 256, "chunk_overlap": 25},
            {"chunk_size": 512, "chunk_overlap": 50},
        ]

        reports = evaluate_chunking_grid(
            documents=docs,
            golden_path=golden_qa_path,
            grid=grid,
            embedding_provider=mock_provider,
            top_k_values=[3],
        )

        assert len(reports) == 2
        assert reports[0].config_label == "chunk_256_overlap_25"
        assert reports[1].config_label == "chunk_512_overlap_50"

    def test_report_contains_parameters(self, text_dir, golden_qa_path):
        docs = load_documents_from_text_dir(text_dir)
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[0.1] * 384] * 100
        mock_provider.embed_query.return_value = [[0.1] * 384]

        grid = [{"chunk_size": 512, "chunk_overlap": 50}]

        reports = evaluate_chunking_grid(
            documents=docs,
            golden_path=golden_qa_path,
            grid=grid,
            embedding_provider=mock_provider,
            top_k_values=[3],
        )

        params = reports[0].parameters
        assert params["chunk_size"] == 512
        assert params["chunk_overlap"] == 50
        assert params["num_documents"] == 2
        assert params["num_chunks"] > 0
