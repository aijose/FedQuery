"""Tests for cross-encoder reranker."""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def mock_reranker():
    """Create a reranker with a mocked CrossEncoder model."""
    with patch("src.retrieval.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        MockCE.return_value = mock_model
        reranker = CrossEncoderReranker(model_name="test-model")
        yield reranker, mock_model


class TestCrossEncoderReranker:
    def test_rerank_sorts_by_score(self, mock_reranker):
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        chunks = [
            {"chunk_text": "low relevance", "id": "a"},
            {"chunk_text": "high relevance", "id": "b"},
            {"chunk_text": "medium relevance", "id": "c"},
        ]

        result = reranker.rerank("test query", chunks)

        assert len(result) == 3
        assert result[0]["id"] == "b"
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["id"] == "c"
        assert result[2]["id"] == "a"

    def test_rerank_top_k(self, mock_reranker):
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        chunks = [
            {"chunk_text": "a"},
            {"chunk_text": "b"},
            {"chunk_text": "c"},
        ]

        result = reranker.rerank("query", chunks, top_k=2)

        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9

    def test_rerank_empty_chunks(self, mock_reranker):
        reranker, mock_model = mock_reranker
        result = reranker.rerank("query", [])
        assert result == []
        mock_model.predict.assert_not_called()

    def test_rerank_preserves_original_fields(self, mock_reranker):
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = [0.8]

        chunks = [{"chunk_text": "hello", "document_date": "2024-01-31", "extra": "data"}]
        result = reranker.rerank("query", chunks)

        assert result[0]["document_date"] == "2024-01-31"
        assert result[0]["extra"] == "data"
        assert "rerank_score" in result[0]

    def test_rerank_does_not_mutate_input(self, mock_reranker):
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = [0.5]

        chunks = [{"chunk_text": "hello", "id": "1"}]
        original = dict(chunks[0])
        reranker.rerank("query", chunks)

        assert chunks[0] == original
        assert "rerank_score" not in chunks[0]

    def test_rerank_uses_text_fallback(self, mock_reranker):
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = [0.7]

        # Uses 'text' key when 'chunk_text' is missing
        chunks = [{"text": "fallback text"}]
        reranker.rerank("query", chunks)

        call_args = mock_model.predict.call_args[0][0]
        assert call_args[0] == ("query", "fallback text")

    def test_model_name_property(self, mock_reranker):
        reranker, _ = mock_reranker
        assert reranker.model_name == "test-model"

    def test_rerank_builds_correct_pairs(self, mock_reranker):
        reranker, mock_model = mock_reranker
        mock_model.predict.return_value = [0.5, 0.3]

        chunks = [
            {"chunk_text": "first chunk"},
            {"chunk_text": "second chunk"},
        ]
        reranker.rerank("my query", chunks)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [("my query", "first chunk"), ("my query", "second chunk")]
