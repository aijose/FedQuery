"""Unit tests for assess_query date extraction and metadata hints."""

from unittest.mock import patch, MagicMock

from src.agent.nodes import assess_query


class TestAssessQueryDateExtraction:
    """assess_query should return needs_retrieval and optional metadata_hints."""

    @patch("src.agent.nodes.get_llm")
    def test_json_with_date_range(self, mock_get_llm):
        """LLM returns valid JSON with date range → metadata_hints populated."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": "2024-12-01", "date_end": "2024-12-31"}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "Who dissented at the December 2024 FOMC meeting?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is True
        assert result["metadata_hints"] == {
            "date_start": "2024-12-01",
            "date_end": "2024-12-31",
        }

    @patch("src.agent.nodes.get_llm")
    def test_json_with_year_range(self, mock_get_llm):
        """LLM returns valid JSON with full year range."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": "2024-01-01", "date_end": "2024-12-31"}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "What were the FOMC decisions in 2024?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is True
        assert result["metadata_hints"]["date_start"] == "2024-01-01"
        assert result["metadata_hints"]["date_end"] == "2024-12-31"

    @patch("src.agent.nodes.get_llm")
    def test_json_without_dates(self, mock_get_llm):
        """LLM returns JSON with null dates → metadata_hints is None."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": null, "date_end": null}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "How did the FOMC characterize inflation risks?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is True
        assert result["metadata_hints"] is None

    @patch("src.agent.nodes.get_llm")
    def test_json_no_retrieval(self, mock_get_llm):
        """Non-FOMC query → needs_retrieval false, no metadata_hints."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": false, "date_start": null, "date_end": null}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "Hello, how are you?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is False
        assert result["metadata_hints"] is None

    @patch("src.agent.nodes.get_llm")
    def test_bad_json_fallback_yes(self, mock_get_llm):
        """LLM returns plain 'yes' → fallback to heuristic, no metadata_hints."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="yes")
        mock_get_llm.return_value = mock_llm

        state = {"query": "What is the federal funds rate?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is True
        assert result["metadata_hints"] is None

    @patch("src.agent.nodes.get_llm")
    def test_bad_json_fallback_no(self, mock_get_llm):
        """LLM returns plain 'no' → fallback to heuristic, no metadata_hints."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="no")
        mock_get_llm.return_value = mock_llm

        state = {"query": "What is the weather?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is False
        assert result["metadata_hints"] is None

    @patch("src.agent.nodes.get_llm")
    def test_malformed_json_fallback(self, mock_get_llm):
        """LLM returns broken JSON → graceful fallback."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": "2024-12-01"'  # missing closing brace
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "December 2024 meeting?"}
        result = assess_query(state)

        # Falls back to yes/no heuristic — content contains "true"
        assert result["needs_retrieval"] is True
        assert result["metadata_hints"] is None

    @patch("src.agent.nodes.get_llm")
    def test_only_date_start_set(self, mock_get_llm):
        """If only date_start is set but date_end is null → no metadata_hints."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": "2024-12-01", "date_end": null}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test"}
        result = assess_query(state)

        assert result["needs_retrieval"] is True
        assert result["metadata_hints"] is None

    @patch("src.agent.nodes.get_llm")
    def test_code_fence_wrapped_json(self, mock_get_llm):
        """LLM wraps JSON in markdown code fences → should parse correctly."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n{"needs_retrieval": true, "date_start": "2024-12-01", "date_end": "2024-12-31"}\n```'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "Who dissented at the December 2024 FOMC meeting?"}
        result = assess_query(state)

        assert result["needs_retrieval"] is True
        assert result["metadata_hints"] == {
            "date_start": "2024-12-01",
            "date_end": "2024-12-31",
        }

    @patch("src.agent.nodes.get_llm")
    def test_top_k_hint_full_year(self, mock_get_llm):
        """Full year query → top_k_hint should be set (e.g. 20)."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": "2024-01-01", "date_end": "2024-12-31", "top_k_hint": 20}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "List the fed funds rate for each month in 2024"}
        result = assess_query(state)

        assert result["top_k_hint"] == 20

    @patch("src.agent.nodes.get_llm")
    def test_top_k_hint_null_defaults_to_none(self, mock_get_llm):
        """Narrow query → top_k_hint null → None."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": "2024-12-01", "date_end": "2024-12-31", "top_k_hint": null}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "Who dissented in December 2024?"}
        result = assess_query(state)

        assert result["top_k_hint"] is None

    @patch("src.agent.nodes.get_llm")
    def test_top_k_hint_clamped_to_50(self, mock_get_llm):
        """top_k_hint above 50 is ignored."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": null, "date_end": null, "top_k_hint": 100}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test"}
        result = assess_query(state)

        assert result["top_k_hint"] is None

    @patch("src.agent.nodes.get_llm")
    def test_top_k_hint_missing_from_json(self, mock_get_llm):
        """Old-format JSON without top_k_hint → None."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"needs_retrieval": true, "date_start": null, "date_end": null}'
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test"}
        result = assess_query(state)

        assert result["top_k_hint"] is None
