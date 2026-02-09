"""Unit tests for two-pass retrieval in search_corpus."""

from src.agent.nodes import search_corpus


def _make_chunk(chunk_id, document_date="2024-12-18", relevance_score=0.7):
    return {
        "chunk_id": chunk_id,
        "document_name": f"Doc-{chunk_id}",
        "document_date": document_date,
        "document_id": f"doc-{chunk_id}",
        "section_header": "Section",
        "chunk_text": f"Text for {chunk_id}",
        "relevance_score": relevance_score,
    }


class TestTwoPassRetrieval:
    """search_corpus should use two-pass when metadata_hints contain dates."""

    def test_no_hints_single_pass(self):
        """No metadata_hints → single unfiltered search."""
        calls = []

        def mock_search(query, top_k=5, where=None):
            calls.append({"query": query, "top_k": top_k, "where": where})
            return [_make_chunk("a"), _make_chunk("b")]

        state = {"query": "inflation risks", "metadata_hints": None}
        result = search_corpus(state, mock_search)

        assert len(calls) == 1
        assert calls[0]["where"] is None
        assert len(result["retrieved_chunks"]) == 2

    def test_hints_without_date_start_single_pass(self):
        """metadata_hints exists but no date_start → single pass."""
        calls = []

        def mock_search(query, top_k=5, where=None):
            calls.append({"query": query, "top_k": top_k, "where": where})
            return [_make_chunk("a")]

        state = {"query": "test", "metadata_hints": {"date_start": None, "date_end": None}}
        result = search_corpus(state, mock_search)

        assert len(calls) == 1
        assert calls[0]["where"] is None

    def test_two_pass_with_date_hints(self):
        """Date hints present → two search calls (filtered + unfiltered)."""
        calls = []

        def mock_search(query, top_k=5, where=None):
            calls.append({"query": query, "top_k": top_k, "where": where})
            if where is not None:
                return [_make_chunk("dec-1"), _make_chunk("dec-2")]
            else:
                return [_make_chunk("any-1"), _make_chunk("any-2")]

        state = {
            "query": "December 2024 dissent",
            "metadata_hints": {"date_start": "2024-12-01", "date_end": "2024-12-31"},
        }
        result = search_corpus(state, mock_search)

        assert len(calls) == 2
        # First call should have where filter
        assert calls[0]["where"] is not None
        assert calls[0]["where"]["$and"][0]["document_date"]["$gte"] == "2024-12-01"
        assert calls[0]["where"]["$and"][1]["document_date"]["$lte"] == "2024-12-31"
        # Second call is unfiltered
        assert calls[1]["where"] is None

        # Filtered chunks first, then unfiltered
        ids = [r["chunk_id"] for r in result["retrieved_chunks"]]
        assert ids == ["dec-1", "dec-2", "any-1", "any-2"]

    def test_two_pass_dedup(self):
        """Overlapping results between passes should be deduped."""
        def mock_search(query, top_k=5, where=None):
            if where is not None:
                return [_make_chunk("shared"), _make_chunk("filtered-only")]
            else:
                return [_make_chunk("shared"), _make_chunk("unfiltered-only")]

        state = {
            "query": "test",
            "metadata_hints": {"date_start": "2024-12-01", "date_end": "2024-12-31"},
        }
        result = search_corpus(state, mock_search)

        ids = [r["chunk_id"] for r in result["retrieved_chunks"]]
        assert ids == ["shared", "filtered-only", "unfiltered-only"]

    def test_two_pass_filtered_priority(self):
        """Filtered results should appear before unfiltered in merged output."""
        def mock_search(query, top_k=5, where=None):
            if where is not None:
                return [_make_chunk("f1", relevance_score=0.5), _make_chunk("f2", relevance_score=0.4)]
            else:
                return [_make_chunk("u1", relevance_score=0.9), _make_chunk("u2", relevance_score=0.8)]

        state = {
            "query": "test",
            "metadata_hints": {"date_start": "2024-01-01", "date_end": "2024-12-31"},
        }
        result = search_corpus(state, mock_search)

        ids = [r["chunk_id"] for r in result["retrieved_chunks"]]
        # Filtered first, even though unfiltered have higher scores
        assert ids[:2] == ["f1", "f2"]

    def test_two_pass_capped_at_10(self):
        """Merged results should be capped at 10."""
        def mock_search(query, top_k=5, where=None):
            if where is not None:
                return [_make_chunk(f"f{i}") for i in range(8)]
            else:
                return [_make_chunk(f"u{i}") for i in range(8)]

        state = {
            "query": "test",
            "metadata_hints": {"date_start": "2024-01-01", "date_end": "2024-12-31"},
        }
        result = search_corpus(state, mock_search)

        assert len(result["retrieved_chunks"]) == 10

    def test_uses_reformulated_query(self):
        """Should use reformulated_query when available."""
        calls = []

        def mock_search(query, top_k=5, where=None):
            calls.append(query)
            return [_make_chunk("a")]

        state = {
            "query": "original",
            "reformulated_query": "reformulated",
            "metadata_hints": None,
        }
        search_corpus(state, mock_search)

        assert calls[0] == "reformulated"

    def test_empty_filtered_results(self):
        """If filtered pass returns nothing, unfiltered results are used."""
        def mock_search(query, top_k=5, where=None):
            if where is not None:
                return []
            else:
                return [_make_chunk("u1"), _make_chunk("u2")]

        state = {
            "query": "test",
            "metadata_hints": {"date_start": "2024-12-01", "date_end": "2024-12-31"},
        }
        result = search_corpus(state, mock_search)

        ids = [r["chunk_id"] for r in result["retrieved_chunks"]]
        assert ids == ["u1", "u2"]
