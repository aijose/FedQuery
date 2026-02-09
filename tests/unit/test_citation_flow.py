"""Unit tests for synthesize_answer citation parsing, validate_citations, and respond."""

from unittest.mock import patch, MagicMock

from src.agent.nodes import synthesize_answer, validate_citations, respond


def _make_chunk(chunk_id, document_name="Doc", document_date="2024-01-31",
                section_header="Section", chunk_text="Some text.", relevance_score=0.6):
    return {
        "chunk_id": chunk_id,
        "document_name": document_name,
        "document_date": document_date,
        "document_id": "doc-1",
        "section_header": section_header,
        "chunk_text": chunk_text,
        "relevance_score": relevance_score,
    }


CHUNKS = [
    _make_chunk("aaa", chunk_text="Inflation eased over the past year."),
    _make_chunk("bbb", chunk_text="The federal funds rate was held steady."),
    _make_chunk("ccc", chunk_text="Labor market remained tight."),
    _make_chunk("ddd", chunk_text="Consumer spending was strong."),
]


class TestSynthesizeAnswerCitationParsing:
    """synthesize_answer should only include citations the LLM actually referenced."""

    @patch("src.agent.nodes.get_llm")
    def test_only_cited_sources_included(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Inflation eased [Source 1] while rates held steady [Source 2]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 2
        assert result["citations"][0]["chunk_id"] == "aaa"
        assert result["citations"][0]["source_index"] == 1
        assert result["citations"][1]["chunk_id"] == "bbb"
        assert result["citations"][1]["source_index"] == 2

    @patch("src.agent.nodes.get_llm")
    def test_no_sources_cited_yields_empty_citations(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="I cannot find relevant information in the provided sources."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert result["citations"] == []

    @patch("src.agent.nodes.get_llm")
    def test_single_source_cited(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="The labor market was tight [Source 3]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 1
        assert result["citations"][0]["chunk_id"] == "ccc"
        assert result["citations"][0]["source_index"] == 3

    @patch("src.agent.nodes.get_llm")
    def test_out_of_range_source_ignored(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Data shows [Source 1] and [Source 99]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 1
        assert result["citations"][0]["chunk_id"] == "aaa"

    @patch("src.agent.nodes.get_llm")
    def test_citations_ordered_by_first_appearance(self, mock_get_llm):
        """[Source 4] appears before [Source 1], so citation[0] should be chunk ddd."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Spending was strong [Source 4] and inflation eased [Source 1]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 2
        assert result["citations"][0]["chunk_id"] == "ddd"
        assert result["citations"][0]["source_index"] == 4
        assert result["citations"][1]["chunk_id"] == "aaa"
        assert result["citations"][1]["source_index"] == 1

    @patch("src.agent.nodes.get_llm")
    def test_duplicate_source_refs_deduplicated(self, mock_get_llm):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="First [Source 2], then again [Source 2], also [Source 4]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 2
        assert result["citations"][0]["chunk_id"] == "bbb"
        assert result["citations"][0]["source_index"] == 2
        assert result["citations"][1]["chunk_id"] == "ddd"
        assert result["citations"][1]["source_index"] == 4

    @patch("src.agent.nodes.get_llm")
    def test_comma_separated_sources(self, mock_get_llm):
        """LLM uses [Source 1, Source 2, Source 3] comma-separated format."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="The minutes covered inflation [Source 1, Source 2] and labor [Source 3]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 3
        assert result["citations"][0]["chunk_id"] == "aaa"
        assert result["citations"][1]["chunk_id"] == "bbb"
        assert result["citations"][2]["chunk_id"] == "ccc"

    @patch("src.agent.nodes.get_llm")
    def test_comma_separated_sources_deduped(self, mock_get_llm):
        """Comma-separated refs with overlap are deduped."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Data shows [Source 1, Source 3] and confirms [Source 3, Source 4]."
        )
        mock_get_llm.return_value = mock_llm

        state = {"query": "test", "retrieved_chunks": CHUNKS}
        result = synthesize_answer(state)

        assert len(result["citations"]) == 3
        ids = [c["chunk_id"] for c in result["citations"]]
        assert ids == ["aaa", "ccc", "ddd"]


class TestValidateCitations:
    """validate_citations should drop citations not in retrieved_chunks."""

    def test_valid_citations_pass_through(self):
        state = {
            "retrieved_chunks": CHUNKS,
            "citations": [
                {"chunk_id": "aaa", "quoted_excerpt": "old"},
                {"chunk_id": "ccc", "quoted_excerpt": "old"},
            ],
        }
        result = validate_citations(state)
        assert len(result["citations"]) == 2
        # Excerpts should be refreshed from actual chunk text
        assert result["citations"][0]["quoted_excerpt"] == "Inflation eased over the past year."
        assert result["citations"][1]["quoted_excerpt"] == "Labor market remained tight."

    def test_unknown_chunk_id_dropped(self):
        state = {
            "retrieved_chunks": CHUNKS,
            "citations": [
                {"chunk_id": "aaa", "quoted_excerpt": ""},
                {"chunk_id": "nonexistent", "quoted_excerpt": ""},
                {"chunk_id": "bbb", "quoted_excerpt": ""},
            ],
        }
        result = validate_citations(state)
        assert len(result["citations"]) == 2
        ids = [c["chunk_id"] for c in result["citations"]]
        assert "nonexistent" not in ids

    def test_empty_citations(self):
        state = {"retrieved_chunks": CHUNKS, "citations": []}
        result = validate_citations(state)
        assert result["citations"] == []

    def test_empty_retrieved_chunks_drops_all(self):
        state = {
            "retrieved_chunks": [],
            "citations": [{"chunk_id": "aaa", "quoted_excerpt": ""}],
        }
        result = validate_citations(state)
        assert result["citations"] == []


def _make_citation(chunk_id, source_index, document_name="Doc",
                   document_date="2024-01-31", section_header="Section"):
    return {
        "chunk_id": chunk_id,
        "document_name": document_name,
        "document_date": document_date,
        "section_header": section_header,
        "relevance_score": 0.6,
        "quoted_excerpt": "text",
        "source_index": source_index,
    }


class TestRespondSourceRemapping:
    """respond should remap [Source N] in the answer to sequential [N] matching the footer."""

    def test_non_contiguous_sources_remapped(self):
        """[Source 6] and [Source 10] should become [1] and [2]."""
        state = {
            "confidence": "high",
            "answer": "Rates held [Source 6] and inflation eased [Source 10].",
            "citations": [
                _make_citation("aaa", source_index=6),
                _make_citation("bbb", source_index=10),
            ],
        }
        result = respond(state)
        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
        assert "[Source 6]" not in result["answer"]
        assert "[Source 10]" not in result["answer"]

    def test_contiguous_sources_remapped(self):
        """[Source 1] and [Source 2] should stay as [1] and [2]."""
        state = {
            "confidence": "high",
            "answer": "Inflation eased [Source 1] while rates held [Source 2].",
            "citations": [
                _make_citation("aaa", source_index=1),
                _make_citation("bbb", source_index=2),
            ],
        }
        result = respond(state)
        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
        assert "[Source 1]" not in result["answer"]

    def test_repeated_source_refs_all_remapped(self):
        """Multiple references to the same source should all get remapped."""
        state = {
            "confidence": "high",
            "answer": "First [Source 5], then [Source 8], and again [Source 5].",
            "citations": [
                _make_citation("aaa", source_index=5),
                _make_citation("bbb", source_index=8),
            ],
        }
        result = respond(state)
        # [Source 5] → [1] (appears twice in body + once in footer), [Source 8] → [2]
        assert result["answer"].count("[1]") == 3  # 2 in body + 1 in footer
        assert result["answer"].count("[2]") == 2  # 1 in body + 1 in footer
        assert "[Source 5]" not in result["answer"]
        assert "[Source 8]" not in result["answer"]

    def test_first_appearance_order_in_output(self):
        """First citation the reader encounters should be [1], second [2], etc."""
        state = {
            "confidence": "high",
            "answer": "Spending strong [Source 10], rates held [Source 3].",
            "citations": [
                _make_citation("ddd", source_index=10, document_name="Doc10"),
                _make_citation("aaa", source_index=3, document_name="Doc3"),
            ],
        }
        result = respond(state)
        # [Source 10] is first in text → becomes [1], [Source 3] → [2]
        assert "Spending strong [1], rates held [2]." in result["answer"]
        # Footer should match: [1] = Doc10, [2] = Doc3
        assert "[1] Doc10" in result["answer"]
        assert "[2] Doc3" in result["answer"]

    def test_no_citations_no_sources_footer(self):
        """With no citations, answer is returned as-is."""
        state = {
            "confidence": "high",
            "answer": "No specific information available.",
            "citations": [],
        }
        result = respond(state)
        assert result["answer"] == "No specific information available."
        assert "Sources:" not in result["answer"]

    def test_comma_separated_sources_remapped(self):
        """[Source 3, Source 5] should be remapped to [1, 2]."""
        state = {
            "confidence": "high",
            "answer": "Inflation eased and rates held [Source 3, Source 5].",
            "citations": [
                _make_citation("aaa", source_index=3),
                _make_citation("bbb", source_index=5),
            ],
        }
        result = respond(state)
        assert "[1, 2]" in result["answer"]
        assert "[Source 3, Source 5]" not in result["answer"]

    def test_insufficient_confidence_returns_uncertainty(self):
        """Insufficient confidence should return uncertainty message, not remapped sources."""
        state = {
            "query": "test query",
            "confidence": "insufficient",
            "answer": "Some answer [Source 1].",
            "citations": [_make_citation("aaa", source_index=1)],
            "retrieved_chunks": CHUNKS,
        }
        result = respond(state)
        assert "unable to find sufficient information" in result["answer"]
        assert result["citations"] == []
