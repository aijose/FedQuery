"""Tests for the evaluation runner."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.eval_runner import run_retrieval_evaluation
from src.models.evaluation import EvaluationReport


@pytest.fixture
def golden_qa_path(tmp_path):
    """Create a minimal golden QA dataset for testing."""
    qa_data = [
        {
            "id": "test-01",
            "question": "What was the rate in January?",
            "category": "factual",
            "expected_answer_keywords": ["5-1/4", "5-1/2"],
            "relevant_documents": [
                {"type": "statement", "date": "2024-01-31"}
            ],
            "relevant_sections": [],
            "relevant_text_fragments": [
                "target range for the federal funds rate at 5-1/4 to 5-1/2"
            ],
            "difficulty": "easy",
        },
        {
            "id": "test-02",
            "question": "Who dissented in September?",
            "category": "factual",
            "expected_answer_keywords": ["Bowman"],
            "relevant_documents": [
                {"type": "statement", "date": "2024-09-18"}
            ],
            "relevant_sections": [],
            "relevant_text_fragments": [
                "Michelle W. Bowman"
            ],
            "difficulty": "easy",
        },
        {
            "id": "test-oos",
            "question": "What is the GDP of France?",
            "category": "out-of-scope",
            "expected_answer_keywords": [],
            "relevant_documents": [],
            "relevant_sections": [],
            "relevant_text_fragments": [],
            "difficulty": "easy",
        },
    ]
    path = tmp_path / "test_golden.json"
    path.write_text(json.dumps(qa_data))
    return path


def _make_mock_search_fn(fixed_results: list[dict]):
    """Create a mock search function returning fixed results."""
    def search_fn(query: str, top_k: int) -> list[dict]:
        return fixed_results[:top_k]
    return search_fn


class TestRunRetrievalEvaluation:
    def test_basic_report_structure(self, golden_qa_path):
        results = [
            {
                "chunk_text": "target range for the federal funds rate at 5-1/4 to 5-1/2 percent",
                "document_type": "statement",
                "document_date": "2024-01-31",
                "metadata": {},
            },
            {
                "chunk_text": "Michelle W. Bowman preferred a smaller cut",
                "document_type": "statement",
                "document_date": "2024-09-18",
                "metadata": {},
            },
        ]
        search_fn = _make_mock_search_fn(results)

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_qa_path,
            top_k_values=[3, 5],
            config_label="test-config",
        )

        assert isinstance(report, EvaluationReport)
        assert report.config_label == "test-config"
        assert report.overall_metrics["num_questions"] == 3
        assert len(report.per_question) == 3
        assert len(report.per_category) == 2  # factual + out-of-scope

    def test_perfect_retrieval(self, golden_qa_path):
        """Mock search returns exactly the right documents."""
        results = [
            {
                "chunk_text": "target range for the federal funds rate at 5-1/4 to 5-1/2 percent",
                "document_type": "statement",
                "document_date": "2024-01-31",
                "metadata": {},
            },
            {
                "chunk_text": "Michelle W. Bowman voted against",
                "document_type": "statement",
                "document_date": "2024-09-18",
                "metadata": {},
            },
        ]
        search_fn = _make_mock_search_fn(results)

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_qa_path,
            top_k_values=[5],
        )

        # Factual questions: result[0] matches test-01, result[1] matches test-02
        # The first question should find its doc in result[0]
        factual_cat = next(c for c in report.per_category if c.category == "factual")
        assert factual_cat.count == 2
        assert factual_cat.avg_hit_rate_at_k[5] == 1.0

    def test_empty_search_results(self, golden_qa_path):
        search_fn = _make_mock_search_fn([])

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_qa_path,
            top_k_values=[3],
        )

        assert report.overall_metrics["num_questions"] == 3
        # Factual questions should have 0 scores
        factual_cat = next(c for c in report.per_category if c.category == "factual")
        assert factual_cat.avg_mrr == 0.0
        assert factual_cat.avg_precision_at_k[3] == 0.0

    def test_out_of_scope_vacuous_truth(self, golden_qa_path):
        """Out-of-scope questions should get perfect scores (vacuous truth)."""
        search_fn = _make_mock_search_fn([
            {"chunk_text": "irrelevant text", "document_type": "x", "document_date": "y", "metadata": {}},
        ])

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_qa_path,
            top_k_values=[3],
        )

        oos = next(q for q in report.per_question if q.question_id == "test-oos")
        assert oos.recall_at_k[3] == 1.0
        assert oos.hit_rate_at_k[3] == 1.0
        assert oos.chunk_text_recall_at_k[3] == 1.0

    def test_parameters_stored(self, golden_qa_path):
        search_fn = _make_mock_search_fn([])
        params = {"chunk_size": 512, "model": "test"}

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_qa_path,
            parameters=params,
        )

        assert report.parameters == params

    def test_metadata_based_doc_extraction(self, golden_qa_path):
        """Results with nested metadata should also work."""
        results = [
            {
                "chunk_text": "target range for the federal funds rate at 5-1/4 to 5-1/2",
                "text": "",
                "metadata": {
                    "document_type": "statement",
                    "document_date": "2024-01-31",
                },
            },
        ]
        search_fn = _make_mock_search_fn(results)

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_qa_path,
            top_k_values=[3],
        )

        q1 = next(q for q in report.per_question if q.question_id == "test-01")
        assert q1.hit_rate_at_k[3] == 1.0
        assert q1.chunk_text_recall_at_k[3] == 1.0
