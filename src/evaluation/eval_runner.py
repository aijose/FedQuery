"""Evaluation runner for retrieval quality measurement.

Loads golden QA, runs search for each question, computes all metrics,
and aggregates by category. No LLM required — pure retrieval evaluation.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable

from src.evaluation.retrieval_metrics import (
    chunk_text_recall,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.evaluation import (
    CategoryMetrics,
    EvaluationReport,
    QuestionResult,
)

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN_PATH = Path("data/eval/golden_qa.json")


def _build_relevant_doc_ids(qa_entry: dict) -> set[str]:
    """Build a set of relevant document identifiers from a QA entry.

    Uses 'type_date' format (e.g., 'statement_2024-01-31') to match
    against the document_type and document_date metadata stored in chunks.
    """
    ids = set()
    for doc_ref in qa_entry.get("relevant_documents", []):
        doc_type = doc_ref.get("type", "")
        doc_date = doc_ref.get("date", "")
        if doc_type and doc_date:
            ids.add(f"{doc_type}_{doc_date}")
    return ids


def _infer_doc_type_from_name(document_name: str) -> str:
    """Infer document type from document_name field.

    E.g., 'FOMC Minutes - December 18, 2024' -> 'minutes'
          'FOMC Statement - March 20, 2024' -> 'statement'
    """
    name_lower = document_name.lower()
    if "minutes" in name_lower:
        return "minutes"
    if "statement" in name_lower:
        return "statement"
    return ""


def _extract_doc_id_from_chunk(chunk: dict) -> str:
    """Extract a comparable document ID from a retrieved chunk.

    Maps chunk metadata to the 'type_date' format used in golden QA.
    Falls back to inferring document_type from document_name if not present.
    """
    metadata = chunk.get("metadata", {})
    # Chunks from search_fn (via create_search_fn) have flat keys
    doc_type = (
        metadata.get("document_type", "")
        or chunk.get("document_type", "")
    )
    # Fallback: infer from document_name
    if not doc_type:
        doc_name = (
            metadata.get("document_title", "")
            or chunk.get("document_name", "")
        )
        doc_type = _infer_doc_type_from_name(doc_name)

    doc_date = (
        metadata.get("document_date", "")
        or chunk.get("document_date", "")
    )
    return f"{doc_type}_{doc_date}"


def run_retrieval_evaluation(
    search_fn: Callable[[str, int], list[dict]],
    golden_dataset_path: Path = DEFAULT_GOLDEN_PATH,
    top_k_values: list[int] | None = None,
    config_label: str = "baseline",
    parameters: dict | None = None,
) -> EvaluationReport:
    """Run retrieval evaluation against the golden QA dataset.

    Args:
        search_fn: Callable (query, top_k) -> list[dict] where each dict
            has at minimum: text/chunk_text, and metadata or flat keys for
            document_type and document_date.
        golden_dataset_path: Path to the golden QA JSON file.
        top_k_values: List of k values to evaluate (default [3, 5, 10]).
        config_label: Label for this evaluation configuration.
        parameters: Optional dict of configuration parameters for the report.

    Returns:
        EvaluationReport with per-question, per-category, and overall metrics.
    """
    if top_k_values is None:
        top_k_values = [3, 5, 10]

    with open(golden_dataset_path) as f:
        golden_qa = json.load(f)

    max_k = max(top_k_values)
    question_results: list[QuestionResult] = []

    for qa_entry in golden_qa:
        question_id = qa_entry["id"]
        question = qa_entry["question"]
        category = qa_entry["category"]
        relevant_doc_ids = _build_relevant_doc_ids(qa_entry)
        expected_fragments = qa_entry.get("relevant_text_fragments", [])

        logger.info("Evaluating: %s", question_id)

        # Run search at max_k to get all results we need
        results = search_fn(question, max_k)

        # Extract document IDs and texts from results
        retrieved_doc_ids = [_extract_doc_id_from_chunk(r) for r in results]
        retrieved_texts = [
            r.get("chunk_text", "") or r.get("text", "")
            for r in results
        ]

        # Compute metrics at each k
        p_at_k = {}
        r_at_k = {}
        n_at_k = {}
        h_at_k = {}
        ct_at_k = {}

        for k in top_k_values:
            p_at_k[k] = precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            r_at_k[k] = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            n_at_k[k] = ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            h_at_k[k] = hit_rate_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            ct_at_k[k] = chunk_text_recall(retrieved_texts, expected_fragments, k)

        mrr_score = mrr(retrieved_doc_ids, relevant_doc_ids)

        question_results.append(QuestionResult(
            question_id=question_id,
            question=question,
            category=category,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            mrr=mrr_score,
            ndcg_at_k=n_at_k,
            hit_rate_at_k=h_at_k,
            chunk_text_recall_at_k=ct_at_k,
        ))

    # Aggregate by category
    per_category = _aggregate_by_category(question_results, top_k_values)

    # Compute overall metrics
    overall = _compute_overall(question_results, top_k_values)

    return EvaluationReport(
        config_label=config_label,
        parameters=parameters or {},
        overall_metrics=overall,
        per_category=per_category,
        per_question=question_results,
    )


def _aggregate_by_category(
    results: list[QuestionResult],
    top_k_values: list[int],
) -> list[CategoryMetrics]:
    """Group results by category and compute averages."""
    by_cat: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    categories = []
    for cat, cat_results in sorted(by_cat.items()):
        n = len(cat_results)
        categories.append(CategoryMetrics(
            category=cat,
            count=n,
            avg_precision_at_k={
                k: sum(r.precision_at_k[k] for r in cat_results) / n
                for k in top_k_values
            },
            avg_recall_at_k={
                k: sum(r.recall_at_k[k] for r in cat_results) / n
                for k in top_k_values
            },
            avg_mrr=sum(r.mrr for r in cat_results) / n,
            avg_ndcg_at_k={
                k: sum(r.ndcg_at_k[k] for r in cat_results) / n
                for k in top_k_values
            },
            avg_hit_rate_at_k={
                k: sum(r.hit_rate_at_k[k] for r in cat_results) / n
                for k in top_k_values
            },
            avg_chunk_text_recall_at_k={
                k: sum(r.chunk_text_recall_at_k[k] for r in cat_results) / n
                for k in top_k_values
            },
        ))
    return categories


def _compute_overall(
    results: list[QuestionResult],
    top_k_values: list[int],
) -> dict:
    """Compute overall averages across all questions."""
    if not results:
        return {}
    n = len(results)
    return {
        "num_questions": n,
        "avg_precision_at_k": {
            k: sum(r.precision_at_k[k] for r in results) / n
            for k in top_k_values
        },
        "avg_recall_at_k": {
            k: sum(r.recall_at_k[k] for r in results) / n
            for k in top_k_values
        },
        "avg_mrr": sum(r.mrr for r in results) / n,
        "avg_ndcg_at_k": {
            k: sum(r.ndcg_at_k[k] for r in results) / n
            for k in top_k_values
        },
        "avg_hit_rate_at_k": {
            k: sum(r.hit_rate_at_k[k] for r in results) / n
            for k in top_k_values
        },
        "avg_chunk_text_recall_at_k": {
            k: sum(r.chunk_text_recall_at_k[k] for r in results) / n
            for k in top_k_values
        },
    }
