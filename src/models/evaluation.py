"""Evaluation result data models."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class QuestionResult:
    """Metrics for a single evaluation question."""

    question_id: str
    question: str
    category: str
    precision_at_k: dict[int, float]  # k -> score
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    chunk_text_recall_at_k: dict[int, float]


@dataclass
class CategoryMetrics:
    """Aggregated metrics for a question category."""

    category: str
    count: int
    avg_precision_at_k: dict[int, float]
    avg_recall_at_k: dict[int, float]
    avg_mrr: float
    avg_ndcg_at_k: dict[int, float]
    avg_hit_rate_at_k: dict[int, float]
    avg_chunk_text_recall_at_k: dict[int, float]


@dataclass
class EvaluationReport:
    """Complete retrieval evaluation report."""

    config_label: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parameters: dict = field(default_factory=dict)
    overall_metrics: dict = field(default_factory=dict)
    per_category: list[CategoryMetrics] = field(default_factory=list)
    per_question: list[QuestionResult] = field(default_factory=list)
