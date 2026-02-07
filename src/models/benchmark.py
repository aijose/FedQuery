"""Benchmark result data model."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from src.models.enums import IndexType


@dataclass
class BenchmarkResult:
    """Results from HNSW vs IVF index comparison benchmark."""

    index_type: IndexType
    query_latency_ms: float
    recall_at_k: float
    memory_usage_mb: float
    corpus_size: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not isinstance(self.index_type, IndexType):
            self.index_type = IndexType(self.index_type)
        if self.query_latency_ms < 0:
            raise ValueError("query_latency_ms must be >= 0")
        if not 0.0 <= self.recall_at_k <= 1.0:
            raise ValueError("recall_at_k must be between 0.0 and 1.0")
        if self.memory_usage_mb < 0:
            raise ValueError("memory_usage_mb must be >= 0")
        if self.corpus_size <= 0:
            raise ValueError("corpus_size must be > 0")
