"""Citation data model."""

from dataclasses import dataclass
from datetime import date


@dataclass
class Citation:
    """A reference linking an answer claim to a source chunk."""

    document_name: str
    document_date: date
    chunk_id: str
    relevance_score: float
    quoted_excerpt: str
    section_header: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(f"relevance_score must be between 0.0 and 1.0, got {self.relevance_score}")
        if not self.quoted_excerpt:
            raise ValueError("quoted_excerpt must not be empty")
