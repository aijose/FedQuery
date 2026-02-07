"""Query and Answer data models."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from src.models.citation import Citation
from src.models.enums import Confidence


@dataclass
class Query:
    """A natural language question from the analyst."""

    question_text: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.question_text:
            raise ValueError("question_text must not be empty")


@dataclass
class Answer:
    """A synthesized response grounded in retrieved chunks."""

    query_id: str
    response_text: str
    confidence: Confidence
    citations: list[Citation] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not isinstance(self.confidence, Confidence):
            self.confidence = Confidence(self.confidence)
        if not self.response_text:
            raise ValueError("response_text must not be empty")
