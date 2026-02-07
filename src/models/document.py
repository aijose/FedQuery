"""FOMC Document data model."""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

from src.models.enums import DocumentType


@dataclass
class FOMCDocument:
    """A press statement or meeting minutes document from the Federal Reserve."""

    title: str
    date: date
    document_type: DocumentType
    source_url: str
    raw_text: str
    raw_html: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ingested_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not isinstance(self.document_type, DocumentType):
            self.document_type = DocumentType(self.document_type)
        if not self.title:
            raise ValueError("title must not be empty")
        if not self.source_url:
            raise ValueError("source_url must not be empty")
        if not self.raw_text:
            raise ValueError("raw_text must not be empty")
