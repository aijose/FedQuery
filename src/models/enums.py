"""Enumeration types for FedQuery data models."""

from enum import Enum


class DocumentType(str, Enum):
    STATEMENT = "statement"
    MINUTES = "minutes"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class IndexType(str, Enum):
    HNSW = "hnsw"
    IVF = "ivf"
