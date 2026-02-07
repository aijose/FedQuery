"""Agent state definition for the LangGraph workflow."""

from typing import TypedDict


class ChunkResult(TypedDict):
    """A retrieved chunk from the vector store."""
    chunk_id: str
    document_name: str
    document_date: str
    document_id: str
    section_header: str
    chunk_text: str
    relevance_score: float


class AgentState(TypedDict):
    """State object passed through the LangGraph workflow."""
    query: str
    retrieved_chunks: list[ChunkResult]
    confidence: str  # "high" | "medium" | "low" | "insufficient"
    reformulation_attempts: int
    reformulated_query: str | None
    answer: str | None
    citations: list[dict]
    needs_retrieval: bool
