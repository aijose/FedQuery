"""Document Chunk data model."""

import uuid
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """A segment of an FOMC document sized for embedding and retrieval."""

    document_id: str
    chunk_text: str
    chunk_index: int
    token_count: int
    embedding: list[float] = field(default_factory=list)
    section_header: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if not self.chunk_text:
            raise ValueError("chunk_text must not be empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0")
        if self.token_count <= 0:
            raise ValueError("token_count must be > 0")
        if self.embedding and len(self.embedding) != 384:
            raise ValueError(f"embedding dimension must be 384, got {len(self.embedding)}")  # bge-small-en-v1.5 is also 384-dim
