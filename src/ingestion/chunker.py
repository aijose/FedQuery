"""Recursive text chunker with FOMC section header metadata."""

import re

from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument

# Section header patterns for FOMC documents (from research.md)
SECTION_PATTERNS = [
    re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE),
    re.compile(r"^(Participants'.+)", re.MULTILINE),
    re.compile(r"^(Committee Policy.+)", re.MULTILINE),
    re.compile(r"^(Developments in.+)", re.MULTILINE),
    re.compile(r"^(Staff Review.+)", re.MULTILINE),
    re.compile(r"^(Staff Economic.+)", re.MULTILINE),
    re.compile(r"^(Financial Developments.+)", re.MULTILINE),
]


def detect_section_header(text: str) -> str | None:
    """Detect if the given text line is a section header.

    Returns the header text if matched, None otherwise.
    """
    text = text.strip()
    for pattern in SECTION_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text recursively using a hierarchy of separators.

    Tries paragraph breaks first, then sentences, then words.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    estimated_tokens = _estimate_tokens(text)
    if estimated_tokens <= chunk_size:
        return [text]

    # Find the first working separator
    separator = separators[0]
    remaining_separators = separators[1:] if len(separators) > 1 else separators

    parts = text.split(separator)
    if len(parts) == 1 and remaining_separators != separators:
        return _split_text_recursive(text, chunk_size, chunk_overlap, remaining_separators)

    chunks = []
    current_chunk = ""

    for part in parts:
        candidate = (current_chunk + separator + part).strip() if current_chunk else part.strip()

        if _estimate_tokens(candidate) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_text = _get_overlap_text(current_chunk, chunk_overlap)
            current_chunk = (overlap_text + separator + part).strip() if overlap_text else part.strip()
        else:
            current_chunk = candidate

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _get_overlap_text(text: str, overlap_tokens: int) -> str:
    """Get the last N tokens worth of text for overlap, snapping to a word boundary."""
    if overlap_tokens <= 0:
        return ""
    chars = overlap_tokens * 4  # ~4 chars per token
    if len(text) <= chars:
        return text
    candidate = text[-chars:]
    # Snap forward to the next word boundary to avoid mid-word truncation
    space_idx = candidate.find(" ")
    if space_idx != -1:
        candidate = candidate[space_idx + 1:]
    return candidate


def chunk_document(
    document: FOMCDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split a document into chunks with section header metadata.

    Uses recursive character splitting with paragraph-first boundaries.
    Captures FOMC section headers via regex and attaches them as metadata.
    """
    text = document.raw_text
    raw_chunks = _split_text_recursive(text, chunk_size, chunk_overlap)

    # Track the current section header as we process chunks in order
    current_header = None
    result = []

    for idx, chunk_text in enumerate(raw_chunks):
        # Check if this chunk starts with or contains a section header
        lines = chunk_text.split("\n")
        for line in lines:
            header = detect_section_header(line)
            if header:
                current_header = header
                break

        result.append(
            DocumentChunk(
                document_id=document.id,
                chunk_text=chunk_text,
                chunk_index=idx,
                token_count=_estimate_tokens(chunk_text),
                section_header=current_header,
            )
        )

    return result
