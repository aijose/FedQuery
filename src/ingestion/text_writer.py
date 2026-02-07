"""Save FOMC document text, HTML, and chunks to disk."""

import logging
from pathlib import Path

from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument

logger = logging.getLogger(__name__)


def build_text_path(doc: FOMCDocument, base_dir: Path) -> Path:
    """Compute the file path for a document's text file without writing.

    Returns a path like: base_dir/2024/statement_2024-01-31.txt
    """
    return base_dir / str(doc.date.year) / f"{doc.document_type.value}_{doc.date.isoformat()}.txt"


def build_chunks_path(doc: FOMCDocument, base_dir: Path) -> Path:
    """Compute the file path for a document's chunks debug file.

    Returns a path like: base_dir/2024/statement_2024-01-31.chunks.txt
    """
    return base_dir / str(doc.date.year) / f"{doc.document_type.value}_{doc.date.isoformat()}.chunks.txt"


def build_html_path(doc: FOMCDocument, base_dir: Path) -> Path:
    """Compute the file path for a document's HTML file without writing.

    Returns a path like: base_dir/2024/statement_2024-01-31.html
    """
    return base_dir / str(doc.date.year) / f"{doc.document_type.value}_{doc.date.isoformat()}.html"


def save_document_text(doc: FOMCDocument, base_dir: Path) -> Path | None:
    """Save the cleaned text of an FOMC document to a predictably named file.

    Returns the Path to the written file, or None if the file already exists
    or the write failed.
    """
    target = build_text_path(doc, base_dir)

    if target.exists():
        logger.debug("Text file already exists, skipping: %s", target)
        return None

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.raw_text, encoding="utf-8")
        logger.info("Saved text file: %s", target)
        return target
    except OSError as e:
        logger.warning("Failed to write text file %s: %s", target, e)
        return None


def save_document_html(doc: FOMCDocument, base_dir: Path) -> Path | None:
    """Save the original HTML of an FOMC document to a predictably named file.

    Returns the Path to the written file, or None if the document has no
    raw_html, the file already exists, or the write failed.
    """
    if not doc.raw_html:
        return None

    target = build_html_path(doc, base_dir)

    if target.exists():
        logger.debug("HTML file already exists, skipping: %s", target)
        return None

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.raw_html, encoding="utf-8")
        logger.info("Saved HTML file: %s", target)
        return target
    except OSError as e:
        logger.warning("Failed to write HTML file %s: %s", target, e)
        return None


def _format_chunks(chunks: list[DocumentChunk], doc: FOMCDocument) -> str:
    """Format chunks into a human-readable debug file."""
    lines = [
        f"Document: {doc.title}",
        f"Date: {doc.date.isoformat()}",
        f"Source: {doc.source_url}",
        f"Total chunks: {len(chunks)}",
        "",
    ]

    for chunk in chunks:
        header = chunk.section_header or "(no section)"
        lines.append(f"{'=' * 72}")
        lines.append(f"CHUNK {chunk.chunk_index}  |  section: {header}  |  tokens: {chunk.token_count}  |  id: {chunk.id}")
        lines.append(f"{'=' * 72}")
        lines.append(chunk.chunk_text)
        lines.append("")

    return "\n".join(lines)


def save_document_chunks(
    chunks: list[DocumentChunk],
    doc: FOMCDocument,
    base_dir: Path,
) -> Path | None:
    """Save chunked text for a document to a debug file.

    Writes a readable file showing each chunk with its index, section
    header, token count, and text. Useful for inspecting chunk boundaries.

    Returns the Path to the written file, or None if the file already
    exists or the write failed.
    """
    if not chunks:
        return None

    target = build_chunks_path(doc, base_dir)

    if target.exists():
        logger.debug("Chunks file already exists, skipping: %s", target)
        return None

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_format_chunks(chunks, doc), encoding="utf-8")
        logger.info("Saved chunks file: %s", target)
        return target
    except OSError as e:
        logger.warning("Failed to write chunks file %s: %s", target, e)
        return None
