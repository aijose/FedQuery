"""Ingestion pipeline orchestrator.

Wires together: scraper → cleaner → chunker → embedding → chroma_store.
"""

import logging

from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from src.ingestion.chunker import chunk_document
from src.ingestion.scraper import scrape_fomc_documents
from src.ingestion.text_writer import save_document_chunks, save_document_html, save_document_text
from src.models.document import FOMCDocument
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(
    years: list[int],
    store: ChromaStore | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    save_chunks: bool = False,
) -> dict:
    """Run the full ingestion pipeline for the given years.

    Steps:
    1. Scrape FOMC documents from federalreserve.gov
    2. For each document, check for duplicates
    3. Chunk the document text with section metadata
    4. Embed chunks using the embedding provider
    5. Store in ChromaDB

    Returns a summary dict with counts.
    """
    from config.settings import get_settings
    settings = get_settings()

    if store is None:
        store = ChromaStore(path=str(settings.chroma_path))

    embedding_provider = SentenceTransformerEmbeddingProvider()

    # Step 1: Scrape documents
    documents = scrape_fomc_documents(years)
    logger.info("Scraped %d documents", len(documents))

    documents_ingested = 0
    documents_skipped = 0
    chunks_stored = 0
    text_files_saved = 0
    html_files_saved = 0
    errors = 0

    for doc in documents:
        # Step 2: Deduplication check
        if store.has_document(doc.source_url):
            logger.info("Skipping duplicate: %s", doc.source_url)
            documents_skipped += 1
            continue

        try:
            # Step 3: Chunk
            chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                logger.warning("No chunks produced for %s", doc.title)
                continue

            # Step 4: Embed
            chunk_texts = [c.chunk_text for c in chunks]
            embeddings = embedding_provider.embed(chunk_texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

            # Step 4.5: Save text and HTML to disk
            try:
                text_path = save_document_text(doc, settings.text_path)
                if text_path:
                    text_files_saved += 1
            except Exception:
                pass  # already handled inside save_document_text

            try:
                html_path = save_document_html(doc, settings.html_path)
                if html_path:
                    html_files_saved += 1
            except Exception:
                pass  # already handled inside save_document_html

            if save_chunks:
                try:
                    save_document_chunks(chunks, doc, settings.text_path)
                except Exception:
                    pass  # already handled inside save_document_chunks

            # Step 5: Store
            n_stored = store.add_chunks(chunks, doc)
            chunks_stored += n_stored
            documents_ingested += 1
            logger.info(
                "Ingested %s: %d chunks", doc.title, n_stored
            )
        except Exception as e:
            logger.error("Error ingesting %s: %s", doc.title, e)
            errors += 1

    return {
        "documents_ingested": documents_ingested,
        "documents_skipped": documents_skipped,
        "chunks_stored": chunks_stored,
        "text_files_saved": text_files_saved,
        "html_files_saved": html_files_saved,
        "errors": errors,
        "total_documents": len(documents),
    }
