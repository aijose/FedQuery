"""Chunking parameter grid evaluation.

Re-chunks documents with different parameters, embeds into in-memory ChromaDB,
and evaluates retrieval quality for each configuration. Never disturbs the
persistent store.
"""

import logging
import re
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.embedding.provider import EmbeddingProvider
from src.evaluation.eval_runner import run_retrieval_evaluation
from src.ingestion.chunker import chunk_document
from src.models.document import FOMCDocument
from src.models.enums import DocumentType
from src.models.evaluation import EvaluationReport
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

# Filename pattern: {type}_{date}.txt (excluding .chunks.txt)
_FILENAME_RE = re.compile(r"^(statement|minutes)_(\d{4}-\d{2}-\d{2})\.txt$")

DEFAULT_GRID = [
    {"chunk_size": 256, "chunk_overlap": 25},
    {"chunk_size": 256, "chunk_overlap": 50},
    {"chunk_size": 512, "chunk_overlap": 50},   # current default (baseline)
    {"chunk_size": 512, "chunk_overlap": 100},
    {"chunk_size": 768, "chunk_overlap": 75},
    {"chunk_size": 768, "chunk_overlap": 150},
    {"chunk_size": 1024, "chunk_overlap": 100},
]


def load_documents_from_text_dir(text_dir: Path) -> list[FOMCDocument]:
    """Load FOMCDocuments from saved text files.

    Reconstructs documents from data/texts/{year}/{type}_{date}.txt files.
    Skips .chunks.txt files.
    """
    documents = []
    if not text_dir.exists():
        return documents

    for year_dir in sorted(text_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for txt_file in sorted(year_dir.iterdir()):
            match = _FILENAME_RE.match(txt_file.name)
            if not match:
                continue

            doc_type_str = match.group(1)
            date_str = match.group(2)
            doc_type = DocumentType(doc_type_str)
            doc_date = date.fromisoformat(date_str)

            raw_text = txt_file.read_text(encoding="utf-8")
            if not raw_text.strip():
                continue

            title = f"FOMC {doc_type_str.title()} {date_str}"
            source_url = f"https://www.federalreserve.gov/monetarypolicy/{doc_type_str}{date_str.replace('-', '')}.htm"

            documents.append(FOMCDocument(
                title=title,
                date=doc_date,
                document_type=doc_type,
                source_url=source_url,
                raw_text=raw_text,
            ))

    return documents


def _create_search_fn_for_store(
    store: ChromaStore,
    embedding_provider: EmbeddingProvider,
):
    """Create a search function for evaluation from a ChromaStore."""
    def search_fn(query: str, top_k: int) -> list[dict]:
        query_embedding = embedding_provider.embed([query])[0]
        raw_results = store.query(query_embedding=query_embedding, top_k=top_k)
        results = []
        for r in raw_results:
            metadata = r.get("metadata", {})
            results.append({
                "chunk_text": r.get("text", ""),
                "document_type": metadata.get("document_type", ""),
                "document_date": metadata.get("document_date", ""),
                "metadata": metadata,
            })
        return results
    return search_fn


def evaluate_chunking_grid(
    documents: list[FOMCDocument],
    golden_path: Path,
    grid: list[dict],
    embedding_provider: EmbeddingProvider,
    top_k_values: list[int] | None = None,
    console: Console | None = None,
) -> list[EvaluationReport]:
    """Evaluate multiple chunking configurations.

    For each grid config:
    1. Chunk all documents with the specified parameters
    2. Embed chunks using the embedding provider
    3. Store in a fresh in-memory ChromaDB collection
    4. Run retrieval evaluation
    5. Collect the report

    Returns a list of EvaluationReport, one per grid config.
    """
    if top_k_values is None:
        top_k_values = [3, 5, 10]

    reports = []

    for i, config in enumerate(grid):
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        label = f"chunk_{chunk_size}_overlap_{chunk_overlap}"

        if console:
            console.print(
                f"\n[bold cyan]Config {i + 1}/{len(grid)}: "
                f"chunk_size={chunk_size}, overlap={chunk_overlap}[/bold cyan]"
            )

        # Chunk all documents
        all_chunks = []
        doc_chunk_map = []  # (document, chunks) pairs
        for doc in documents:
            chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
            doc_chunk_map.append((doc, chunks))

        if console:
            console.print(f"  Chunks: {len(all_chunks)}")

        # Embed all chunk texts in batch
        chunk_texts = [c.chunk_text for c in all_chunks]
        if console:
            console.print("  Embedding...")
        embeddings = embedding_provider.embed(chunk_texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        # Create in-memory ChromaDB and add chunks
        store = ChromaStore(path=":memory:")
        for doc, chunks in doc_chunk_map:
            store.add_chunks(chunks, doc)

        if console:
            console.print(f"  Stored: {store.count} chunks")

        # Create search function and run evaluation
        search_fn = _create_search_fn_for_store(store, embedding_provider)

        report = run_retrieval_evaluation(
            search_fn=search_fn,
            golden_dataset_path=golden_path,
            top_k_values=top_k_values,
            config_label=label,
            parameters={
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_documents": len(documents),
                "num_chunks": len(all_chunks),
            },
        )
        reports.append(report)

        if console:
            overall = report.overall_metrics
            mrr_val = overall.get("avg_mrr", 0.0)
            max_k = max(top_k_values)
            p_val = overall.get("avg_precision_at_k", {}).get(max_k, 0.0)
            ctr_val = overall.get("avg_chunk_text_recall_at_k", {}).get(max_k, 0.0)
            console.print(
                f"  MRR={mrr_val:.3f}  P@{max_k}={p_val:.3f}  CTR@{max_k}={ctr_val:.3f}"
            )

        # Clean up the in-memory collection
        del store

    return reports


def format_grid_comparison(
    reports: list[EvaluationReport],
    console: Console,
) -> None:
    """Print a Rich comparison table of all grid configurations."""
    if not reports:
        return

    # Determine k values from first report
    k_values = sorted(reports[0].overall_metrics.get("avg_precision_at_k", {}).keys())
    max_k = max(k_values) if k_values else 10

    table = Table(title="Chunking Grid Comparison")
    table.add_column("Config", style="bold")
    table.add_column("Chunks", justify="right")
    table.add_column("MRR", justify="right")
    for k in k_values:
        table.add_column(f"P@{k}", justify="right")
    for k in k_values:
        table.add_column(f"CTR@{k}", justify="right")

    best_mrr = -1.0
    best_config = ""

    for report in reports:
        overall = report.overall_metrics
        params = report.parameters
        mrr_val = overall.get("avg_mrr", 0.0)

        if mrr_val > best_mrr:
            best_mrr = mrr_val
            best_config = report.config_label

        row = [
            f"{params.get('chunk_size', '?')}/{params.get('chunk_overlap', '?')}",
            str(params.get("num_chunks", "?")),
            f"{mrr_val:.3f}",
        ]
        for k in k_values:
            row.append(f"{overall.get('avg_precision_at_k', {}).get(k, 0.0):.3f}")
        for k in k_values:
            row.append(f"{overall.get('avg_chunk_text_recall_at_k', {}).get(k, 0.0):.3f}")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[bold green]Best MRR: {best_config} ({best_mrr:.3f})[/bold green]")
