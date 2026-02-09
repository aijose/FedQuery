"""CLI command for running retrieval quality evaluation."""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from config.settings import get_settings
from src.agent.mcp_client import create_search_fn
from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from src.evaluation.eval_runner import DEFAULT_GOLDEN_PATH, run_retrieval_evaluation
from src.models.evaluation import EvaluationReport
from src.vectorstore.chroma_store import ChromaStore

console = Console()


def _format_report(report: EvaluationReport, verbose: bool = False) -> None:
    """Print a Rich-formatted evaluation report."""
    console.print(f"\n[bold]Evaluation Report: {report.config_label}[/bold]")
    console.print(f"Questions: {report.overall_metrics.get('num_questions', 0)}")
    if report.parameters:
        for key, val in report.parameters.items():
            console.print(f"  {key}: {val}")
    console.print()

    # Overall metrics table
    overall = report.overall_metrics
    k_values = sorted(overall.get("avg_precision_at_k", {}).keys())

    table = Table(title="Overall Metrics")
    table.add_column("Metric", style="bold")
    for k in k_values:
        table.add_column(f"@{k}", justify="right")

    for metric_name, metric_key in [
        ("Precision", "avg_precision_at_k"),
        ("Recall", "avg_recall_at_k"),
        ("NDCG", "avg_ndcg_at_k"),
        ("Hit Rate", "avg_hit_rate_at_k"),
        ("Chunk Text Recall", "avg_chunk_text_recall_at_k"),
    ]:
        values = overall.get(metric_key, {})
        row = [metric_name] + [f"{values.get(k, 0.0):.3f}" for k in k_values]
        table.add_row(*row)

    # MRR doesn't vary by k
    mrr_val = overall.get("avg_mrr", 0.0)
    mrr_row = ["MRR"] + [f"{mrr_val:.3f}"] * len(k_values)
    table.add_row(*mrr_row)

    console.print(table)

    # Per-category table
    if report.per_category:
        cat_table = Table(title="Metrics by Category")
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("N", justify="right")
        cat_table.add_column("MRR", justify="right")
        for k in k_values:
            cat_table.add_column(f"P@{k}", justify="right")
            cat_table.add_column(f"CTR@{k}", justify="right")

        for cat in report.per_category:
            row = [cat.category, str(cat.count), f"{cat.avg_mrr:.3f}"]
            for k in k_values:
                row.append(f"{cat.avg_precision_at_k.get(k, 0.0):.3f}")
                row.append(f"{cat.avg_chunk_text_recall_at_k.get(k, 0.0):.3f}")
            cat_table.add_row(*row)

        console.print(cat_table)

    # Per-question detail (verbose only)
    if verbose and report.per_question:
        detail_table = Table(title="Per-Question Results")
        detail_table.add_column("ID", style="bold")
        detail_table.add_column("Category")
        detail_table.add_column("MRR", justify="right")
        max_k = max(k_values)
        detail_table.add_column(f"P@{max_k}", justify="right")
        detail_table.add_column(f"CTR@{max_k}", justify="right")

        for qr in report.per_question:
            detail_table.add_row(
                qr.question_id,
                qr.category,
                f"{qr.mrr:.3f}",
                f"{qr.precision_at_k.get(max_k, 0.0):.3f}",
                f"{qr.chunk_text_recall_at_k.get(max_k, 0.0):.3f}",
            )
        console.print(detail_table)


def evaluate(
    top_k: Annotated[
        str,
        typer.Option("--top-k", help="Comma-separated k values (e.g., '3,5,10')"),
    ] = "3,5,10",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show per-question details"),
    ] = False,
    golden_path: Annotated[
        str,
        typer.Option("--golden-path", help="Path to golden QA dataset"),
    ] = str(DEFAULT_GOLDEN_PATH),
    chunking_grid: Annotated[
        bool,
        typer.Option("--chunking-grid", help="Run chunking parameter grid evaluation"),
    ] = False,
):
    """Evaluate retrieval quality against a golden QA dataset."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    k_values = [int(k.strip()) for k in top_k.split(",")]
    qa_path = Path(golden_path)

    if not qa_path.exists():
        console.print(f"[bold red]Golden QA file not found: {qa_path}[/bold red]")
        raise typer.Exit(1)

    if chunking_grid:
        _run_chunking_grid(k_values, qa_path, verbose)
        return

    settings = get_settings()
    store = ChromaStore(path=str(settings.chroma_path))

    if store.count == 0:
        console.print(
            "[bold red]No documents in the vector store.[/bold red]\n"
            "Run 'fedquery ingest --years 2024' first."
        )
        raise typer.Exit(1)

    console.print("[bold]FedQuery Retrieval Evaluation[/bold]")
    console.print(f"Corpus size: {store.count} chunks")

    embedding_provider = SentenceTransformerEmbeddingProvider()
    search_fn = create_search_fn(store, embedding_provider)

    # Wrap search_fn to return dicts with the keys eval_runner expects
    def eval_search_fn(query: str, top_k: int) -> list[dict]:
        results = search_fn(query, top_k)
        return [
            {
                "chunk_text": r["chunk_text"],
                "document_name": r.get("document_name", ""),
                "document_type": r.get("document_type", ""),
                "document_date": r.get("document_date", ""),
                "metadata": {
                    "document_title": r.get("document_name", ""),
                    "document_type": r.get("document_type", ""),
                    "document_date": r.get("document_date", ""),
                },
            }
            for r in results
        ]

    with console.status("[bold green]Running evaluation..."):
        report = run_retrieval_evaluation(
            search_fn=eval_search_fn,
            golden_dataset_path=qa_path,
            top_k_values=k_values,
            config_label="baseline",
            parameters={
                "embedding_model": settings.fedquery_embedding_model,
                "chunk_size": settings.fedquery_chunk_size,
                "chunk_overlap": settings.fedquery_chunk_overlap,
                "corpus_chunks": store.count,
            },
        )

    _format_report(report, verbose)


def _run_chunking_grid(
    k_values: list[int],
    qa_path: Path,
    verbose: bool,
) -> None:
    """Run chunking parameter grid evaluation."""
    # Imported here to avoid circular imports and heavy loading on regular evaluate
    from src.evaluation.chunking_eval import (
        DEFAULT_GRID,
        evaluate_chunking_grid,
        format_grid_comparison,
        load_documents_from_text_dir,
    )

    settings = get_settings()
    text_dir = settings.text_path

    console.print("[bold]FedQuery Chunking Grid Evaluation[/bold]")
    console.print(f"Text directory: {text_dir}")

    with console.status("[bold green]Loading documents from text files..."):
        documents = load_documents_from_text_dir(text_dir)

    if not documents:
        console.print("[bold red]No text files found.[/bold red]")
        raise typer.Exit(1)

    console.print(f"Loaded {len(documents)} documents")

    embedding_provider = SentenceTransformerEmbeddingProvider()

    reports = evaluate_chunking_grid(
        documents=documents,
        golden_path=qa_path,
        grid=DEFAULT_GRID,
        embedding_provider=embedding_provider,
        top_k_values=k_values,
        console=console,
    )

    format_grid_comparison(reports, console)

    if verbose:
        for report in reports:
            _format_report(report, verbose=True)
