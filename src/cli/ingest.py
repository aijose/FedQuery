"""CLI command for FOMC document ingestion."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import get_settings
from src.ingestion.pipeline import run_ingestion_pipeline
from src.vectorstore.chroma_store import ChromaStore

console = Console()
app = typer.Typer()


@app.command()
def ingest(
    years: Annotated[
        list[int],
        typer.Option("--years", "-y", help="Years to ingest FOMC documents for"),
    ],
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Target chunk size in tokens"),
    ] = 512,
    chunk_overlap: Annotated[
        int,
        typer.Option("--chunk-overlap", help="Overlap between chunks in tokens"),
    ] = 50,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
):
    """Ingest FOMC documents from federalreserve.gov into the vector store."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    store = ChromaStore(path=str(settings.chroma_path))

    console.print(f"[bold]FedQuery Ingestion[/bold]")
    console.print(f"Years: {years}")
    console.print(f"Chunk size: {chunk_size} tokens, overlap: {chunk_overlap} tokens")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting FOMC documents...", total=None)
        result = run_ingestion_pipeline(
            years=years,
            store=store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        progress.update(task, completed=True)

    console.print()
    console.print("[bold green]Ingestion complete![/bold green]")
    console.print(f"  Documents ingested: {result['documents_ingested']}")
    console.print(f"  Documents skipped (duplicates): {result['documents_skipped']}")
    console.print(f"  Chunks stored: {result['chunks_stored']}")
    console.print(f"  Errors: {result['errors']}")
    console.print(f"  Total in store: {store.count} chunks")
