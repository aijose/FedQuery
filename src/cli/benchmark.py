"""CLI command for running HNSW vs IVF benchmark."""

import logging
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from config.settings import get_settings
from src.vectorstore.benchmark import format_benchmark_report, run_benchmark
from src.vectorstore.chroma_store import ChromaStore

console = Console()
app = typer.Typer()


@app.command()
def benchmark(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
):
    """Run HNSW vs IVF benchmark on the FOMC corpus."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    store = ChromaStore(path=str(settings.chroma_path))

    if store.count == 0:
        console.print(
            "[bold red]No documents in the vector store.[/bold red]\n"
            "Run 'fedquery ingest --years 2024' first."
        )
        raise typer.Exit(1)

    console.print("[bold]FedQuery Benchmark: HNSW vs IVF[/bold]")
    console.print(f"Corpus size: {store.count} chunks")
    console.print()

    # Extract embeddings from ChromaDB
    collection = store._collection
    all_data = collection.get(include=["embeddings"])
    embeddings = np.array(all_data["embeddings"], dtype=np.float32)

    console.print(f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]} dimensions)")

    with console.status("[bold green]Running benchmark..."):
        results = run_benchmark(embeddings)

    report = format_benchmark_report(results)
    console.print()
    console.print(report)
