"""CLI command for asking questions about FOMC documents."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import get_settings
from src.agent.graph import build_graph
from src.agent.mcp_client import create_search_fn
from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from src.vectorstore.chroma_store import ChromaStore

console = Console()
app = typer.Typer()


@app.command()
def ask(
    question: Annotated[
        str,
        typer.Argument(help="Your question about FOMC monetary policy"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
):
    """Ask a question about FOMC monetary policy documents."""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    settings = get_settings()

    if not settings.anthropic_api_key and settings.fedquery_llm_provider == "anthropic":
        console.print(
            "[bold red]ANTHROPIC_API_KEY not set.[/bold red]\n"
            "Export your API key: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
        raise typer.Exit(1)

    store = ChromaStore(path=str(settings.chroma_path))

    if store.count == 0:
        console.print(
            "[bold red]No documents in the vector store.[/bold red]\n"
            "Run 'fedquery ingest --years 2024' first to ingest FOMC documents."
        )
        raise typer.Exit(1)

    embedding_provider = SentenceTransformerEmbeddingProvider()
    search_fn = create_search_fn(store, embedding_provider)
    graph = build_graph(search_fn)

    initial_state = {
        "query": question,
        "retrieved_chunks": [],
        "confidence": "insufficient",
        "reformulation_attempts": 0,
        "reformulated_query": None,
        "answer": None,
        "citations": [],
        "needs_retrieval": True,
    }

    with console.status("[bold green]Thinking..."):
        result = graph.invoke(initial_state)

    confidence = result.get("confidence", "insufficient")
    answer = result.get("answer", "No answer generated.")

    # Format confidence indicator
    confidence_colors = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
        "insufficient": "red",
    }
    color = confidence_colors.get(confidence, "white")

    header = Text()
    header.append("FedQuery", style="bold")
    header.append(f"  Confidence: ", style="dim")
    header.append(confidence, style=f"bold {color}")

    console.print()
    console.print(Panel(answer, title=header, border_style=color, padding=(1, 2)))
