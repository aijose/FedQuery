"""CLI command for asking questions about FOMC documents."""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import get_settings
from src.agent.graph import build_graph

console = Console()
app = typer.Typer()

logger = logging.getLogger(__name__)


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

    if settings.fedquery_use_mcp:
        search_fn, cleanup = _build_mcp_search(settings, verbose)
    else:
        search_fn, cleanup = _build_direct_search(settings)

    try:
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
            "metadata_hints": None,
        }

        with console.status("[bold green]Thinking..."):
            result = graph.invoke(initial_state)
    finally:
        if cleanup:
            cleanup()

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


def _build_mcp_search(settings, verbose):
    """Build search function via MCP server subprocess."""
    from src.agent.mcp_client import MCPSearchClient, create_mcp_search_fn

    reranker = _get_reranker(settings)

    mcp_client = MCPSearchClient()
    mcp_client.connect()
    logger.info("MCP mode: server subprocess started")

    search_fn = create_mcp_search_fn(mcp_client, reranker)
    return search_fn, mcp_client.close


def _build_direct_search(settings):
    """Build search function via direct ChromaStore calls."""
    from src.agent.mcp_client import create_direct_search_fn
    from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
    from src.vectorstore.chroma_store import ChromaStore

    store = ChromaStore(path=str(settings.chroma_path))

    if store.count == 0:
        console.print(
            "[bold red]No documents in the vector store.[/bold red]\n"
            "Run 'fedquery ingest --years 2024' first to ingest FOMC documents."
        )
        raise typer.Exit(1)

    embedding_provider = SentenceTransformerEmbeddingProvider()
    reranker = _get_reranker(settings)
    search_fn = create_direct_search_fn(store, embedding_provider, reranker)
    logger.info("Direct mode: in-process ChromaStore")
    return search_fn, None


def _get_reranker(settings):
    """Load cross-encoder reranker if enabled."""
    if settings.fedquery_reranker_enabled:
        from src.retrieval.reranker import CrossEncoderReranker
        return CrossEncoderReranker(settings.fedquery_reranker_model)
    return None
