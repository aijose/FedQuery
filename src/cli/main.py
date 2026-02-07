"""FedQuery CLI entry point."""

import typer

from src.cli.ask import ask
from src.cli.benchmark import benchmark
from src.cli.evaluate import evaluate
from src.cli.ingest import ingest

app = typer.Typer(
    name="fedquery",
    help="FOMC Agentic RAG Research Assistant - Ask citation-grounded questions about Federal Reserve monetary policy.",
)

app.command(name="ask")(ask)
app.command(name="ingest")(ingest)
app.command(name="benchmark")(benchmark)
app.command(name="evaluate")(evaluate)


if __name__ == "__main__":
    app()
