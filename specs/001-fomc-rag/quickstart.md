# FedQuery Quickstart Guide

> **Goal**: Ingest FOMC documents and run your first query in under 10 minutes (SC-005).

## Prerequisites

- **Python 3.11+**
- **Anthropic API key** (sign up at https://console.anthropic.com)
- ~500 MB disk space for embeddings and vector store

## Installation

```bash
git clone <repo-url>
cd FedQuery
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

This installs all dependencies: LangChain, LangGraph, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), MCP Python SDK, Typer, Rich, BeautifulSoup4, and FAISS.

## Configuration

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Optional settings** (defaults work out of the box):

| Setting | Default | Description |
|---------|---------|-------------|
| `FEDQUERY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for embeddings |
| `FEDQUERY_LLM_PROVIDER` | `anthropic` | LLM backend (configurable via LangChain) |
| `FEDQUERY_CHROMA_PATH` | `./data/chroma` | ChromaDB persistence directory |

## Step 1 - Ingest Documents

```bash
fedquery ingest --years 2023 2024
```

This command:
1. Scrapes FOMC statements and minutes from federalreserve.gov for the specified years
2. Chunks documents with section header metadata
3. Generates embeddings using all-MiniLM-L6-v2
4. Stores vectors and metadata in ChromaDB

**Expected output:**

```
Fetching FOMC documents for 2023-2024...
  [################] 16/16 statements downloaded
  [################] 16/16 minutes downloaded
Chunking 32 documents... 847 chunks created
Generating embeddings... done (42s)
Stored in ChromaDB at ./data/chroma

Ingestion complete: 32 documents, 847 chunks indexed.
```

## Step 2 - Ask a Question

```bash
fedquery ask "What did the Fed say about inflation in March 2024?"
```

The LangGraph agent starts the MCP server, retrieves relevant chunks, and synthesizes an answer with citations.

**Expected output:**

```
In the March 2024 FOMC statement, the Committee noted that inflation
"has eased over the past year but remains elevated." The Committee
indicated it does not expect it will be appropriate to reduce the
target range until it has gained "greater confidence that inflation
is moving sustainably toward 2 percent."

Sources:
  [1] FOMC Statement, March 20, 2024, §Policy Decision (chunks 3, 7)
  [2] FOMC Minutes, March 19-20, 2024, §Economic Outlook (chunks 12, 15)
```

## Optional - Run Benchmark

Compare HNSW and IVF index performance using FAISS:

```bash
fedquery benchmark --index-types hnsw ivf
```

**Expected output:**

```
Benchmark Results (847 chunks, 50 queries)
------------------------------------------
Index Type  | Build Time | Avg Query (ms) | Recall@10
HNSW        | 1.2s       | 0.8ms          | 0.98
IVF         | 0.9s       | 0.5ms          | 0.95
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ANTHROPIC_API_KEY not set` | Export the env var or add to `.env` file |
| `ChromaDB permission error` | Check write permissions on `./data/chroma` |
| `Connection error during ingest` | Verify internet access to federalreserve.gov |
| `Model download slow` | First run downloads ~80 MB for all-MiniLM-L6-v2; subsequent runs use cache |
