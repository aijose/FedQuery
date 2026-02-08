# FedQuery

FOMC Agentic RAG Research Assistant — ask citation-grounded questions about Federal Reserve monetary policy.

## What It Does

FedQuery ingests FOMC press statements and meeting minutes from federalreserve.gov, indexes them in a local vector database, and uses an LLM-powered agent to answer analyst questions with traceable citations.

```
$ fedquery ask "What did the Fed say about inflation in January 2024?"

╭─ FedQuery  Confidence: high ─────────────────────────────────────────╮
│                                                                       │
│  The January 2024 FOMC minutes indicate that inflation "has eased     │
│  over the past year but remains somewhat elevated" above the          │
│  Committee's 2 percent objective. [Source 1]                          │
│                                                                       │
│  Sources:                                                             │
│    [1] FOMC Minutes - Jan 31, 2024, §Economic Outlook (chunk a3f1)   │
│    [2] FOMC Statement - Jan 31, 2024, §Policy Decision (chunk b7e2)  │
╰───────────────────────────────────────────────────────────────────────╯
```

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd FedQuery
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add your API key (e.g. ANTHROPIC_API_KEY=sk-ant-...)

# Ingest FOMC documents
fedquery ingest --years 2024

# Ask a question
fedquery ask "What was the Fed's stance on interest rates in 2024?"
```

## Architecture

### Ingestion Pipeline

```
federalreserve.gov → Scraper → Cleaner → Chunker → Embedder → ChromaDB
```

- **Scraper** (`src/ingestion/scraper.py`): Fetches FOMC calendar pages, extracts statement and minutes URLs, downloads HTML documents.
- **Cleaner** (`src/ingestion/cleaner.py`): Strips HTML, normalizes whitespace, handles tables gracefully.
- **Chunker** (`src/ingestion/chunker.py`): Recursive character splitting (~512 tokens, ~50 token overlap) with FOMC section header detection via regex. Section headers like "Participants' Views on Current Conditions" and "Committee Policy Action" are captured as chunk metadata.
- **Embedder**: all-MiniLM-L6-v2 (384 dimensions) via a swappable `EmbeddingProvider` interface.
- **Vector Store**: ChromaDB with cosine similarity, storing 7 metadata fields per chunk.

### Chunking and Citation Design

FOMC documents have predictable section headers. The chunker captures these via regex and attaches them as metadata to each chunk:

```
SECTION_PATTERNS = [
    r"^(Participants'.+)",
    r"^(Committee Policy.+)",
    r"^(Developments in.+)",
    r"^(Staff Review.+)",
]
```

This enables human-readable citations without a full document parser:

```
FOMC Minutes, Jan 2024, §Economic Outlook, Chunk 2
```

Recursive character splitting respects paragraph boundaries before falling back to sentence and word boundaries, keeping chunks coherent.

### Agent Workflow (LangGraph)

```
assess_query → search_corpus → evaluate_confidence
                                      │
                    ┌─────────────────┼──────────────────┐
                    │                 │                   │
              confidence ≥       confidence =       insufficient
               "medium"            "low" &          or max retries
                    │            attempts < 2             │
                    ▼                 ▼                   ▼
           synthesize_answer   reformulate_query      respond
                    │                 │            (uncertainty)
                    ▼                 └→ search_corpus
           validate_citations
                    │
                    ▼
                 respond
```

- **Confidence thresholds**: high ≥ 0.55, medium ≥ 0.40, low ≥ 0.25, insufficient < 0.25 (calibrated for all-MiniLM-L6-v2 cosine similarity)
- **Reformulation**: When confidence is low, the agent rephrases the query and retries (max 2 attempts)
- **Citation validation**: Each citation is verified against retrieved chunks before being included in the response
- **Uncertainty handling**: When evidence is insufficient, the system explicitly says so rather than fabricating an answer

### LLM Agnosticism

The agent uses LangChain's `BaseChatModel` abstraction. Swap providers by changing environment variables:

```bash
# Default: Anthropic Claude
export FEDQUERY_LLM_PROVIDER=anthropic
export FEDQUERY_LLM_MODEL=claude-sonnet-4-5-20250929

# Alternative: Google Gemini
export FEDQUERY_LLM_PROVIDER=google
export FEDQUERY_LLM_MODEL=gemini-2.0-flash
```

### MCP Server

Two tools exposed via the MCP Python SDK (stdio transport):

| Tool | Purpose |
|------|---------|
| `search_fomc(query, top_k)` | Semantic search across the vector store |
| `get_document(doc_id)` | Fetch full document content by ID |

All retrieval goes through MCP tools — the agent never accesses the vector store directly.

## HNSW vs IVF Benchmark

```bash
fedquery benchmark
```

Compares two approximate nearest neighbor index types using FAISS on the same FOMC corpus:

| Metric | HNSW | IVF |
|--------|------|-----|
| Query Latency | Lower at small scale | Better at >1M vectors |
| Recall@k | Typically >98% | Typically >95% |
| Memory | Higher (graph structure) | Lower (inverted lists) |

**Conclusion**: For the FOMC corpus size (hundreds to low thousands of chunks), HNSW (ChromaDB's default) is the right choice. IVF becomes advantageous only at millions of vectors where memory constraints matter.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key for Claude |
| `FEDQUERY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `FEDQUERY_LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic` or `google`) |
| `FEDQUERY_LLM_MODEL` | `claude-sonnet-4-5-20250929` | Model identifier |
| `FEDQUERY_CHROMA_PATH` | `./data/chroma` | ChromaDB persistence path |
| `FEDQUERY_CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `FEDQUERY_CHUNK_OVERLAP` | `50` | Overlap between chunks in tokens |

## CLI Commands

```bash
fedquery ingest --years 2023 2024    # Download and index FOMC documents
fedquery ask "your question here"     # Ask a question with citations
fedquery benchmark                    # Run HNSW vs IVF comparison
```

## Project Structure

```
src/
├── agent/           # LangGraph workflow, state, nodes
├── cli/             # Typer CLI commands (ask, ingest, benchmark)
├── embedding/       # EmbeddingProvider interface + sentence-transformers
├── ingestion/       # Scraper, cleaner, chunker, pipeline
├── llm/             # LLM configuration (LangChain abstraction)
├── mcp_server/      # MCP server with search_fomc + get_document tools
├── models/          # Data models (FOMCDocument, DocumentChunk, Citation, etc.)
└── vectorstore/     # ChromaDB store + FAISS benchmark
config/              # Pydantic settings
tests/
├── contract/        # MCP tool contract tests
├── integration/     # Pipeline and Q&A flow tests
└── unit/            # Component unit tests
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

55 tests covering unit, integration, and contract tests. One FAISS IVF test is skipped on macOS ARM due to a known segfault when running alongside ChromaDB in the same process.

## Design Principles

1. **Local-First**: Everything runs locally except LLM API calls
2. **Open-Source Only**: No proprietary dependencies for core functionality
3. **Retrieval Correctness**: Precision over recall — better to say "I don't know" than fabricate
4. **Citation Grounding**: Every claim traceable to a specific source passage
5. **Simplicity**: Minimum complexity needed for the current requirements
