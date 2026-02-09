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
- **Embedder**: BAAI/bge-small-en-v1.5 (384 dimensions, 512 max_seq_length) via a swappable `EmbeddingProvider` interface.
- **Vector Store**: ChromaDB with cosine similarity, storing 7 metadata fields per chunk.

### Chunking Strategy

**Parameters**: ~512 tokens per chunk, ~50 token overlap. These were chosen through grid evaluation — 512 keeps enough context for FOMC policy language. The embedding model (bge-small-en-v1.5) supports 512 max_seq_length, so the full chunk text is embedded without truncation. The 50-token overlap ensures cross-chunk concepts (e.g., a policy rationale split across paragraphs) aren't lost at boundaries.

**Boundary handling**: Recursive character splitting tries paragraph breaks first (`\n\n`), then sentence breaks, then word boundaries. Overlap text snaps to word boundaries to avoid mid-word splits.

**Section header detection**: FOMC documents have predictable section headers. The chunker captures these via regex and attaches them as metadata:

```
SECTION_PATTERNS = [
    r"^(Participants'.+)",
    r"^(Committee Policy.+)",
    r"^(Developments in.+)",
    r"^(Staff Review.+)",
]
```

This enables human-readable citations without a full document parser — each chunk carries its section context (e.g., `§Committee Policy Action`).

### Retrieval Design

**Bi-encoder search**: Queries are embedded with bge-small-en-v1.5 (with a BGE-specific query instruction prefix) and matched against chunks via cosine similarity in ChromaDB. ChromaDB returns cosine *distance* in [0, 2], converted to similarity as `1.0 - distance / 2.0`.

**Two-pass date-filtered retrieval**: Temporal queries (e.g., "December 2024", "all of 2021") are detected by the `assess_query` LLM call, which extracts `date_start`/`date_end` hints. When present, `search_corpus` runs two passes: (1) a filtered pass with a ChromaDB `where` clause constraining `document_date` to the target range, then (2) an unfiltered pass. Results are merged with filtered chunks prioritized, deduped, and capped at `top_k`. This ensures date-specific content surfaces even when FOMC documents use near-identical template language across meetings.

**Adaptive retrieval count (`top_k_hint`)**: The `assess_query` node estimates how many search results are needed based on query scope. Single-meeting queries use the default (10), full-year queries request ~30 (to cover all ~8 FOMC meetings), and multi-year queries request 40-50. This prevents the bi-encoder's tendency to cluster results from a few meetings when documents share similar language.

**Cross-encoder reranking** (optional, `FEDQUERY_RERANKER_ENABLED=true`): When enabled, the retriever over-fetches 3x candidates from the bi-encoder, then reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2`. This improves precision at the cost of latency — the cross-encoder scores each (query, chunk) pair individually rather than comparing pre-computed embeddings.

**Embedding provider interface**: The `EmbeddingProvider` abstraction allows swapping models by changing the `FEDQUERY_EMBEDDING_MODEL` config variable. The provider supports model-specific query preprocessing (e.g., BGE instruction prefix) via the `embed_query()` method.

### Agent Workflow (LangGraph)

```
                        ┌─── no retrieval needed ───→ respond
                        │
assess_query ───────────┤
 (date hints,           │
  top_k_hint)           └─── needs retrieval ──→ search_corpus → evaluate_confidence
                                                 (two-pass if         │
                                                  date hints)  ┌──────┼──────────────────┐
                                                               │      │                   │
                                                         confidence ≥ confidence =     insufficient
                                                          "medium"      "low" &        or max retries
                                                               │     attempts < 2          │
                                                               ▼          ▼                ▼
                                                      synthesize_answer  reformulate     respond
                                                               │          _query      (uncertainty)
                                                               ▼          │
                                                      validate_citations  └→ search_corpus
                                                               │
                                                               ▼
                                                            respond
```

- **Query assessment**: The LLM classifies the query, extracts date ranges for temporal filtering, and estimates `top_k_hint` (how many results to retrieve based on query scope)
- **Confidence thresholds**: high ≥ 0.55, medium ≥ 0.40, low ≥ 0.25, insufficient < 0.25 (may need recalibration after embedding model changes)
- **Reformulation**: When confidence is low, the agent rephrases the query and retries (max 2 attempts)
- **Uncertainty handling**: When evidence is insufficient, the system explicitly says so rather than fabricating an answer

### Citation Grounding

Citations are enforced through a three-stage pipeline:

1. **`synthesize_answer`**: The LLM is prompted to cite sources as `[Source N]` references. After generation, the node parses which `[Source N]` markers actually appear in the output text — including comma-separated groups like `[Source 1, Source 2, Source 3]` — and builds citations *only* for referenced chunks, ordered by first appearance.
2. **`validate_citations`**: Each citation's `chunk_id` is verified against the set of retrieved chunks. Citations referencing unknown chunk IDs are dropped.
3. **`respond`**: Source references are remapped to sequential numbering (`[1]`, `[2]`, ...) matching the Sources footer, so the final output never has gaps like `[1], [3]`.

This means the system cannot cite a source it didn't retrieve, and it won't list sources in the footer that aren't actually referenced in the answer text.

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
| `search_fomc(query, top_k, where?)` | Semantic search across the vector store, with optional metadata filter |
| `get_document(doc_id)` | Fetch full document content by ID |

By default (`FEDQUERY_USE_MCP=true`), the CLI spawns the MCP server as a subprocess and communicates via stdio — the same pattern Claude Desktop uses for local MCP tools. The server loads the embedding model and ChromaDB store in its own process, and the agent issues `search_fomc` / `get_document` tool calls over the MCP protocol.

Set `FEDQUERY_USE_MCP=false` for a direct in-process mode that skips the subprocess (faster for development/debugging, same retrieval results). Use `--verbose` to see which mode is active.

## HNSW vs IVF Benchmark

```bash
fedquery benchmark
```

Compares two approximate nearest neighbor (ANN) index types using FAISS on the full 5-year FOMC corpus (1,545 chunks, 384 dimensions).

**HNSW** (Hierarchical Navigable Small World) builds a multi-layer graph where each vector is a node connected to its approximate nearest neighbors. Queries navigate from coarse upper layers to fine lower layers, like a skip list over geometric space. Tuning knobs: `M` (edges per node — graph density), `efSearch` (beam width — recall/latency tradeoff). Higher M = better recall + more memory; higher efSearch = better recall + higher latency.

**IVF** (Inverted File Index) partitions the vector space into Voronoi cells using k-means clustering. At query time, it only searches the `nprobe` nearest cells rather than the full dataset. Tuning knobs: `nlist` (number of cells — partition granularity), `nprobe` (cells searched — recall/speed tradeoff). When `nprobe = nlist`, IVF degrades to brute-force. When `nprobe = 1`, recall drops to ~40-54% because relevant vectors in adjacent cells are missed entirely.

### Benchmark Results (1,545 chunks)

| Metric | HNSW (M=32 ef=64) | IVF (nlist=25 nprobe=10) | Brute-Force |
|--------|-------------------|--------------------------|-------------|
| Recall@5 | 0.933 | 0.959 | 1.000 |
| Avg Latency | 0.030ms | 0.017ms | 0.031ms |
| P99 Latency | 0.057ms | 0.022ms | 0.036ms |
| Index Size | 2,727 KB | 2,367 KB | 2,318 KB |
| Real Query Recall@5 | 1.000 | 1.000 | 1.000 |

Both hit 100% recall on real FOMC queries — the recall gap only appears on random vectors where geometric edge cases matter. Full parameter sweeps (HNSW: M={16,32,64} x efSearch={32,64,128}; IVF: nlist={10,25,39} x nprobe={1,5,10,all}) are in [`optimization_efforts.md`](optimization_efforts.md).

**Why HNSW is the right default**: ChromaDB uses HNSW internally — no separate index to build or sync. HNSW scales without retuning as the corpus grows (IVF needs centroid retraining). And at this corpus size, all methods are sub-millisecond — thousands of times faster than the LLM API call. IVF's memory advantage only matters at millions of vectors.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key for Claude |
| `FEDQUERY_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `FEDQUERY_LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic` or `google`) |
| `FEDQUERY_LLM_MODEL` | `claude-sonnet-4-5-20250929` | Model identifier |
| `FEDQUERY_CHROMA_PATH` | `./data/chroma` | ChromaDB persistence path |
| `FEDQUERY_CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `FEDQUERY_CHUNK_OVERLAP` | `50` | Overlap between chunks in tokens |
| `FEDQUERY_RERANKER_ENABLED` | `false` | Enable cross-encoder reranking |
| `FEDQUERY_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model |
| `FEDQUERY_USE_MCP` | `true` | Use MCP server subprocess (`false` for direct mode) |

## CLI Commands

```bash
fedquery ingest --years 2023 2024    # Download and index FOMC documents
fedquery ask "your question here"     # Ask a question with citations
fedquery evaluate                     # Run retrieval quality evaluation
fedquery benchmark                    # Run HNSW vs IVF comparison
```

## Project Structure

```
src/
├── agent/           # LangGraph workflow, state, nodes
├── cli/             # Typer CLI commands (ask, ingest, evaluate, benchmark)
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

192 tests covering unit, integration, contract, and MCP round-trip tests. One FAISS IVF test is skipped on macOS ARM due to a known segfault when running alongside ChromaDB in the same process.

## Design Principles

1. **Local-First**: Everything runs locally except LLM API calls
2. **Open-Source Only**: No proprietary dependencies for core functionality
3. **Retrieval Correctness**: Precision over recall — better to say "I don't know" than fabricate
4. **Citation Grounding**: Every claim traceable to a specific source passage
5. **Simplicity**: Minimum complexity needed for the current requirements
