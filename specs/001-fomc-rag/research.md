# Phase 0 Research: Technology Decisions

> FOMC Agentic RAG — FedQuery Project

This document captures all technology decisions made during Phase 0 research, including rationale and alternatives considered for each choice.

---

## 1. Vector Database

| Aspect | Detail |
|--------|--------|
| **Decision** | ChromaDB (primary) + FAISS (benchmark only) |
| **Version** | chromadb >= 0.4, faiss-cpu >= 1.7 |

### Rationale

ChromaDB is the simplest path to a working vector store. It installs via `pip install chromadb`, is Python-native, and includes built-in persistence to disk. No external server process is required. This keeps the development loop tight and the deployment footprint minimal.

FAISS is included solely for the HNSW vs IVF indexing benchmark (see Section 9). ChromaDB only exposes HNSW internally, so FAISS is needed to build an IVF index on the same corpus for a fair comparison.

### Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Qdrant** | More powerful query features, filtering, production-grade | Requires separate server process (`docker run qdrant/qdrant`) | Overhead not justified for prototype scope |
| **FAISS-only** | Maximum control, widely used in research | No built-in persistence, no metadata storage, requires manual serialization | Too low-level; would need to build metadata layer from scratch |
| **Pinecone / Weaviate** | Managed, scalable | Cloud dependency, API keys, network latency | Violates local-first prototype constraint |

---

## 2. Embedding Model

| Aspect | Detail |
|--------|--------|
| **Decision** | `all-MiniLM-L6-v2` via `sentence-transformers`, behind a swappable `EmbeddingProvider` interface |
| **Dimensions** | 384 |
| **Model size** | ~80 MB |

### Rationale

`all-MiniLM-L6-v2` is the fastest general-purpose sentence embedding model with broad community documentation. The 384-dimensional output keeps vector storage compact and similarity search fast.

A provider interface (`EmbeddingProvider`) abstracts the model choice behind a config option, allowing upgrade to higher-quality models later without code changes.

FOMC documents are formal English prose. While the language is domain-specific (monetary policy terminology), general-purpose models handle it reasonably well. Chunking strategy and prompt engineering are bigger quality levers than model selection at this stage.

### Alternatives Considered

| Model | Dims | Size | Quality | Why Not (for now) |
|-------|------|------|---------|-------------------|
| `BAAI/bge-small-en-v1.5` | 384 | ~130 MB | Better retrieval on MTEB benchmarks | Viable upgrade candidate; slightly less documented |
| `nomic-embed-text-v1.5` | 768 | ~550 MB | Best quality among small models | 7x larger, 768-dim increases storage and search cost |
| OpenAI `text-embedding-3-small` | 1536 | API | High quality | Cloud dependency, cost per query, latency |

### Upgrade Path

Change one config value to swap models:

```yaml
embedding:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2  # swap to bge-small-en-v1.5 or nomic-embed-text-v1.5
```

---

## 3. Chunking Strategy

| Aspect | Detail |
|--------|--------|
| **Decision** | Recursive character split (~512 tokens, ~50 token overlap) with section header metadata via regex |
| **Overlap** | ~50 tokens (10% of chunk size) |

### Rationale

Recursive character splitting is simple and proven. It respects paragraph boundaries before falling back to sentence and word boundaries.

FOMC documents have predictable, consistent section headers:
- "Participants' Views on Current Conditions and the Economic Outlook"
- "Committee Policy Action"
- "Developments in Financial Markets and Open Market Operations"

These headers are captured with lightweight regex patterns and stored as chunk metadata. This enables human-readable citations without building a full document parser:

```
FOMC Minutes, Jan 2024, §Economic Outlook, Chunk 2
```

This citation format satisfies Constitution Principle IV (traceable, human-readable sourcing).

### Alternatives Considered

| Strategy | Pros | Cons | Why Not |
|----------|------|------|---------|
| **Full section-aware splitting** | Best citation fidelity, natural document boundaries | Requires custom parser per document type (statements vs minutes vs speeches) | Too much parsing code for prototype |
| **Semantic chunking** | Adaptive chunk boundaries based on topic shifts | Adds embedding cost at chunking time, harder to debug | Complexity not justified at this stage |
| **Fixed-size splitting** | Simplest implementation | Splits mid-sentence, poor retrieval quality | Quality too low |

### Section Header Patterns

```python
SECTION_PATTERNS = [
    r"^#{1,3}\s+(.+)",                          # Markdown headers
    r"^(Participants'.+|Committee Policy.+)",     # FOMC-specific sections
    r"^(Developments in.+|Staff Review.+)",       # Additional known sections
]
```

---

## 4. MCP Server Design

| Aspect | Detail |
|--------|--------|
| **Decision** | Two tools: `search_fomc(query, top_k)` and `get_document(doc_id)` |
| **SDK** | `mcp` Python SDK (modelcontextprotocol/python-sdk) |
| **Transport** | stdio (local communication) |

### Rationale

Two tools provide clean separation of concerns:

- **`search_fomc(query, top_k)`** -- Semantic search across the vector store. Returns ranked chunks with metadata and similarity scores. This is the primary retrieval path.
- **`get_document(doc_id)`** -- Fetch full document context by ID. Allows the agent to drill into a specific document after search surfaces relevant chunks. Supports the "retrieve then deep-read" pattern.

All retrieval goes through MCP per requirement FR-009, making the vector store an implementation detail hidden behind the tool interface.

The `mcp` Python SDK handles stdio transport for local server-client communication. No HTTP server needed.

### Alternatives Considered

| Design | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Single `search` tool** | Simplest | No way to fetch full document context after search | Too limiting for multi-turn reasoning |
| **Three tools (+ `list_documents`)** | Complete CRUD | `list_documents` unnecessary for prototype; agent never needs to browse without a query | Scope creep |
| **Resource-based (MCP resources)** | More RESTful | Resources are for static content; search is dynamic | Misuse of MCP abstraction |

### MCP Server Implementation Notes

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("fomc-rag")

@server.tool()
async def search_fomc(query: str, top_k: int = 5) -> list[dict]:
    """Search FOMC documents by semantic similarity."""
    ...

@server.tool()
async def get_document(doc_id: str) -> dict:
    """Retrieve full document content and metadata."""
    ...

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)
```

---

## 5. Agent Orchestration

| Aspect | Detail |
|--------|--------|
| **Decision** | LangChain + LangGraph |
| **LangChain role** | LLM-agnostic abstraction layer |
| **LangGraph role** | State machine for branching agent workflow |

### Rationale

**LangChain** provides the LLM abstraction. Swapping between Claude and Gemini is a config change, not a code change. This directly supports the LLM-agnostic requirement.

**LangGraph** provides a state machine / directed graph for the agent workflow. The FOMC RAG workflow has conditional branching that maps cleanly to a graph:

```
assess_query
    -> decide_retrieval (needs search? or direct answer?)
        -> search (via MCP)
            -> evaluate_confidence
                -> [low confidence] reformulate -> retry search
                -> [sufficient confidence] synthesize
                    -> validate_citations
                        -> respond
```

This is not a linear pipeline. The confidence evaluation node (FR-006) conditionally loops back to reformulation, which is natural to express as a graph edge but awkward in a simple loop.

### Agent State Schema

```python
class AgentState(TypedDict):
    query: str                          # Original user query
    retrieved_chunks: list[dict]        # Chunks from MCP search
    confidence_score: float             # 0.0-1.0, from LLM self-assessment
    reformulation_attempts: int         # Cap at 2 retries
    reformulated_query: str | None      # Rewritten query for retry
    final_answer: str                   # Synthesized response
    citations: list[Citation]           # Validated source references
```

### Workflow Graph Nodes

| Node | Responsibility | Transitions |
|------|---------------|-------------|
| `assess_query` | Classify query type, extract key terms | -> `decide_retrieval` |
| `decide_retrieval` | Determine if search is needed | -> `search` or -> `synthesize` |
| `search` | Call `search_fomc` via MCP | -> `evaluate_confidence` |
| `evaluate_confidence` | LLM scores retrieval relevance (FR-006) | -> `reformulate` or -> `synthesize` |
| `reformulate` | Rewrite query for better retrieval | -> `search` (max 2 retries) |
| `synthesize` | Generate answer from retrieved context | -> `validate_citations` |
| `validate_citations` | Verify each citation maps to a real chunk | -> `respond` |
| `respond` | Format final output with citations | terminal |

### Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **LiteLLM + custom loop** | Lighter dependency, simple | Branching logic gets messy in a while loop; no built-in state management | Doesn't handle conditional graph well |
| **Claude SDK directly** | Lowest latency, native tool use | Locked to one LLM provider | Not LLM-agnostic |
| **AutoGen** | Multi-agent patterns | Heavy framework, complex setup | Overkill for single-agent RAG |
| **CrewAI** | Role-based agents | Multi-agent focus; single agent is simpler | Wrong abstraction level |

---

## 6. User Interface

| Aspect | Detail |
|--------|--------|
| **Decision** | CLI using Typer + Rich |
| **Output** | Markdown rendering, colored citations, confidence indicators |

### Rationale

A CLI is the fastest path to a working demo. Typer provides argument parsing with type hints. Rich provides terminal markdown rendering, syntax highlighting, and colored output for citations.

The CLI is scriptable (pipe queries from a file), easy to demo in a terminal session, and closer to a production interaction pattern than a notebook.

### Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Jupyter notebook** | Visual, good for presentations, inline charts | Not production-like, harder to script | Better as secondary demo format, not primary |
| **Streamlit** | Web UI, interactive | Adds web framework dependency, deployment concerns | Out of scope for prototype |
| **Both CLI + notebook** | Covers all audiences | Splits development effort across two interfaces | Focus on one, do it well |

### CLI Interaction Example

```
$ fedquery "What did the FOMC say about inflation in January 2024?"

+-- FedQuery ----------------------------------------+
| Confidence: 0.87                                   |
|                                                    |
| The January 2024 FOMC minutes indicate that...     |
|                                                    |
| Sources:                                           |
|  [1] FOMC Minutes, Jan 2024, §Economic Outlook, C3|
|  [2] FOMC Statement, Jan 31 2024, P2               |
+----------------------------------------------------+
```

---

## 7. Data Source

| Aspect | Detail |
|--------|--------|
| **Decision** | Scrape from federalreserve.gov |
| **Libraries** | `requests` + `BeautifulSoup4` |
| **Document types** | Press statements, meeting minutes |

### Rationale

The Federal Reserve website is the authentic, authoritative source for FOMC documents. Statements and minutes are published as HTML at predictable URL patterns. Scraping demonstrates a real ingestion pipeline rather than relying on a pre-bundled dataset.

### FOMC Document URL Patterns

```
# Press statements
https://www.federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm

# Meeting minutes
https://www.federalreserve.gov/monetarypolicy/fomcminutes{YYYYMMDD}.htm

# Minutes calendar (index page)
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
```

Minutes are typically published approximately 3 weeks after each FOMC meeting. The calendar page provides an index of all meetings and links to associated documents.

### Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Pre-bundled dataset** | Zero scraping code, instant setup | Doesn't demonstrate ingestion pipeline | Misses a key project deliverable |
| **Scrape + cache with fallback** | More robust, handles rate limits | More code for error handling and caching | Adds complexity; simple retry is sufficient for prototype |
| **FRED API** | Structured data access | Covers economic data, not FOMC text documents | Wrong data type |

---

## 8. MCP Server Runtime Details

### Python SDK Setup

The `mcp` Python SDK (`modelcontextprotocol/python-sdk`) is the official implementation for building MCP servers in Python.

```bash
pip install mcp
```

### Transport: stdio

For local development, the MCP server communicates over stdio (stdin/stdout). The client (LangGraph agent) spawns the server as a subprocess and communicates via JSON-RPC over stdio pipes. No HTTP, no ports, no network configuration.

### Server Registration

The client configuration points to the server script:

```json
{
  "mcpServers": {
    "fomc-rag": {
      "command": "python",
      "args": ["src/mcp_server/server.py"]
    }
  }
}
```

---

## 9. HNSW vs IVF Benchmark Approach

### Purpose

Compare two approximate nearest neighbor (ANN) index types on the FOMC corpus to validate the default ChromaDB (HNSW) choice and document the trade-offs.

### Methodology

Use FAISS to build both index types on the identical embedded corpus. Measure three metrics against a brute-force baseline (FAISS `IndexFlatL2`).

| Metric | Measurement | Tool |
|--------|-------------|------|
| **Latency** | Average query time over 100 queries (ms) | `time.perf_counter` |
| **Recall@k** | Overlap of top-k results with brute-force top-k | Set intersection |
| **Memory** | Peak RSS during index load | `tracemalloc` |

### Index Configurations

```python
# HNSW (what ChromaDB uses internally)
index_hnsw = faiss.IndexHNSWFlat(dim, 32)  # M=32 connections
index_hnsw.hnsw.efSearch = 64

# IVF (inverted file index)
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist=100)
index_ivf.nprobe = 10

# Brute-force baseline
index_flat = faiss.IndexFlatL2(dim)
```

### Expected Outcome

For the FOMC corpus size (hundreds to low thousands of documents), both HNSW and IVF should achieve >95% recall. HNSW typically has lower latency at small scale. IVF becomes advantageous at millions of vectors. This benchmark validates that ChromaDB's HNSW default is appropriate for the project's scale.

---

## Summary Decision Matrix

| Component | Decision | Key Rationale |
|-----------|----------|---------------|
| Vector DB | ChromaDB + FAISS (benchmark) | Simplest setup, Python-native, built-in persistence |
| Embedding | all-MiniLM-L6-v2 + provider interface | Fast, small, swappable |
| Chunking | Recursive split + section regex | Simple splitting, human-readable citations |
| MCP Server | 2 tools (search + get_document) | Clean separation, covers retrieval patterns |
| Orchestration | LangChain + LangGraph | LLM-agnostic + branching workflow support |
| UI | Typer + Rich CLI | Fast to build, scriptable, production-like |
| Data Source | Scrape federalreserve.gov | Authentic source, demonstrates pipeline |
| ANN Index | HNSW (default) with IVF benchmark | Validated via controlled comparison |
