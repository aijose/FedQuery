# Implementation Plan: FOMC Agentic RAG Research Assistant

**Branch**: `001-fomc-rag` | **Date**: 2026-02-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-fomc-rag/spec.md`

## Summary

Build a local Agentic RAG prototype that ingests FOMC press statements and meeting minutes, stores them in a vector database with semantic embeddings, exposes retrieval via a local MCP server, and uses a LangGraph-orchestrated agent (Claude by default, LLM-agnostic) to answer analyst questions with citation-grounded responses. Includes an HNSW vs IVF benchmark comparison.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: LangChain, LangGraph, ChromaDB, FAISS, sentence-transformers, MCP Python SDK, Typer, Rich, BeautifulSoup4, requests
**Storage**: ChromaDB (local persistent vector store), FAISS (benchmark only)
**Testing**: pytest (unit + integration), contract tests for MCP tools
**Target Platform**: Local machine (macOS/Linux), CLI interface
**Project Type**: Single project
**Performance Goals**: <30s per question end-to-end (SC-001), <10 min setup for new user (SC-005)
**Constraints**: All local except LLM API calls, open-source dependencies only, ~2-3 hour prototype scope
**Scale/Scope**: 2-3 years of FOMC documents (~100-200 documents, ~5k-10k chunks)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Local-First** | PASS | ChromaDB runs locally, embeddings generated locally via sentence-transformers, MCP server is local process. Only external call is Anthropic API for LLM. |
| **II. Open-Source Only** | PASS | ChromaDB (Apache-2.0), FAISS (MIT), sentence-transformers (Apache-2.0), LangChain (MIT), LangGraph (MIT), MCP SDK (MIT), Typer (MIT), Rich (MIT), BeautifulSoup4 (MIT). |
| **III. Retrieval Correctness** | PASS | Agent workflow includes confidence evaluation node; low-confidence results trigger reformulation or uncertainty response. FR-006 enforced at graph level. |
| **IV. Citation Grounding** | PASS | Recursive chunking with section header metadata enables traceable citations (doc name + date + section + chunk). validate_citations node in agent graph verifies grounding. |
| **V. Simplicity** | PASS | Single project structure, CLI interface, recursive chunking (not complex parsing), provider interfaces only where justified (embedding swap, LLM swap). LangChain/LangGraph justified by branching workflow needs. |

**Gate Result**: ALL PASS — proceed to Phase 0.

## Design Decisions

| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| Vector DB | ChromaDB + FAISS (benchmark) | Simplest setup, Python-native, built-in persistence. FAISS needed for HNSW vs IVF comparison. | Qdrant (separate server), FAISS-only (no persistence) |
| Embedding | all-MiniLM-L6-v2 (swappable) | Lightest footprint, provider interface for future upgrades. Quality lever is chunking + prompting, not model size. | bge-small-en-v1.5, nomic-embed-text-v1.5 |
| Chunking | Recursive split + section metadata | Simple splitting with regex-captured FOMC section headers for meaningful citations. ~512 tokens, ~50 overlap. | Full section-aware parsing, semantic chunking |
| MCP Tools | search_fomc + get_document | Clean separation: broad retrieval + targeted drill-down. All retrieval via MCP per FR-009. | Single tool, three tools with list_documents |
| Agent Stack | LangChain + LangGraph | LLM-agnostic (swap Claude/Gemini via config). State machine handles branching: retrieval decisions, confidence eval, reformulation, citation validation. | LiteLLM + custom loop, Claude SDK directly |
| Interface | CLI (Typer + Rich) | Fast to build, easy to demo, scriptable. Rich terminal output with formatted citations. | Jupyter notebook, both |
| Data Source | Scrape federalreserve.gov | Authentic source, predictable URL patterns, demonstrates full ingestion pipeline. | Pre-bundled dataset, scrape + cache fallback |

## Project Structure

### Documentation (this feature)

```text
specs/001-fomc-rag/
├── plan.md              # This file
├── research.md          # Phase 0: Technology decisions and rationale
├── data-model.md        # Phase 1: Entity definitions and relationships
├── quickstart.md        # Phase 1: Developer setup guide
├── contracts/           # Phase 1: MCP tool and agent workflow contracts
│   ├── mcp-tools.md
│   └── agent-workflow.md
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/
├── models/              # Data models (FOMCDocument, DocumentChunk, Citation, etc.)
├── ingestion/           # Document scraping, cleaning, chunking, embedding pipeline
├── embedding/           # EmbeddingProvider interface and implementations
├── vectorstore/         # ChromaDB integration, FAISS benchmark utilities
├── mcp_server/          # MCP server with search_fomc and get_document tools
├── agent/               # LangGraph workflow (nodes, edges, state schema)
├── llm/                 # LangChain LLM provider configuration
└── cli/                 # Typer CLI commands (ingest, ask, mcp-server, benchmark)

tests/
├── contract/            # MCP tool contract tests
├── integration/         # End-to-end retrieval and agent tests
└── unit/                # Chunking, embedding, citation validation tests

config/                  # Configuration files (model settings, API keys template)
```

**Structure Decision**: Single project layout. All source under `src/` with domain-based module organization. Tests mirror source structure under `tests/`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| LangChain + LangGraph dependency | Agent workflow has conditional branching (retrieval decisions, confidence evaluation, reformulation retries, citation validation) that maps to a state machine | Simple while loop can't cleanly express conditional paths and retry logic without nested conditionals |
| FAISS as secondary vector DB | ChromaDB only supports HNSW; FAISS needed to benchmark IVF per FR-008 | Using only ChromaDB would not satisfy the HNSW vs IVF comparison requirement |
| EmbeddingProvider abstraction | User requires ability to swap embedding models without code changes | Hardcoding all-MiniLM-L6-v2 would require code changes to switch models |

## Post-Design Constitution Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| **I. Local-First** | PASS | No changes — all local except LLM API |
| **II. Open-Source Only** | PASS | All dependencies verified open-source with permissive licenses |
| **III. Retrieval Correctness** | PASS | Agent workflow enforces confidence evaluation before synthesis |
| **IV. Citation Grounding** | PASS | Section metadata in chunks + validate_citations node ensures traceability |
| **V. Simplicity** | PASS | Complexity additions justified in tracking table above |

**Post-Design Gate Result**: ALL PASS — plan is constitution-compliant.
