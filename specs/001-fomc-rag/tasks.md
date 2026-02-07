# Tasks: FOMC Agentic RAG Research Assistant

**Input**: Design documents from `/specs/001-fomc-rag/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Tests are included per the spec's testing requirements (pytest, contract tests for MCP tools, integration tests).

**Organization**: Tasks grouped by user story. US2 (Ingestion) is a prerequisite for US1 (Q&A) at runtime, but implementation can proceed independently with test fixtures.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3)
- Exact file paths included in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependency management, directory structure

- [x] T001 Create project directory structure per plan.md: `src/{models,ingestion,embedding,vectorstore,mcp_server,agent,llm,cli}/`, `tests/{contract,integration,unit}/`, `config/`
- [x] T002 Create `pyproject.toml` with all dependencies: langchain, langgraph, chromadb, faiss-cpu, sentence-transformers, mcp, typer, rich, beautifulsoup4, requests, pytest, anthropic, langchain-anthropic
- [x] T003 [P] Create `config/settings.py` with configuration management using pydantic-settings: ANTHROPIC_API_KEY, FEDQUERY_EMBEDDING_MODEL, FEDQUERY_LLM_PROVIDER, FEDQUERY_CHROMA_PATH defaults per quickstart.md
- [x] T004 [P] Create `.env.example` with all required/optional environment variables in `config/.env.example`
- [x] T005 [P] Create `src/__init__.py` and all subpackage `__init__.py` files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data models and provider interfaces that ALL user stories depend on

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create enum definitions (DocumentType, Confidence, IndexType) in `src/models/enums.py`
- [x] T007 [P] Create FOMCDocument dataclass in `src/models/document.py` per data-model.md: id, title, date, document_type, source_url, raw_text, ingested_at with validation
- [x] T008 [P] Create DocumentChunk dataclass in `src/models/chunk.py` per data-model.md: id, document_id, chunk_text, chunk_index, section_header, embedding, token_count with validation
- [x] T009 [P] Create Citation dataclass in `src/models/citation.py` per data-model.md: document_name, document_date, section_header, chunk_id, relevance_score, quoted_excerpt
- [x] T010 [P] Create Query and Answer dataclasses in `src/models/query.py` per data-model.md: includes confidence enum state transitions
- [x] T011 [P] Create BenchmarkResult dataclass in `src/models/benchmark.py` per data-model.md
- [x] T012 Create EmbeddingProvider abstract base class in `src/embedding/provider.py` per data-model.md: embed(texts) -> list[list[float]], dimension property
- [x] T013 Implement SentenceTransformerEmbeddingProvider in `src/embedding/sentence_transformer.py`: wraps all-MiniLM-L6-v2, configurable model name via settings
- [x] T014 Create LLM configuration module in `src/llm/config.py`: factory function returning BaseChatModel based on FEDQUERY_LLM_PROVIDER setting, default ChatAnthropic with temperature=0

**Checkpoint**: Foundation ready — all models, embedding provider, and LLM config available. User story implementation can begin.

---

## Phase 3: User Story 2 - Ingest and Index FOMC Documents (Priority: P2, but implemented first as prerequisite)

**Goal**: Pipeline that downloads FOMC documents from federalreserve.gov, cleans, chunks with section metadata, embeds, and stores in ChromaDB. Deduplication on re-ingestion.

**Independent Test**: Run `fedquery ingest --years 2024` and verify ChromaDB contains expected chunk count with valid embeddings and metadata.

**Why first**: Although P2 in priority, US1 (Q&A) requires an ingested corpus. Building ingestion first creates testable data for all subsequent stories.

### Tests for User Story 2

- [x] T015 [P] [US2] Unit test for FOMC scraper in `tests/unit/test_scraper.py`: test URL pattern generation, HTML parsing, document extraction for both statements and minutes
- [x] T016 [P] [US2] Unit test for chunking in `tests/unit/test_chunker.py`: test recursive split respects paragraph boundaries, section header regex captures known FOMC headers, overlap works correctly, ~512 token target
- [x] T017 [P] [US2] Unit test for deduplication in `tests/unit/test_dedup.py`: test that re-ingesting same source_url does not create duplicate documents
- [x] T018 [P] [US2] Integration test for full ingestion pipeline in `tests/integration/test_ingestion.py`: scrape → clean → chunk → embed → store, verify ChromaDB collection has correct metadata fields

### Implementation for User Story 2

- [x] T019 [P] [US2] Implement FOMC document scraper in `src/ingestion/scraper.py`: fetch calendar page from federalreserve.gov/monetarypolicy/fomccalendars.htm, extract statement and minutes URLs by year, download and parse HTML to clean text using BeautifulSoup4, return list of FOMCDocument objects
- [x] T020 [P] [US2] Implement text cleaner in `src/ingestion/cleaner.py`: strip HTML artifacts, normalize whitespace, gracefully skip/flag tables and non-textual content per edge case requirements
- [x] T021 [US2] Implement recursive chunker with section metadata in `src/ingestion/chunker.py`: recursive character split (~512 tokens, ~50 overlap), regex patterns for FOMC section headers (SECTION_PATTERNS from research.md), attach section_header to each DocumentChunk
- [x] T022 [US2] Implement ChromaDB vector store integration in `src/vectorstore/chroma_store.py`: create/connect to `fomc_chunks` collection with cosine distance, add_chunks() method storing embeddings + all 7 metadata fields per data-model.md, query() method returning ranked results, deduplication check by source_url
- [x] T023 [US2] Implement ingestion pipeline orchestrator in `src/ingestion/pipeline.py`: wire scraper → cleaner → chunker → embedding provider → chroma_store, accept year range parameter, report progress, handle errors per edge cases (non-textual content, connection errors)
- [x] T024 [US2] Implement `fedquery ingest` CLI command in `src/cli/ingest.py`: Typer command accepting --years parameter, calls pipeline orchestrator, Rich progress bar and completion summary

**Checkpoint**: User Story 2 complete. Run `fedquery ingest --years 2024` and verify ChromaDB contains indexed chunks with section metadata.

---

## Phase 4: User Story 1 - Ask a Grounded Question (Priority: P1) MVP

**Goal**: Agent answers natural language questions about FOMC monetary policy with citation-grounded responses. Handles uncertainty. All retrieval through MCP.

**Independent Test**: Run `fedquery ask "What did the Fed say about inflation in March 2024?"` and verify answer contains citations referencing specific FOMC documents and sections.

### Tests for User Story 1

- [x] T025 [P] [US1] Contract test for search_fomc MCP tool in `tests/contract/test_mcp_search.py`: verify response schema matches contracts/mcp-tools.md, test empty corpus error, test invalid query error, verify relevance_score range
- [x] T026 [P] [US1] Contract test for get_document MCP tool in `tests/contract/test_mcp_get_doc.py`: verify response schema matches contracts/mcp-tools.md, test not_found error
- [x] T027 [P] [US1] Unit test for confidence evaluation in `tests/unit/test_confidence.py`: test threshold logic (high ≥0.80, medium ≥0.60, low ≥0.40, insufficient <0.40) per agent-workflow.md
- [x] T028 [P] [US1] Integration test for full Q&A flow in `tests/integration/test_qa_flow.py`: test grounded answer with citations, test uncertainty response for off-topic query, test multi-document comparative query

### Implementation for User Story 1

- [x] T029 [US1] Implement MCP server with search_fomc tool in `src/mcp_server/server.py`: MCP Python SDK, @server.tool() for search_fomc(query, top_k) per contracts/mcp-tools.md, queries ChromaDB via chroma_store, returns ranked chunks with all response fields, handles empty_corpus and invalid_query errors
- [x] T030 [US1] Implement MCP server get_document tool in `src/mcp_server/server.py`: @server.tool() for get_document(doc_id) per contracts/mcp-tools.md, fetches full document from storage, handles not_found error
- [x] T031 [US1] Implement MCP server entry point and stdio transport in `src/mcp_server/server.py`: async main() with stdio_server, server registration config
- [x] T032 [US1] Implement AgentState TypedDict in `src/agent/state.py` per agent-workflow.md: query, retrieved_chunks, confidence, reformulation_attempts, reformulated_query, answer, citations, needs_retrieval
- [x] T033 [US1] Implement assess_query node in `src/agent/nodes.py`: LLM determines if retrieval is needed, sets needs_retrieval flag
- [x] T034 [US1] Implement search_corpus node in `src/agent/nodes.py`: calls search_fomc MCP tool via MCP client, populates retrieved_chunks in state
- [x] T035 [US1] Implement evaluate_confidence node in `src/agent/nodes.py`: compute average relevance_score from retrieved_chunks, map to confidence level per thresholds in agent-workflow.md
- [x] T036 [US1] Implement reformulate_query node in `src/agent/nodes.py`: LLM rephrases query based on original + retrieved context, increment reformulation_attempts, cap at 2
- [x] T037 [US1] Implement synthesize_answer node in `src/agent/nodes.py`: LLM generates answer grounded in retrieved_chunks with explicit citation instructions, produces answer text + Citation objects with quoted_excerpt
- [x] T038 [US1] Implement validate_citations node in `src/agent/nodes.py`: verify each citation's chunk_id exists in retrieved_chunks, verify quoted_excerpt is substring of chunk_text, remove invalid citations
- [x] T039 [US1] Implement respond node in `src/agent/nodes.py`: format final output — grounded answer with sources list OR uncertainty message per response formats in agent-workflow.md
- [x] T040 [US1] Implement LangGraph workflow in `src/agent/graph.py`: define StateGraph with all nodes, conditional edges per agent-workflow.md flow diagram, compile graph
- [x] T041 [US1] Implement MCP client integration in `src/agent/mcp_client.py`: spawn MCP server subprocess, connect via stdio transport, provide tool-calling interface for LangGraph nodes
- [x] T042 [US1] Implement `fedquery ask` CLI command in `src/cli/ask.py`: Typer command accepting question string, invokes LangGraph agent, Rich formatted output with confidence indicator and citation list
- [x] T043 [US1] Handle edge cases in agent workflow: empty vector store returns clear message, LLM API unreachable fails with clear error message, zero relevant passages triggers uncertainty response

**Checkpoint**: User Story 1 (MVP) complete. Full Q&A flow works: question → MCP retrieval → agent synthesis → cited answer.

---

## Phase 5: User Story 3 - Compare Vector Index Strategies (Priority: P3)

**Goal**: Benchmark comparing HNSW and IVF on FOMC corpus using FAISS, producing a report with latency, recall, and memory metrics.

**Independent Test**: Run `fedquery benchmark --index-types hnsw ivf` and verify report contains three measurable metrics for both index types.

### Tests for User Story 3

- [x] T044 [P] [US3] Unit test for benchmark runner in `tests/unit/test_benchmark.py`: test HNSW index build, test IVF index build, test recall calculation against brute-force baseline, test latency measurement, test memory measurement

### Implementation for User Story 3

- [x] T045 [US3] Implement FAISS index builder in `src/vectorstore/faiss_benchmark.py`: build IndexHNSWFlat (M=32, efSearch=64), build IndexIVFFlat (nlist=100, nprobe=10), build IndexFlatL2 brute-force baseline per research.md configurations
- [x] T046 [US3] Implement benchmark runner in `src/vectorstore/benchmark.py`: load embeddings from ChromaDB, build all three FAISS indexes, measure query latency (avg over 100 queries via time.perf_counter), measure recall@k (set intersection with brute-force), measure memory (tracemalloc peak RSS), produce BenchmarkResult objects
- [x] T047 [US3] Implement benchmark report formatter in `src/vectorstore/benchmark.py`: generate comparison table with latency, recall@k, memory for HNSW vs IVF, include trade-off explanation and guidance per SC-004
- [x] T048 [US3] Implement `fedquery benchmark` CLI command in `src/cli/benchmark.py`: Typer command accepting --index-types parameter, runs benchmark, Rich formatted output table

**Checkpoint**: User Story 3 complete. Benchmark produces comparison report with three metrics.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and validation across all stories

- [x] T049 Implement Typer CLI entry point in `src/cli/main.py`: register ingest, ask, benchmark subcommands, app-level help text
- [x] T050 Create `src/cli/__main__.py` for `python -m fedquery` execution
- [x] T051 Write README.md with design rationale (chunking, retrieval, agent behavior), grounding/citation enforcement, HNSW vs IVF understanding per constitution development workflow requirements
- [x] T052 Validate quickstart.md flow end-to-end: fresh venv → pip install → ingest → ask → verify answer with citations in under 10 minutes (SC-005)
- [x] T053 Add .gitignore entries for data/chroma/, .env, __pycache__/, *.egg-info/

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — BLOCKS all user stories
- **Phase 3 (US2 - Ingestion)**: Depends on Phase 2 — implemented first to create test corpus
- **Phase 4 (US1 - Q&A)**: Depends on Phase 2 for models/providers. At runtime depends on US2 for data, but implementation uses test fixtures
- **Phase 5 (US3 - Benchmark)**: Depends on Phase 2 for models. At runtime depends on US2 for corpus data
- **Phase 6 (Polish)**: Depends on all user stories being complete

### User Story Dependencies

- **US2 (Ingestion)**: No story dependencies. Foundational models + embedding provider only
- **US1 (Q&A)**: Runtime dependency on US2 (needs ingested data). Implementation can use test fixtures
- **US3 (Benchmark)**: Runtime dependency on US2 (needs embedded corpus in ChromaDB)

### Within Each User Story

- Tests written first (should fail before implementation)
- Models/state before services/nodes
- Individual nodes before graph assembly
- Core implementation before CLI integration
- Edge cases after happy path

### Parallel Opportunities

**Phase 2**: T007, T008, T009, T010, T011 can all run in parallel (independent model files)

**Phase 3 (US2)**: T015-T018 tests in parallel; T019, T020 in parallel (scraper + cleaner are independent)

**Phase 4 (US1)**: T025-T028 tests in parallel; T033-T039 nodes can partially parallel (assess_query, evaluate_confidence, validate_citations touch different concerns but share nodes.py — consider splitting to separate files for true parallelism)

**Phase 5 (US3)**: T044 test independent; T045 (index builder) independent of T047 (formatter)

---

## Parallel Example: Phase 2 (Foundational)

```
# Launch all model definitions in parallel:
Task: T007 "Create FOMCDocument model in src/models/document.py"
Task: T008 "Create DocumentChunk model in src/models/chunk.py"
Task: T009 "Create Citation model in src/models/citation.py"
Task: T010 "Create Query and Answer models in src/models/query.py"
Task: T011 "Create BenchmarkResult model in src/models/benchmark.py"
```

## Parallel Example: Phase 4 (US1 - Q&A)

```
# Launch all contract tests in parallel:
Task: T025 "Contract test for search_fomc in tests/contract/test_mcp_search.py"
Task: T026 "Contract test for get_document in tests/contract/test_mcp_get_doc.py"
Task: T027 "Unit test for confidence evaluation in tests/unit/test_confidence.py"
Task: T028 "Integration test for Q&A flow in tests/integration/test_qa_flow.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 — Q&A with Citations)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (models, providers)
3. Complete Phase 3: US2 Ingestion (creates corpus for testing)
4. Complete Phase 4: US1 Q&A (core value proposition)
5. **STOP and VALIDATE**: Ask a question, verify cited answer
6. Demo-ready at this point

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US2 (Ingestion) → Test: documents indexed in ChromaDB
3. Add US1 (Q&A) → Test: cited answers work → **MVP Demo!**
4. Add US3 (Benchmark) → Test: HNSW vs IVF report generated
5. Polish → README, quickstart validation, cleanup

### Suggested MVP Scope

**US2 + US1 = MVP**: Ingest FOMC documents and answer questions with citations. This covers the core value proposition and satisfies SC-001 through SC-003, SC-005, SC-006.

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [USn] label maps task to specific user story
- US2 is implemented before US1 despite lower priority because US1 needs data at runtime
- All MCP retrieval goes through MCP tools per FR-009 — agent nodes must NOT import chroma_store directly
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
