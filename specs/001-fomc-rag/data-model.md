# Data Model: FOMC Agentic RAG Research Assistant

**Feature Branch**: `001-fomc-rag`
**Created**: 2026-02-06

---

## Entity Relationship Diagram

```
+------------------+       1:N       +------------------+
|  FOMCDocument    |---------------->|  DocumentChunk   |
|                  |                 |                  |
| id (PK)          |                 | id (PK)          |
| title             |                 | document_id (FK) |
| date              |                 | chunk_text       |
| document_type     |                 | chunk_index      |
| source_url        |                 | section_header   |
| raw_text          |                 | embedding        |
| ingested_at       |                 | token_count      |
+------------------+                 +------------------+
                                            ^
                                            | N:1
                                            |
+------------------+       1:1       +------------------+       1:N       +------------------+
|     Query        |---------------->|     Answer       |---------------->|    Citation      |
|                  |                 |                  |                 |                  |
| id (PK)          |                 | id (PK)          |                 | document_name    |
| question_text     |                 | query_id (FK)    |                 | document_date    |
| timestamp         |                 | response_text    |                 | section_header   |
+------------------+                 | citations        |                 | chunk_id (FK) ---|----> DocumentChunk
                                     | confidence       |                 | relevance_score  |
                                     | created_at       |                 | quoted_excerpt   |
                                     +------------------+                 +------------------+


+------------------+
| BenchmarkResult  |  (standalone, no FK relationships)
|                  |
| id (PK)          |
| index_type       |
| query_latency_ms |
| recall_at_k      |
| memory_usage_mb  |
| corpus_size      |
| created_at       |
+------------------+
```

---

## Entities

### 1. FOMCDocument

A press statement or meeting minutes document published by the Federal Reserve.

| Field          | Type       | Constraints                                       | Description                                |
|----------------|------------|---------------------------------------------------|--------------------------------------------|
| `id`           | `str`      | PK, UUID v4                                       | Unique identifier                          |
| `title`        | `str`      | NOT NULL                                          | Document title (e.g., "FOMC Statement - March 2024") |
| `date`         | `date`     | NOT NULL, must be a valid calendar date           | Publication date of the document           |
| `document_type`| `str`      | NOT NULL, enum: `"statement"` or `"minutes"`      | Type of FOMC document                      |
| `source_url`   | `str`      | NOT NULL, must be a valid URL                     | URL where the document was downloaded from |
| `raw_text`     | `str`      | NOT NULL                                          | Full cleaned text content of the document  |
| `ingested_at`  | `datetime` | NOT NULL, auto-set on creation                    | Timestamp when the document was ingested   |

**Relationships**:
- Has many `DocumentChunk` (one-to-many via `DocumentChunk.document_id`)

**Validation Rules**:
- `document_type` must be one of `"statement"` or `"minutes"`
- `date` must be a valid calendar date
- `source_url` must be a well-formed URL (scheme + host at minimum)
- Duplicate detection: a document with the same `source_url` must not be ingested twice (supports re-ingestion without duplication per User Story 2, Scenario 3)

---

### 2. DocumentChunk

A segment of an FOMC document sized for embedding and retrieval. Chunks are produced during the ingestion pipeline by splitting `FOMCDocument.raw_text` using a text-based strategy (paragraph or fixed token window with overlap).

| Field            | Type           | Constraints                                          | Description                                              |
|------------------|----------------|------------------------------------------------------|----------------------------------------------------------|
| `id`             | `str`          | PK, UUID v4                                          | Unique identifier                                        |
| `document_id`    | `str`          | FK to `FOMCDocument.id`, NOT NULL                    | Parent document reference                                |
| `chunk_text`     | `str`          | NOT NULL, must not be empty                          | Text content of this chunk                               |
| `chunk_index`    | `int`          | NOT NULL, >= 0                                       | Zero-based position of this chunk within its document    |
| `section_header` | `str` or null  | Nullable                                             | Section heading captured via regex during chunking       |
| `embedding`      | `list[float]`  | NOT NULL, length must equal 384                      | Dense vector from `all-MiniLM-L6-v2` embedding model    |
| `token_count`    | `int`          | NOT NULL, > 0                                        | Token count of `chunk_text` (used for context budgeting) |

**Relationships**:
- Belongs to one `FOMCDocument` (many-to-one via `document_id`)
- Referenced by many `Citation` (via `Citation.chunk_id`)

**Validation Rules**:
- `chunk_text` must not be empty string
- `chunk_index` must be >= 0
- `embedding` dimension must be 384 (matching `all-MiniLM-L6-v2` output dimension)
- `(document_id, chunk_index)` should be unique (no duplicate chunk positions within a document)

---

### 3. Query

A natural language question submitted by the analyst.

| Field           | Type       | Constraints                    | Description                          |
|-----------------|------------|--------------------------------|--------------------------------------|
| `id`            | `str`      | PK, UUID v4                   | Unique identifier                    |
| `question_text` | `str`      | NOT NULL, must not be empty    | The analyst's natural language query |
| `timestamp`     | `datetime` | NOT NULL, auto-set on creation | When the query was submitted         |

**Relationships**:
- Has one `Answer` (one-to-one via `Answer.query_id`)

---

### 4. Answer

A synthesized response grounded in retrieved chunks. The agent produces this after retrieving relevant passages via the MCP server and synthesizing them through the LLM.

| Field           | Type             | Constraints                                                     | Description                                        |
|-----------------|------------------|-----------------------------------------------------------------|----------------------------------------------------|
| `id`            | `str`            | PK, UUID v4                                                    | Unique identifier                                  |
| `query_id`      | `str`            | FK to `Query.id`, NOT NULL, unique                             | The query this answer responds to                  |
| `response_text` | `str`            | NOT NULL                                                       | Synthesized answer text                            |
| `citations`     | `list[Citation]` | NOT NULL (may be empty list)                                   | Ordered list of citations supporting the answer    |
| `confidence`    | `str`            | NOT NULL, enum: `"high"` / `"medium"` / `"low"` / `"insufficient"` | Confidence level of the answer                |
| `created_at`    | `datetime`       | NOT NULL, auto-set on creation                                 | When the answer was generated                      |

**Relationships**:
- Belongs to one `Query` (one-to-one via `query_id`)
- Has many `Citation` (embedded list)

**State Transitions**:
- The `confidence` field drives agent behavior per FR-006:
  - `"high"`: Answer is well-supported by multiple relevant passages
  - `"medium"`: Answer is partially supported; some claims may lack strong grounding
  - `"low"`: Answer is weakly supported; user should verify claims
  - `"insufficient"`: Triggers uncertainty communication -- the system explicitly states that retrieved passages are insufficient to answer the question rather than generating unsupported claims

---

### 5. Citation

A reference linking a claim in an answer to a specific source chunk. Enables the traceability requirement (FR-005).

| Field             | Type          | Constraints                                           | Description                                         |
|-------------------|---------------|-------------------------------------------------------|-----------------------------------------------------|
| `document_name`   | `str`         | NOT NULL                                              | Human-readable document title for display           |
| `document_date`   | `date`        | NOT NULL                                              | Publication date of the source document              |
| `section_header`  | `str` or null | Nullable                                              | Section heading if available from chunk metadata     |
| `chunk_id`        | `str`         | FK to `DocumentChunk.id`, NOT NULL                    | The specific chunk this citation references          |
| `relevance_score` | `float`       | NOT NULL, range [0.0, 1.0]                            | Cosine similarity or retrieval relevance score       |
| `quoted_excerpt`  | `str`         | NOT NULL, must be substring of referenced chunk_text  | Verbatim excerpt from the source passage             |

**Relationships**:
- Belongs to one `Answer` (embedded in `Answer.citations` list)
- References one `DocumentChunk` (via `chunk_id`)

**Validation Rules**:
- `relevance_score` must be between 0.0 and 1.0 inclusive
- `quoted_excerpt` must be a substring of the `chunk_text` of the `DocumentChunk` referenced by `chunk_id`

---

### 6. BenchmarkResult

Results from the HNSW vs IVF index comparison benchmark (FR-008). This entity is standalone with no foreign key relationships to the core RAG entities.

| Field              | Type       | Constraints                              | Description                                    |
|--------------------|------------|------------------------------------------|------------------------------------------------|
| `id`               | `str`      | PK, UUID v4                              | Unique identifier                              |
| `index_type`       | `str`      | NOT NULL, enum: `"hnsw"` or `"ivf"`     | Which index strategy was benchmarked           |
| `query_latency_ms` | `float`    | NOT NULL, >= 0                           | Average query latency in milliseconds          |
| `recall_at_k`      | `float`    | NOT NULL, range [0.0, 1.0]              | Recall@k metric (fraction of true top-k found) |
| `memory_usage_mb`  | `float`    | NOT NULL, >= 0                           | Memory footprint of the index in megabytes     |
| `corpus_size`      | `int`      | NOT NULL, > 0                            | Number of chunks in the corpus during benchmark |
| `created_at`       | `datetime` | NOT NULL, auto-set on creation           | When the benchmark was run                     |

---

## ChromaDB Collection Schema

ChromaDB is the local vector database used for storing and retrieving document chunks. The following describes how the data model maps to ChromaDB's collection abstraction.

### Collection: `fomc_chunks`

ChromaDB stores documents as a flat collection of items, each with an ID, an embedding vector, a text document, and arbitrary metadata.

| ChromaDB Field  | Source                          | Description                                          |
|-----------------|---------------------------------|------------------------------------------------------|
| `id`            | `DocumentChunk.id`              | Unique chunk ID (UUID string)                        |
| `embedding`     | `DocumentChunk.embedding`       | 384-dim float vector from `all-MiniLM-L6-v2`        |
| `document`      | `DocumentChunk.chunk_text`      | The chunk text stored for retrieval and display      |
| `metadata`      | (see below)                     | Structured metadata for filtering and traceability   |

**Metadata fields stored per chunk**:

| Metadata Key      | Type   | Source                            | Description                                  |
|--------------------|--------|-----------------------------------|----------------------------------------------|
| `document_id`      | `str`  | `FOMCDocument.id`                 | Parent document UUID                         |
| `document_title`   | `str`  | `FOMCDocument.title`              | For citation display without extra lookups   |
| `document_date`    | `str`  | `FOMCDocument.date` (ISO 8601)   | For date-range filtering in queries          |
| `document_type`    | `str`  | `FOMCDocument.document_type`      | `"statement"` or `"minutes"` for filtering   |
| `chunk_index`      | `int`  | `DocumentChunk.chunk_index`       | Ordering within document                     |
| `section_header`   | `str`  | `DocumentChunk.section_header`    | Section heading if captured (empty string if null) |
| `token_count`      | `int`  | `DocumentChunk.token_count`       | For context window budgeting                 |

**Query pattern**: The MCP server's retrieval tool calls `collection.query()` with the embedded query vector and optional metadata filters (e.g., `document_type`, date range). ChromaDB returns the top-k results ranked by cosine similarity, which are then mapped to `Citation` objects for the answer.

---

## Provider Interfaces

### EmbeddingProvider

Abstraction over the embedding model used to convert text into dense vectors. The default implementation wraps `sentence-transformers/all-MiniLM-L6-v2`.

```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Interface for text embedding generation."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors. Each vector is a list of floats
            with dimension matching the model configuration (384 for
            all-MiniLM-L6-v2).

        Raises:
            ValueError: If texts is empty.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (e.g., 384)."""
        ...
```

**Default implementation**: `SentenceTransformerEmbeddingProvider` wrapping `all-MiniLM-L6-v2` (384 dimensions, runs locally).

---

### LLMProvider (via LangChain)

The LLM used for answer synthesis is accessed through LangChain's `BaseChatModel` abstraction. This allows swapping between providers (Anthropic Claude, OpenAI, local models) without changing agent logic.

```python
# LangChain's BaseChatModel is the abstraction layer.
# The agent uses it through LangChain's standard interface:

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic

# Default configuration (from spec assumptions):
llm: BaseChatModel = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
)

# The agent calls the LLM via LangChain's invoke/stream interface:
# response = llm.invoke(messages)
```

**Key points**:
- The agent orchestration layer depends on `BaseChatModel`, not on a specific provider
- Default provider is Anthropic Claude (accessed via API key per spec assumptions)
- `temperature=0` for deterministic, grounded answers
- LangChain handles prompt formatting, retry logic, and streaming

---

## Enum Definitions

For reference, the enumerations used across entities:

| Enum Name        | Values                                          | Used By                     |
|------------------|------------------------------------------------|-----------------------------|
| `DocumentType`   | `"statement"`, `"minutes"`                     | `FOMCDocument.document_type`|
| `Confidence`     | `"high"`, `"medium"`, `"low"`, `"insufficient"`| `Answer.confidence`         |
| `IndexType`      | `"hnsw"`, `"ivf"`                              | `BenchmarkResult.index_type`|
