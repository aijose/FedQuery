# Agent Workflow Contract

## Agent State Schema

```python
class AgentState(TypedDict):
    query: str                              # Current user question
    retrieved_chunks: list[ChunkResult]     # Chunks retrieved so far
    confidence: str                         # "high" | "medium" | "low" | "insufficient"
    reformulation_attempts: int             # Number of query reformulations (max 2)
    reformulated_query: str | None          # Rephrased query for retry
    answer: str | None                      # Final synthesized answer
    citations: list[Citation]               # Citations for the answer
    needs_retrieval: bool                   # Whether retrieval is needed
    metadata_hints: dict | None             # {"date_start": "YYYY-MM-DD", "date_end": "YYYY-MM-DD"} or None
    top_k_hint: int | None                  # LLM-estimated result count (1-50) or None for default (10)
```

## Graph Nodes

| Node | Description | Input State | Output State |
|------|-------------|-------------|--------------|
| `assess_query` | Analyze query, extract date hints and top_k estimate | `query` | `needs_retrieval`, `metadata_hints`, `top_k_hint` |
| `search_corpus` | Two-pass retrieval (filtered + unfiltered when date hints present) | `query` or `reformulated_query`, `metadata_hints`, `top_k_hint` | `retrieved_chunks` |
| `evaluate_confidence` | Assess relevance scores of retrieved chunks | `retrieved_chunks` | `confidence` |
| `reformulate_query` | Rephrase query for better retrieval | `query`, `retrieved_chunks` | `reformulated_query`, `reformulation_attempts` |
| `synthesize_answer` | Generate answer grounded in retrieved chunks | `query`, `retrieved_chunks` | `answer`, `citations` |
| `validate_citations` | Verify each citation maps to actual chunk | `answer`, `citations`, `retrieved_chunks` | `citations` (validated) |
| `respond` | Return final answer or uncertainty message | `answer`, `citations`, `confidence` | terminal |

## Graph Edges (Conditional Routing)

```
┌─────────────┐
│ assess_query │
└──────┬──────┘
       │
       ├── needs_retrieval = true ──────────────────┐
       │                                            ▼
       │                                    ┌───────────────┐
       │                                    │ search_corpus  │
       │                                    └───────┬───────┘
       │                                            │
       │                                            ▼
       │                                  ┌─────────────────────┐
       │                                  │ evaluate_confidence  │
       │                                  └──────────┬──────────┘
       │                                             │
       │                          ┌──────────────────┼──────────────────┐
       │                          │                  │                  │
       │                   confidence ≥          confidence =      confidence =
       │                    "medium"               "low" &        "insufficient"
       │                          │             attempts < 2      OR attempts ≥ 2
       │                          ▼                  │                  │
       │                ┌──────────────────┐         ▼                  │
       │                │ synthesize_answer │  ┌──────────────┐         │
       │                └────────┬─────────┘  │ reformulate  │         │
       │                         │            │    _query     │         │
       │                         ▼            └──────┬───────┘         │
       │               ┌────────────────────┐        │                 │
       │               │ validate_citations │        │                 │
       │               └────────┬───────────┘        │                 │
       │                        │              ┌─────┘                 │
       │                        ▼              │                       │
       │                  ┌─────────┐          │                       │
       └─── (no retrieval)│ respond │◄─────────┘───────────────────────┘
                          └─────────┘
```

## Edge Conditions

| From | To | Condition |
|------|----|-----------|
| `assess_query` | `search_corpus` | `needs_retrieval = true` |
| `assess_query` | `respond` | `needs_retrieval = false` (context sufficient) |
| `search_corpus` | `evaluate_confidence` | always |
| `evaluate_confidence` | `synthesize_answer` | `confidence ≥ "medium"` |
| `evaluate_confidence` | `reformulate_query` | `confidence = "low"` AND `reformulation_attempts < 2` |
| `evaluate_confidence` | `respond` (uncertainty) | `confidence = "insufficient"` OR `reformulation_attempts ≥ 2` |
| `reformulate_query` | `search_corpus` | always (retry with new query) |
| `synthesize_answer` | `validate_citations` | always |
| `validate_citations` | `respond` | always |

## Confidence Thresholds

| Level | Relevance Score Range | Action |
|-------|----------------------|--------|
| `high` | avg score ≥ 0.55 | Synthesize answer |
| `medium` | avg score ≥ 0.40 | Synthesize answer |
| `low` | avg score ≥ 0.25 | Reformulate and retry |
| `insufficient` | avg score < 0.25 | Return uncertainty message |

> **Note**: Thresholds were recalibrated from the original spec (0.80/0.60/0.40) to match cosine similarity on the FOMC corpus. These may need further recalibration when switching embedding models (currently bge-small-en-v1.5).

## Response Formats

### Grounded Answer

```
{answer_text}

Sources:
  [1] {document_name}, {date}, §{section_header} (chunk {chunk_id})
  [2] ...
```

### Uncertainty Response

```
I was unable to find sufficient information in the FOMC document corpus
to answer this question confidently. The available documents may not
cover this topic.

Searched: {reformulated_queries}
Best matches (low relevance):
  [1] {document_name} (score: {relevance_score})
```
