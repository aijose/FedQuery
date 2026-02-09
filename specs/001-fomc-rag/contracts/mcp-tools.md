# MCP Tool Contracts

## Tool 1: search_fomc

**Description**: Search the FOMC document corpus using semantic similarity.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `query` | string | Yes | — | Natural language search query |
| `top_k` | integer | No | 5 | Number of results to return (max 50) |
| `where` | object | No | — | ChromaDB metadata filter (e.g., `{"$and": [{"document_date": {"$gte": "2024-01-01"}}, {"document_date": {"$lte": "2024-12-31"}}]}`) |

### Response

Array of objects:

```json
[
  {
    "chunk_id": "uuid-string",
    "document_name": "FOMC Minutes, January 30-31, 2024",
    "document_date": "2024-01-31",
    "document_id": "uuid-string",
    "section_header": "Participants' Views on Current Conditions",
    "chunk_text": "The matched text passage...",
    "relevance_score": 0.87
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Unique identifier of the matched chunk |
| `document_name` | string | Title of the source FOMC document |
| `document_date` | string (ISO date) | Date of the FOMC document |
| `document_id` | string | ID of the parent document (for use with `get_document`) |
| `section_header` | string (nullable) | Section of the document this chunk belongs to |
| `chunk_text` | string | The matched text passage |
| `relevance_score` | float (0-1) | Cosine similarity score |

### Errors

| Code | Description |
|------|-------------|
| `empty_corpus` | Vector store has no documents ingested |
| `invalid_query` | Query is empty or exceeds 1000 characters |

---

## Tool 2: get_document

**Description**: Retrieve the full text and metadata of a specific FOMC document.

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `document_id` | string | Yes | — | Unique identifier of the document |

### Response

```json
{
  "id": "uuid-string",
  "title": "FOMC Minutes, January 30-31, 2024",
  "date": "2024-01-31",
  "document_type": "minutes",
  "source_url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm",
  "full_text": "Complete document text...",
  "chunk_count": 47
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Document ID |
| `title` | string | Document title |
| `date` | string (ISO date) | Document date |
| `document_type` | string | `"statement"` or `"minutes"` |
| `source_url` | string | Original URL on federalreserve.gov |
| `full_text` | string | Complete document text |
| `chunk_count` | integer | Number of chunks generated from this document |

### Errors

| Code | Description |
|------|-------------|
| `not_found` | Document ID does not exist |
