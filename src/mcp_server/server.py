"""MCP server exposing FOMC search and document retrieval tools."""

import asyncio
import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from config.settings import get_settings
from src.embedding.sentence_transformer import SentenceTransformerEmbeddingProvider
from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

server = Server("fomc-rag")
_store: ChromaStore | None = None
_embedding_provider: SentenceTransformerEmbeddingProvider | None = None


def _get_store() -> ChromaStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = ChromaStore(path=str(settings.chroma_path))
    return _store


def _get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = SentenceTransformerEmbeddingProvider()
    return _embedding_provider


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_fomc",
            description="Search FOMC documents by semantic similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "top_k": {"type": "integer", "default": 5, "description": "Number of results (max 20)"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_document",
            description="Retrieve full document content and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "Unique document identifier"},
                },
                "required": ["document_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_fomc":
        return await _handle_search_fomc(arguments)
    elif name == "get_document":
        return await _handle_get_document(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_search_fomc(arguments: dict) -> list[TextContent]:
    query = arguments.get("query", "")
    top_k = min(arguments.get("top_k", 5), 20)

    if not query or len(query) > 1000:
        return [TextContent(type="text", text=json.dumps({"error": "invalid_query"}))]

    store = _get_store()
    if store.count == 0:
        return [TextContent(type="text", text=json.dumps({"error": "empty_corpus"}))]

    provider = _get_embedding_provider()
    query_embedding = provider.embed([query])[0]
    raw_results = store.query(query_embedding=query_embedding, top_k=top_k)

    results = []
    for r in raw_results:
        metadata = r.get("metadata", {})
        distance = r.get("distance", 1.0)
        relevance_score = max(0.0, min(1.0, 1.0 - distance / 2.0))

        results.append({
            "chunk_id": r["id"],
            "document_name": metadata.get("document_title", ""),
            "document_date": metadata.get("document_date", ""),
            "document_id": metadata.get("document_id", ""),
            "section_header": metadata.get("section_header", ""),
            "chunk_text": r.get("text", ""),
            "relevance_score": round(relevance_score, 4),
        })

    return [TextContent(type="text", text=json.dumps(results))]


async def _handle_get_document(arguments: dict) -> list[TextContent]:
    doc_id = arguments.get("document_id", "")
    if not doc_id:
        return [TextContent(type="text", text=json.dumps({"error": "not_found"}))]

    store = _get_store()
    chunks = store.get_document_chunks(doc_id)

    if not chunks:
        return [TextContent(type="text", text=json.dumps({"error": "not_found"}))]

    metadata = chunks[0].get("metadata", {})
    full_text = "\n\n".join(c.get("text", "") for c in chunks)

    result = {
        "id": doc_id,
        "title": metadata.get("document_title", ""),
        "date": metadata.get("document_date", ""),
        "document_type": metadata.get("document_type", ""),
        "source_url": metadata.get("source_url", ""),
        "full_text": full_text,
        "chunk_count": len(chunks),
    }

    return [TextContent(type="text", text=json.dumps(result))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
