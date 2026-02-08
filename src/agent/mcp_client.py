"""MCP client integration for the agent workflow.

Provides two retrieval backends:
- MCPSearchClient: spawns MCP server subprocess, communicates via stdio protocol
- create_direct_search_fn: in-process ChromaStore calls (fast dev/debug path)

Both produce the same (query, top_k) -> list[ChunkResult] interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from typing import TYPE_CHECKING

from src.agent.state import ChunkResult

if TYPE_CHECKING:
    from src.embedding.provider import EmbeddingProvider
    from src.retrieval.reranker import CrossEncoderReranker
    from src.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class MCPSearchClient:
    """Sync wrapper around the async MCP stdio client.

    Spawns the MCP server as a subprocess and communicates via stdio.
    Uses a background daemon thread with its own asyncio event loop
    to bridge async MCP calls into sync LangGraph nodes.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session = None
        self._exit_stack = None

    def connect(self):
        """Start background event loop, spawn MCP server, establish session."""
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.session import ClientSession

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()

        async def _connect():
            from contextlib import AsyncExitStack

            self._exit_stack = AsyncExitStack()
            await self._exit_stack.__aenter__()

            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "src.mcp_server"],
            )

            read, write = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            session = ClientSession(read, write)
            await self._exit_stack.enter_async_context(session)
            await session.initialize()

            self._session = session
            logger.info("MCP client connected to server subprocess")

        future = asyncio.run_coroutine_threadsafe(_connect(), self._loop)
        future.result(timeout=30)

    def search(self, query: str, top_k: int = 5) -> list[ChunkResult]:
        """Call search_fomc tool via MCP protocol."""
        if not self._session:
            raise RuntimeError("MCPSearchClient not connected")

        async def _search():
            result = await self._session.call_tool(
                "search_fomc",
                arguments={"query": query, "top_k": top_k},
            )
            text = result.content[0].text
            return json.loads(text)

        future = asyncio.run_coroutine_threadsafe(_search(), self._loop)
        raw_results = future.result(timeout=30)

        if isinstance(raw_results, dict) and "error" in raw_results:
            logger.warning("MCP search error: %s", raw_results["error"])
            return []

        return [
            ChunkResult(
                chunk_id=r["chunk_id"],
                document_name=r["document_name"],
                document_date=r["document_date"],
                document_id=r["document_id"],
                section_header=r["section_header"],
                chunk_text=r["chunk_text"],
                relevance_score=r["relevance_score"],
            )
            for r in raw_results
        ]

    def get_document(self, document_id: str) -> dict:
        """Call get_document tool via MCP protocol."""
        if not self._session:
            raise RuntimeError("MCPSearchClient not connected")

        async def _get_doc():
            result = await self._session.call_tool(
                "get_document",
                arguments={"document_id": document_id},
            )
            text = result.content[0].text
            return json.loads(text)

        future = asyncio.run_coroutine_threadsafe(_get_doc(), self._loop)
        return future.result(timeout=30)

    def close(self):
        """Clean up session, transport, stop event loop, join thread."""
        if self._loop and self._exit_stack:
            async def _cleanup():
                await self._exit_stack.aclose()

            try:
                future = asyncio.run_coroutine_threadsafe(
                    _cleanup(), self._loop
                )
                future.result(timeout=10)
            except Exception:
                logger.debug("MCP cleanup error (expected on shutdown)")

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5)

        self._session = None
        self._exit_stack = None
        self._loop = None
        self._thread = None
        logger.info("MCP client disconnected")


def create_mcp_search_fn(
    mcp_client: MCPSearchClient,
    reranker: CrossEncoderReranker | None = None,
):
    """Create a search function that routes through the MCP server.

    Reranking stays client-side: over-fetches 3x via MCP, reranks locally.
    """

    def search_fn(query: str, top_k: int = 5) -> list[ChunkResult]:
        fetch_k = top_k * 3 if reranker else top_k
        results = mcp_client.search(query, top_k=fetch_k)

        if reranker and results:
            reranked = reranker.rerank(
                query,
                [dict(r) for r in results],
                top_k=top_k,
            )
            results = [
                ChunkResult(
                    chunk_id=r["chunk_id"],
                    document_name=r["document_name"],
                    document_date=r["document_date"],
                    document_id=r["document_id"],
                    section_header=r["section_header"],
                    chunk_text=r["chunk_text"],
                    relevance_score=r["relevance_score"],
                )
                for r in reranked
            ]

        return results

    return search_fn


def create_direct_search_fn(
    store: ChromaStore,
    embedding_provider: EmbeddingProvider,
    reranker: CrossEncoderReranker | None = None,
):
    """Create a search function that calls ChromaStore directly (no MCP).

    This is the fast dev/debug path — same interface as MCP mode.
    """

    def search_fn(query: str, top_k: int = 5) -> list[ChunkResult]:
        fetch_k = top_k * 3 if reranker else top_k
        query_embedding = embedding_provider.embed([query])[0]
        raw_results = store.query(query_embedding=query_embedding, top_k=fetch_k)

        results = []
        for r in raw_results:
            metadata = r.get("metadata", {})
            # ChromaDB cosine distance range [0, 2]
            # similarity = 1.0 - distance / 2.0
            distance = r.get("distance", 1.0)
            relevance_score = max(0.0, min(1.0, 1.0 - distance / 2.0))

            results.append(ChunkResult(
                chunk_id=r["id"],
                document_name=metadata.get("document_title", ""),
                document_date=metadata.get("document_date", ""),
                document_id=metadata.get("document_id", ""),
                section_header=metadata.get("section_header", ""),
                chunk_text=r.get("text", ""),
                relevance_score=relevance_score,
            ))

        if reranker and results:
            reranked = reranker.rerank(
                query,
                [dict(r) for r in results],
                top_k=top_k,
            )
            results = [
                ChunkResult(
                    chunk_id=r["chunk_id"],
                    document_name=r["document_name"],
                    document_date=r["document_date"],
                    document_id=r["document_id"],
                    section_header=r["section_header"],
                    chunk_text=r["chunk_text"],
                    relevance_score=r["relevance_score"],
                )
                for r in reranked
            ]

        return results

    return search_fn


# Backward-compatible alias for existing code that imports create_search_fn
create_search_fn = create_direct_search_fn
