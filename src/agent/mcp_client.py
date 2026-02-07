"""MCP client integration for the agent workflow.

Provides a search function that queries the ChromaDB store via the
embedding provider, matching the ChunkResult schema expected by
the agent graph nodes.
"""

from src.agent.state import ChunkResult
from src.embedding.provider import EmbeddingProvider
from src.vectorstore.chroma_store import ChromaStore


def create_search_fn(
    store: ChromaStore,
    embedding_provider: EmbeddingProvider,
):
    """Create a search function that wraps ChromaDB queries.

    Returns a callable matching the signature expected by search_corpus:
        (query: str, top_k: int) -> list[ChunkResult]

    Note: In a full MCP deployment, this would spawn an MCP server subprocess
    and communicate via stdio. For the MVP, we call the store directly through
    the same interface the MCP server would use.
    """

    def search_fn(query: str, top_k: int = 5) -> list[ChunkResult]:
        query_embedding = embedding_provider.embed([query])[0]
        raw_results = store.query(query_embedding=query_embedding, top_k=top_k)

        results = []
        for r in raw_results:
            metadata = r.get("metadata", {})
            # ChromaDB cosine distance = 1 - cosine_similarity, range [0, 2]
            # Convert back: similarity = 1 - distance
            distance = r.get("distance", 1.0)
            relevance_score = max(0.0, min(1.0, 1.0 - distance))

            results.append(ChunkResult(
                chunk_id=r["id"],
                document_name=metadata.get("document_title", ""),
                document_date=metadata.get("document_date", ""),
                document_id=metadata.get("document_id", ""),
                section_header=metadata.get("section_header", ""),
                chunk_text=r.get("text", ""),
                relevance_score=relevance_score,
            ))

        return results

    return search_fn
