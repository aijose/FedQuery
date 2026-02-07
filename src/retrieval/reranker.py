"""Cross-encoder reranker for two-stage retrieval.

Uses a cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2) to rescore
query-chunk pairs after initial bi-encoder retrieval. The cross-encoder
jointly encodes query+chunk for more accurate relevance scoring.
"""

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        logger.info("Loading cross-encoder: %s", model_name)
        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank chunks by cross-encoder relevance score.

        Args:
            query: The search query.
            chunks: List of chunk dicts, each must have a 'chunk_text' or 'text' key.
            top_k: If set, return only the top-k reranked results.
                   If None, return all chunks reranked.

        Returns:
            Sorted list of chunk dicts (highest relevance first), each with
            an added 'rerank_score' key.
        """
        if not chunks:
            return []

        # Build query-chunk pairs for the cross-encoder
        pairs = []
        for chunk in chunks:
            text = chunk.get("chunk_text", "") or chunk.get("text", "")
            pairs.append((query, text))

        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            enriched = dict(chunk)
            enriched["rerank_score"] = float(score)
            scored_chunks.append(enriched)

        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)

        if top_k is not None:
            return scored_chunks[:top_k]
        return scored_chunks
