"""Sentence Transformer embedding provider implementation."""

from sentence_transformers import SentenceTransformer

from src.embedding.provider import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider wrapping sentence-transformers models.

    Default model: all-MiniLM-L6-v2 (384 dimensions, ~80MB).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
