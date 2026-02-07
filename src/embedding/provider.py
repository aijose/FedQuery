"""Abstract embedding provider interface."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for text embedding generation.

    Implementations wrap specific embedding models (e.g., sentence-transformers).
    Swap models by changing the provider implementation in configuration.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors. Each vector is a list of floats
            with dimension matching the model configuration.

        Raises:
            ValueError: If texts is empty.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension (e.g., 384)."""
        ...
