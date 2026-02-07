"""Sentence Transformer embedding provider implementation."""

import os

from sentence_transformers import SentenceTransformer

from src.embedding.provider import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider wrapping sentence-transformers models.

    Default model: all-MiniLM-L6-v2 (384 dimensions, ~80MB).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Suppress safetensors LOAD REPORT (written to fd 1 by C code)
        # and tqdm progress bars during model loading.
        old_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            try:
                self._model = SentenceTransformer(model_name, local_files_only=True)
            except OSError:
                self._model = SentenceTransformer(model_name)
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            if old_verbosity is None:
                os.environ.pop("TRANSFORMERS_VERBOSITY", None)
            else:
                os.environ["TRANSFORMERS_VERBOSITY"] = old_verbosity
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
