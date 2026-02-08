"""Application configuration management."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """FedQuery application settings loaded from environment variables."""

    # Required
    anthropic_api_key: str = ""

    # Embedding
    fedquery_embedding_model: str = "all-MiniLM-L6-v2"

    # LLM
    fedquery_llm_provider: str = "anthropic"
    fedquery_llm_model: str = "claude-sonnet-4-5-20250929"

    # Storage
    fedquery_chroma_path: str = "./data/chroma"
    fedquery_text_path: str = "./data/texts"
    fedquery_html_path: str = "./data/html"

    # Ingestion
    fedquery_chunk_size: int = 512
    fedquery_chunk_overlap: int = 50

    # Reranking
    fedquery_reranker_enabled: bool = False
    fedquery_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # MCP
    fedquery_use_mcp: bool = True

    @property
    def chroma_path(self) -> Path:
        return Path(self.fedquery_chroma_path)

    @property
    def text_path(self) -> Path:
        return Path(self.fedquery_text_path)

    @property
    def html_path(self) -> Path:
        return Path(self.fedquery_html_path)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
