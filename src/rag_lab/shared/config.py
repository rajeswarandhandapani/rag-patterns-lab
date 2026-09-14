"""Read configuration once, without hard-coding credentials in the lab code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic reads environment/.env values and converts them to annotated types.

    Environment variables take precedence over .env and field defaults.
    Paths are relative to the current working directory: run from the repo root.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""
    azure_openai_embedding_model: str = "text-embedding-3-small"

    # An alias connects the Python attribute (data_dir) to its environment name.
    data_dir: Path = Field(default=Path("data/documents"), alias="RAG_DATA_DIR")
    eval_file: Path = Field(default=Path("data/evaluation/questions.json"), alias="RAG_EVAL_FILE")
    index_dir: Path = Field(default=Path(".rag_indexes"), alias="RAG_INDEX_DIR")
    # These sizes are characters, not tokens. top_k is the maximum chunk count.
    chunk_size: int = Field(default=700, alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="RAG_CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="RAG_TOP_K")

    def require_azure(self) -> None:
        """Fail early with missing variable names instead of making an API request."""
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
                "AZURE_OPENAI_CHAT_DEPLOYMENT": self.azure_openai_chat_deployment,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai_embedding_deployment,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Azure configuration: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    """Reuse one settings object per process; restart after changing .env.

    The @ decorator wraps this function with caching. Tests can call
    get_settings.cache_clear() when they deliberately change the environment.
    """
    return Settings()
