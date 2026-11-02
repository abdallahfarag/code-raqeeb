from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Code Raqeeb"
    environment: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    qdrant_url: HttpUrl = HttpUrl("http://localhost:6333")
    http_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    github_api_url: HttpUrl = HttpUrl("https://api.github.com")
    github_concurrency: int = Field(default=5, ge=1, le=20)

    max_review_files: int = Field(default=20, ge=1)
    max_review_changed_lines: int = Field(default=800, ge=1)

    gemini_api_key: SecretStr | None = None
    github_token: SecretStr | None = None

    qdrant_code_collection: str = "code_chunks"
    qdrant_rules_collection: str = "review_rules"
    dense_embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    sparse_embedding_model: str = "Qdrant/bm25"
    max_index_files: int = Field(default=500, ge=1)
    max_chunk_lines: int = Field(default=150, ge=10)

    llm_model: str = "gemini-3.5-flash-lite"
    llm_requests_per_minute: int = Field(default=8, ge=1)
    llm_max_retries: int = Field(default=3, ge=0)
    review_concurrency: int = Field(default=3, ge=1)
    max_tool_rounds: int = Field(default=2, ge=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
