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


@lru_cache
def get_settings() -> Settings:
    return Settings()
