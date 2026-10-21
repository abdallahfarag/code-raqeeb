from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown variables in .env instead of failing
    )

    app_name: str = "Code Raqeeb"
    environment: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    qdrant_url: HttpUrl = HttpUrl("http://localhost:6333")
    http_timeout_seconds: float = Field(default=5.0, gt=0, le=60)

    gemini_api_key: SecretStr | None = None
    github_token: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
