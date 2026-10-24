from pathlib import Path

import pytest
from pydantic import HttpUrl, SecretStr, ValidationError

from code_raqeeb.config import Settings, get_settings

SETTINGS_ENV_VARS = (
    "APP_NAME",
    "ENVIRONMENT",
    "LOG_LEVEL",
    "QDRANT_URL",
    "HTTP_TIMEOUT_SECONDS",
    "GITHUB_API_URL",
    "GITHUB_CONCURRENCY",
    "MAX_REVIEW_FILES",
    "MAX_REVIEW_CHANGED_LINES",
    "GEMINI_API_KEY",
    "GITHUB_TOKEN",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make_settings() -> Settings:
    return Settings(_env_file=None)  # pyright: ignore[reportCallIssue]


def test_defaults_without_env() -> None:
    settings = make_settings()

    assert settings.app_name == "Code Raqeeb"
    assert settings.environment == "local"
    assert settings.log_level == "INFO"
    assert settings.qdrant_url == HttpUrl("http://localhost:6333")
    assert settings.http_timeout_seconds == 5.0
    assert settings.github_api_url == HttpUrl("https://api.github.com")
    assert settings.github_concurrency == 5
    assert settings.max_review_files == 20
    assert settings.max_review_changed_lines == 800
    assert settings.gemini_api_key is None
    assert settings.github_token is None


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:7000")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", "12.5")

    settings = make_settings()

    assert settings.qdrant_url == HttpUrl("http://qdrant.example:7000")
    assert settings.environment == "production"
    assert settings.log_level == "DEBUG"
    assert settings.http_timeout_seconds == 12.5


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "staging")

    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize("timeout", ["0", "61"])
def test_out_of_range_http_timeout_rejected(monkeypatch: pytest.MonkeyPatch, timeout: str) -> None:
    monkeypatch.setenv("HTTP_TIMEOUT_SECONDS", timeout)

    with pytest.raises(ValidationError):
        make_settings()


def test_secret_hidden_from_repr_and_str(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "super-secret-gemini-key"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    settings = make_settings()

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert isinstance(settings.gemini_api_key, SecretStr)
    assert settings.gemini_api_key.get_secret_value() == secret


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Run from an empty directory so the developer's .env is not read.
    monkeypatch.chdir(tmp_path)

    assert get_settings() is get_settings()
