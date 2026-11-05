import asyncio
from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest
from fastmcp.exceptions import ToolError

from code_raqeeb.config import Settings
from code_raqeeb.guardrails.input import MAX_URL_LENGTH, InputRejectedError
from code_raqeeb.mcp_server import dependencies
from code_raqeeb.mcp_server.dependencies import checked_url, get_service
from code_raqeeb.review.service import ReviewService
from tests.github.factories import make_settings

PR_URL = "https://github.com/octo/hello/pull/7"
PULL_PREFIX = "https://github.com/octo/hello/pull/"


class FakeBuild:
    def __init__(self) -> None:
        self.service = cast(ReviewService, object())
        self.failures: list[Exception] = []
        self.calls: list[tuple[Settings, httpx.AsyncClient]] = []

    async def __call__(self, settings: Settings, http: httpx.AsyncClient) -> ReviewService:
        self.calls.append((settings, http))
        # Lets a concurrent caller run while this build is in progress
        await asyncio.sleep(0)
        if self.failures:
            raise self.failures.pop(0)
        return self.service


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    value = make_settings().model_copy(update={"http_timeout_seconds": 7.0})

    def fake_settings() -> Settings:
        return value

    monkeypatch.setattr(dependencies, "get_settings", fake_settings)
    return value


@pytest.fixture
async def build(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> AsyncIterator[FakeBuild]:
    fake = FakeBuild()
    monkeypatch.setattr(dependencies, "build_review_service", fake)
    monkeypatch.setattr(dependencies, "_service_lock", asyncio.Lock())
    monkeypatch.setattr(dependencies, "_service", None)
    yield fake
    for _, http in fake.calls:
        await http.aclose()


async def test_concurrent_calls_build_the_service_only_once(build: FakeBuild) -> None:
    first, second = await asyncio.gather(get_service(), get_service())

    assert len(build.calls) == 1
    assert first is build.service
    assert second is build.service


async def test_later_calls_reuse_the_built_service(build: FakeBuild) -> None:
    first = await get_service()
    second = await get_service()

    assert len(build.calls) == 1
    assert first is second


async def test_service_is_built_from_the_settings_with_their_http_timeout(
    build: FakeBuild, settings: Settings
) -> None:
    await get_service()

    used_settings, http = build.calls[0]
    assert used_settings is settings
    assert http.timeout == httpx.Timeout(7.0)


async def test_failed_build_is_not_cached_and_the_next_call_retries(build: FakeBuild) -> None:
    build.failures.append(RuntimeError("qdrant is down"))

    with pytest.raises(RuntimeError, match="qdrant is down"):
        await get_service()

    assert await get_service() is build.service
    assert len(build.calls) == 2


@pytest.mark.parametrize(
    "url",
    [
        pytest.param(PR_URL, id="plain"),
        pytest.param(f"{PR_URL}/files", id="files-tab"),
        pytest.param(f"{PR_URL}/", id="trailing-slash"),
        pytest.param(PULL_PREFIX + "1" * (MAX_URL_LENGTH - len(PULL_PREFIX)), id="at-the-limit"),
    ],
)
def test_checked_url_returns_a_valid_url(url: str) -> None:
    assert checked_url(url) == url


def test_checked_url_strips_surrounding_whitespace() -> None:
    assert checked_url(f"  {PR_URL}\n") == PR_URL


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("", id="empty"),
        pytest.param("not a url", id="not-a-url"),
        pytest.param("http://github.com/octo/hello/pull/7", id="http-scheme"),
        pytest.param("https://gitlab.com/octo/hello/pull/7", id="other-host"),
        pytest.param("https://github.com/octo/hello/issues/7", id="issue"),
        pytest.param(PULL_PREFIX + "1" * (MAX_URL_LENGTH + 1 - len(PULL_PREFIX)), id="too-long"),
    ],
)
def test_checked_url_turns_a_rejected_url_into_a_tool_error(url: str) -> None:
    with pytest.raises(ToolError) as caught:
        checked_url(url)

    assert isinstance(caught.value.__cause__, InputRejectedError)
    assert str(caught.value) == str(caught.value.__cause__)
