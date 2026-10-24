from collections.abc import AsyncIterator, Callable, Coroutine

import httpx
import pytest

from code_raqeeb.config import Settings
from code_raqeeb.github.client import GitHubClient

type SyncHandler = Callable[[httpx.Request], httpx.Response]
type AsyncHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
type Handler = SyncHandler | AsyncHandler
type GitHubClientFactory = Callable[[Handler, Settings], GitHubClient]


@pytest.fixture
async def make_github_client() -> AsyncIterator[GitHubClientFactory]:
    """Build a GitHubClient whose HTTP calls are answered by `handler`."""
    http_clients: list[httpx.AsyncClient] = []

    def factory(handler: Handler, settings: Settings) -> GitHubClient:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        http_clients.append(http)
        return GitHubClient(http, settings)

    yield factory

    for http in http_clients:
        await http.aclose()
