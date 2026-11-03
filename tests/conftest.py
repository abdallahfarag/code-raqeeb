import logging
from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import pytest
from fastapi import FastAPI
from guardrails.classes.rc import RC
from guardrails.settings import settings as guardrails_settings

from code_raqeeb.api.dependencies import get_http_client
from code_raqeeb.config import Settings, get_settings
from code_raqeeb.main import create_app

QDRANT_TEST_URL = "http://qdrant.test:6333"

type QdrantHandler = Callable[[httpx.Request], httpx.Response]
type QdrantMockInstaller = Callable[[QdrantHandler], list[httpx.Request]]


def _rc_without_metrics(logger: logging.Logger | None = None) -> RC:
    return RC(enable_metrics=False)  # pyright: ignore[reportCallIssue]


@pytest.fixture(autouse=True, scope="session")
def no_guardrails_telemetry() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(RC, "load", _rc_without_metrics)
        patch.setattr(guardrails_settings, "rc", _rc_without_metrics())
        yield


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(_env_file=None, qdrant_url=QDRANT_TEST_URL)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def app(test_settings: Settings) -> Iterator[FastAPI]:
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: test_settings
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def install_qdrant_mock(app: FastAPI) -> AsyncIterator[QdrantMockInstaller]:
    """Override `get_http_client` with a client whose transport is driven by `handler`.

    Returns the list of requests the mock transport received.
    """
    clients: list[httpx.AsyncClient] = []

    def install(handler: QdrantHandler) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def recording_handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(recording_handler))
        clients.append(client)
        app.dependency_overrides[get_http_client] = lambda: client
        return requests

    yield install

    for client in clients:
        await client.aclose()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as api_client:
        yield api_client
