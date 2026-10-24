import httpx
import pytest

from tests.conftest import QdrantHandler, QdrantMockInstaller


def respond_with(status_code: int) -> QdrantHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    return handler


def raise_error(error_type: type[httpx.TransportError]) -> QdrantHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("simulated failure", request=request)

    return handler


async def test_live_ok_without_calling_qdrant(
    client: httpx.AsyncClient, install_qdrant_mock: QdrantMockInstaller
) -> None:
    requests = install_qdrant_mock(respond_with(200))

    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert requests == []


async def test_ready_when_qdrant_up(
    client: httpx.AsyncClient, install_qdrant_mock: QdrantMockInstaller
) -> None:
    install_qdrant_mock(respond_with(200))

    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "dependencies": {"qdrant": "up"}}


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(respond_with(500), id="http-500"),
        pytest.param(raise_error(httpx.ConnectError), id="connect-error"),
        pytest.param(raise_error(httpx.ReadTimeout), id="read-timeout"),
    ],
)
async def test_not_ready_when_qdrant_down(
    client: httpx.AsyncClient,
    install_qdrant_mock: QdrantMockInstaller,
    handler: QdrantHandler,
) -> None:
    install_qdrant_mock(handler)

    response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "dependencies": {"qdrant": "down"}}


async def test_ready_sends_get_to_qdrant_readyz(
    client: httpx.AsyncClient, install_qdrant_mock: QdrantMockInstaller
) -> None:
    requests = install_qdrant_mock(respond_with(200))

    await client.get("/health/ready")

    assert len(requests) == 1
    sent = requests[0]
    assert sent.method == "GET"
    assert sent.url.host == "qdrant.test"
    assert sent.url.port == 6333
    assert sent.url.path == "/readyz"
