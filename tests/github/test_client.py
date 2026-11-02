import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from code_raqeeb.github.errors import GitHubApiError, PrNotFoundError, RateLimitError
from tests.github.conftest import GitHubClientFactory
from tests.github.factories import (
    HEAD_SHA,
    REF,
    file_payload,
    make_settings,
    pr_payload,
    repo_payload,
    tree_payload,
)


@pytest.mark.parametrize("token", [None, "ghp_test"], ids=["anonymous", "with-token"])
async def test_sends_standard_headers(
    make_github_client: GitHubClientFactory, token: str | None
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=pr_payload(base_repo=repo_payload()))

    client = make_github_client(handler, make_settings(github_token=token))
    pull_request = await client.get_pull_request(REF)

    assert pull_request.number == 7
    [sent] = requests
    assert sent.url == "https://api.github.com/repos/octo/hello/pulls/7"
    assert sent.headers["Accept"] == "application/vnd.github+json"
    assert sent.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert sent.headers["User-Agent"] == "code-raqeeb"
    if token is None:
        assert "Authorization" not in sent.headers
    else:
        assert sent.headers["Authorization"] == f"Bearer {token}"


@pytest.mark.parametrize(
    ("page_sizes", "expected_requests"),
    [
        pytest.param([100, 5], 2, id="short-second-page"),
        pytest.param([100, 0], 2, id="exactly-one-full-page"),
        pytest.param([100] * 31, 30, id="stops-at-30-pages"),
    ],
)
async def test_list_files_pagination(
    make_github_client: GitHubClientFactory, page_sizes: list[int], expected_requests: int
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = int(request.url.params["page"])
        files = [file_payload(f"f{page}_{i}.cs") for i in range(page_sizes[page - 1])]
        return httpx.Response(200, json=files)

    client = make_github_client(handler, make_settings())
    files = await client.list_pull_request_files(REF)

    assert len(requests) == expected_requests
    assert len(files) == sum(page_sizes[:expected_requests])
    pages = [r.url.params["page"] for r in requests]
    assert pages == [str(n) for n in range(1, expected_requests + 1)]
    assert all(r.url.path == "/repos/octo/hello/pulls/7/files" for r in requests)
    assert all(r.url.params["per_page"] == "100" for r in requests)


async def test_get_file_content_uses_raw_accept_and_ref(
    make_github_client: GitHubClientFactory,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="class Foo {}")

    client = make_github_client(handler, make_settings())
    content = await client.get_file_content(REF, "src/Foo.cs", "abc123")

    assert content == "class Foo {}"
    [sent] = requests
    assert sent.url.path == "/repos/octo/hello/contents/src/Foo.cs"
    assert sent.url.params["ref"] == "abc123"
    assert sent.headers["Accept"] == "application/vnd.github.raw+json"


@pytest.mark.parametrize(
    ("status", "headers", "error_type", "status_code"),
    [
        pytest.param(404, {}, PrNotFoundError, None, id="404"),
        pytest.param(429, {}, RateLimitError, None, id="429"),
        pytest.param(403, {"x-ratelimit-remaining": "0"}, RateLimitError, None, id="403-exhausted"),
        pytest.param(403, {"x-ratelimit-remaining": "12"}, GitHubApiError, 403, id="403-forbidden"),
        pytest.param(500, {}, GitHubApiError, 500, id="500"),
    ],
)
async def test_error_status_mapping(
    make_github_client: GitHubClientFactory,
    status: int,
    headers: dict[str, str],
    error_type: type[Exception],
    status_code: int | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, json={"message": "nope"})

    client = make_github_client(handler, make_settings())

    with pytest.raises(error_type) as exc_info:
        await client.get_pull_request(REF)

    assert type(exc_info.value) is error_type
    if isinstance(exc_info.value, GitHubApiError):
        assert exc_info.value.status_code == status_code


async def test_rate_limit_reset_parsed_as_utc(make_github_client: GitHubClientFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"}
        return httpx.Response(403, headers=headers)

    client = make_github_client(handler, make_settings())

    with pytest.raises(RateLimitError) as exc_info:
        await client.get_pull_request(REF)

    reset_at = exc_info.value.reset_at
    assert reset_at == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert reset_at is not None
    assert reset_at.tzinfo is UTC


async def test_network_error_becomes_api_error(make_github_client: GitHubClientFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_github_client(handler, make_settings())

    with pytest.raises(GitHubApiError) as exc_info:
        await client.get_pull_request(REF)

    assert exc_info.value.status_code is None
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


async def test_get_files_content_respects_concurrency_limit(
    make_github_client: GitHubClientFactory,
) -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, text=f"content of {request.url.path}")

    client = make_github_client(handler, make_settings(github_concurrency=2))
    paths = [f"src/File{i}.cs" for i in range(8)]
    contents = await client.get_files_content(REF, paths, "abc123")

    assert peak == 2
    assert contents == {path: f"content of /repos/octo/hello/contents/{path}" for path in paths}


async def test_list_tree_requests_recursive_tree_and_returns_only_blobs(
    make_github_client: GitHubClientFactory,
) -> None:
    requests: list[httpx.Request] = []
    entries = [
        ("src", "tree"),
        ("src/Foo.cs", "blob"),
        ("vendor/submodule", "commit"),
        ("README.md", "blob"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=tree_payload(entries))

    client = make_github_client(handler, make_settings())
    paths = await client.list_tree(REF, HEAD_SHA)

    assert paths == ["src/Foo.cs", "README.md"]
    [sent] = requests
    assert sent.url.path == f"/repos/octo/hello/git/trees/{HEAD_SHA}"
    assert sent.url.params["recursive"] == "1"


async def test_list_tree_raises_when_truncated(make_github_client: GitHubClientFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=tree_payload([("src/Foo.cs", "blob")], truncated=True))

    client = make_github_client(handler, make_settings())

    with pytest.raises(GitHubApiError, match="octo/hello"):
        await client.list_tree(REF, HEAD_SHA)
