import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from code_raqeeb.config import Settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import (
    GitHubApiError,
    GitHubError,
    PrivateRepoError,
    PrNotFoundError,
    PrTooLargeError,
    RateLimitError,
)
from code_raqeeb.github.service import load_pull_request
from code_raqeeb.guardrails.input import MAX_URL_LENGTH
from code_raqeeb.rag.indexer import RepoTooLargeError
from code_raqeeb.review.models import ReviewComment
from code_raqeeb.review.service import ReviewEvent, ReviewResult

PR_URL = "https://github.com/octo/hello/pull/7"
PULL_PREFIX = "https://github.com/octo/hello/pull/"
ENDPOINTS = ["/reviews", "/reviews/sync"]

EVENTS = [
    ReviewEvent(name="accepted", data={"pr_url": PR_URL}),
    ReviewEvent(name="pr_loaded", data={"title": "Pay\nendpoint — رقيب", "files": ["a.cs"]}),
    ReviewEvent(name="indexed", data={"repo": "octo/hello", "skipped": False, "files": 2}),
    ReviewEvent(name="file_reviewed", data={"file": "a.cs", "comments": 1, "errors": []}),
    ReviewEvent(name="verified", data={"kept": 1, "dropped": 0}),
    ReviewEvent(name="done", data={"pr_url": PR_URL, "summary": "1 comment(s)"}),
]

RESULT = ReviewResult(
    pr_url=PR_URL,
    head_sha="head111",
    summary="1 comment(s): 1 high, 0 medium, 0 low",
    comments=[
        ReviewComment(
            path="a.cs",
            line=3,
            severity="high",
            category="bug",
            rule_id=None,
            message="Charge is not awaited",
            suggestion="await the call",
            permalink="https://github.com/octo/hello/blob/head111/a.cs#L3",
        )
    ],
    file_errors=["b.cs: RuntimeError: model unavailable"],
    dropped_comments=2,
)


class FakeReviewService:
    def __init__(self) -> None:
        self.events: list[ReviewEvent] = EVENTS
        self.result = RESULT
        self.error: Exception | None = None
        self.calls: list[str] = []

    async def stream(self, pr_url: str) -> AsyncIterator[ReviewEvent]:
        self.calls.append(pr_url)
        for event in self.events:
            yield event

    async def run(self, pr_url: str) -> ReviewResult:
        self.calls.append(pr_url)
        if self.error is not None:
            raise self.error
        return self.result


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        assert set(fields) == {"event", "data"}
        parsed.append((fields["event"], json.loads(fields["data"])))
    return parsed


@pytest.fixture
def service(app: FastAPI) -> FakeReviewService:
    fake = FakeReviewService()
    app.state.review_service = fake
    return fake


async def test_stream_returns_an_event_stream(
    client: httpx.AsyncClient, service: FakeReviewService
) -> None:
    response = await client.post("/reviews", json={"pr_url": PR_URL})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


async def test_stream_sends_each_event_in_order_with_its_name_and_json_data(
    client: httpx.AsyncClient, service: FakeReviewService
) -> None:
    response = await client.post("/reviews", json={"pr_url": PR_URL})

    assert parse_sse(response.text) == [(event.name, event.data) for event in EVENTS]
    assert service.calls == [PR_URL]


async def test_stream_delivers_an_error_event_with_a_200_status(
    client: httpx.AsyncClient, service: FakeReviewService
) -> None:
    error = {"type": "PrNotFoundError", "message": "Pull request not found"}
    service.events = [ReviewEvent(name="error", data=error)]

    response = await client.post("/reviews", json={"pr_url": PR_URL})

    assert response.status_code == 200
    assert parse_sse(response.text) == [("error", error)]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "url",
    [
        pytest.param("", id="empty"),
        pytest.param("not a url", id="not-a-url"),
        pytest.param("http://github.com/octo/hello/pull/7", id="http-scheme"),
        pytest.param("https://gitlab.com/octo/hello/pull/7", id="other-host"),
        pytest.param("https://github.com/octo/hello/issues/7", id="issue"),
        pytest.param("https://github.com/octo/hello/pull/0", id="pr-zero"),
        pytest.param(PR_URL + "?next=https://evil.test", id="query-string"),
        pytest.param(PULL_PREFIX + "1" * (MAX_URL_LENGTH + 1 - len(PULL_PREFIX)), id="too-long"),
    ],
)
async def test_invalid_url_returns_422_without_calling_the_service(
    client: httpx.AsyncClient, service: FakeReviewService, endpoint: str, url: str
) -> None:
    response = await client.post(endpoint, json={"pr_url": url})

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="missing-field"),
        pytest.param({"pr_url": 7}, id="not-a-string"),
    ],
)
async def test_malformed_body_returns_422_without_calling_the_service(
    client: httpx.AsyncClient, service: FakeReviewService, endpoint: str, body: dict[str, object]
) -> None:
    response = await client.post(endpoint, json=body)

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_url_at_the_length_limit_is_accepted(
    client: httpx.AsyncClient, service: FakeReviewService, endpoint: str
) -> None:
    url = PULL_PREFIX + "1" * (MAX_URL_LENGTH - len(PULL_PREFIX))

    response = await client.post(endpoint, json={"pr_url": url})

    assert response.status_code == 200
    assert service.calls == [url]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_url_reaches_the_service_without_surrounding_whitespace(
    client: httpx.AsyncClient, service: FakeReviewService, endpoint: str
) -> None:
    response = await client.post(endpoint, json={"pr_url": f"  {PR_URL}\n"})

    assert response.status_code == 200
    assert service.calls == [PR_URL]


async def test_sync_returns_the_review_result_json(
    client: httpx.AsyncClient, service: FakeReviewService
) -> None:
    response = await client.post("/reviews/sync", json={"pr_url": PR_URL})

    assert response.status_code == 200
    assert response.json() == RESULT.model_dump(mode="json")
    assert service.calls == [PR_URL]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        pytest.param(PrNotFoundError("Pull request not found"), 404, id="not-found"),
        pytest.param(PrivateRepoError("Repository is private"), 403, id="private"),
        pytest.param(PrTooLargeError("Too many changed files"), 413, id="pr-too-large"),
        pytest.param(RepoTooLargeError("Too many C# files"), 413, id="repo-too-large"),
        pytest.param(RateLimitError(None), 429, id="rate-limit"),
        pytest.param(GitHubApiError("GitHub returned 500", 500), 502, id="api-error"),
        pytest.param(GitHubError("Something else"), 502, id="base-error"),
    ],
)
async def test_sync_maps_errors_to_status_codes_with_the_message_as_detail(
    client: httpx.AsyncClient, service: FakeReviewService, error: Exception, status_code: int
) -> None:
    service.error = error

    response = await client.post("/reviews/sync", json={"pr_url": PR_URL})

    assert response.status_code == status_code
    assert response.json() == {"detail": str(error)}


async def test_sync_returns_403_for_the_private_repo_error_from_the_loader(
    client: httpx.AsyncClient, service: FakeReviewService
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        repo = {"full_name": "octo/hello", "private": True}
        pull = {
            "number": 7,
            "title": "t",
            "state": "open",
            "html_url": PR_URL,
            "base": {"ref": "main", "sha": "base000", "repo": repo},
            "head": {"ref": "feature", "sha": "head111", "repo": None},
            "additions": 1,
            "deletions": 0,
            "changed_files": 1,
        }
        return httpx.Response(200, json=pull)

    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(PrivateRepoError) as caught:
            await load_pull_request(GitHubClient(http, settings), PR_URL, settings)
    service.error = caught.value

    response = await client.post("/reviews/sync", json={"pr_url": PR_URL})

    assert response.status_code == 403
    assert response.json() == {"detail": str(caught.value)}
    assert "ALLOW_PRIVATE_REPOS" in response.json()["detail"]
