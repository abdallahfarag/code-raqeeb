import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import cast, override

import pytest
from langchain_core.tools import BaseTool
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import AsyncQdrantClient

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.errors import PrivateRepoError, PrNotFoundError, RateLimitError
from code_raqeeb.github.models import PrRef, PullRequest
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.rag.indexer import RepoTooLargeError
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.graph import build_review_graph
from code_raqeeb.review.models import DraftComment, FileReview
from code_raqeeb.review.reviewer import FileReviewer
from code_raqeeb.review.service import ReviewEvent, ReviewResult, ReviewService
from tests.github.factories import make_settings
from tests.rag.fakes import FakeEmbedder
from tests.review.conftest import make_context
from tests.review.fakes import (
    BASE_SHA,
    HEAD_SHA,
    PR_URL,
    REPO,
    FakeGitHub,
    ScriptedChatModel,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

CONTROLLER = "src/Orders.Api/Controllers/OrdersController.cs"
REQUEST = "src/Orders.Api/Models/PayOrderRequest.cs"
SERVICE = "src/Orders.Api/Services/OrderService.cs"
REVIEWABLE = [CONTROLLER, REQUEST, SERVICE]

LINE_IN_TWO_FILES = 27

SECRET_DETAIL = "connection string with hunter2"

type ServiceFactory = Callable[[FakeGitHub, FileReviewer], Awaitable[ReviewService]]


class BrokenGitHub(FakeGitHub):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    @override
    async def get_pull_request(self, ref: PrRef) -> PullRequest:
        raise self.error


class FailsForPath(FileReviewer):
    def __init__(self, model: ScriptedChatModel, path: str) -> None:
        super().__init__(model, max_tool_rounds=0)
        self.path = path

    @override
    async def review(
        self, bundle: PullRequestBundle, file: ChangedFile, tools: Sequence[BaseTool]
    ) -> list[DraftComment]:
        if file.path == self.path:
            raise RuntimeError("model unavailable")
        return await super().review(bundle, file, tools)


class FakeQdrant:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeLangfuse:
    def __init__(self) -> None:
        self.shut_down = False

    def shutdown(self) -> None:
        self.shut_down = True


def reviewer_with_comment() -> FileReviewer:
    draft = DraftComment(
        line=LINE_IN_TWO_FILES, severity="high", category="bug", message="Charge is not awaited"
    )
    model = ScriptedChatModel(replies=[], review=FileReview(comments=[draft]))
    return FileReviewer(model, max_tool_rounds=0)


def quiet_reviewer() -> FileReviewer:
    return FileReviewer(ScriptedChatModel(replies=[]), max_tool_rounds=0)


async def collect(service: ReviewService, pr_url: str = PR_URL) -> list[ReviewEvent]:
    return [event async for event in service.stream(pr_url)]


def names(events: list[ReviewEvent]) -> list[str]:
    return [event.name for event in events]


def only(events: list[ReviewEvent], name: str) -> ReviewEvent:
    matches = [event for event in events if event.name == name]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
async def qdrant_store(qdrant: AsyncQdrantClient, embedder: FakeEmbedder) -> QdrantStore:
    store = QdrantStore(qdrant, "test_code", "test_rules")
    await store.ensure_collections(embedder.dense_size)
    return store


@pytest.fixture
def make_service(
    qdrant: AsyncQdrantClient, qdrant_store: QdrantStore, embedder: FakeEmbedder
) -> ServiceFactory:
    async def make(github: FakeGitHub, reviewer: FileReviewer) -> ReviewService:
        return ReviewService(
            graph=build_review_graph(InMemorySaver()),
            context=make_context(github, qdrant_store, embedder, reviewer),
            qdrant=qdrant,
            langfuse=None,
        )

    return make


async def test_stream_yields_every_stage_in_order(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, reviewer_with_comment())

    events = await collect(service)

    assert names(events) == [
        "accepted",
        "pr_loaded",
        "indexed",
        "file_reviewed",
        "file_reviewed",
        "file_reviewed",
        "verified",
        "done",
    ]


async def test_accepted_event_carries_the_url(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, quiet_reviewer())

    events = await collect(service)

    assert only(events, "accepted").data == {"pr_url": PR_URL}


async def test_pr_loaded_event_describes_the_pull_request(
    make_service: ServiceFactory, github: FakeGitHub, bundle: PullRequestBundle
) -> None:
    service = await make_service(github, quiet_reviewer())

    data = only(await collect(service), "pr_loaded").data

    assert data["title"] == github.pull_request.title
    assert data["head_sha"] == HEAD_SHA
    assert sorted(data["files"]) == REVIEWABLE
    assert data["skipped"] == len(bundle.skipped)


async def test_indexed_event_carries_the_index_report(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, quiet_reviewer())

    data = only(await collect(service), "indexed").data

    assert (data["repo"], data["sha"], data["skipped"], data["files"]) == (
        REPO,
        BASE_SHA,
        False,
        2,
    )
    assert data["chunks"] > 0


async def test_file_reviewed_event_is_sent_once_per_file_with_its_comment_count(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, reviewer_with_comment())

    events = await collect(service)

    reviewed = [event.data for event in events if event.name == "file_reviewed"]
    assert sorted(data["file"] for data in reviewed) == REVIEWABLE
    assert [(data["comments"], data["errors"]) for data in reviewed] == [(1, [])] * 3


async def test_verified_event_counts_kept_and_dropped_comments(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, reviewer_with_comment())

    events = await collect(service)

    assert only(events, "verified").data == {"kept": 2, "dropped": 1}


async def test_done_event_validates_as_review_result_and_matches_run(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, reviewer_with_comment())

    done = only(await collect(service), "done")
    result = await service.run(PR_URL)

    assert ReviewResult.model_validate(done.data) == result
    assert done.data == result.model_dump(mode="json")


async def test_run_returns_the_verified_review(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, reviewer_with_comment())

    result = await service.run(PR_URL)

    assert result.pr_url == PR_URL
    assert result.head_sha == HEAD_SHA
    assert sorted(comment.path for comment in result.comments) == [CONTROLLER, SERVICE]
    assert result.dropped_comments == 1
    assert result.file_errors == []
    assert result.summary == "2 comment(s): 2 high, 0 medium, 0 low"


async def test_reviewer_failure_is_reported_for_that_file_and_the_stream_finishes(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    review = FileReview(
        comments=[
            DraftComment(line=LINE_IN_TWO_FILES, severity="high", category="bug", message="Bad")
        ]
    )
    model = ScriptedChatModel(replies=[], review=review)
    service = await make_service(github, FailsForPath(model, SERVICE))
    expected_error = f"{SERVICE}: RuntimeError: model unavailable"

    events = await collect(service)

    reviewed = {e.data["file"]: e.data for e in events if e.name == "file_reviewed"}
    assert sorted(reviewed) == REVIEWABLE
    assert reviewed[SERVICE]["errors"] == [expected_error]
    assert reviewed[SERVICE]["comments"] == 0
    assert reviewed[CONTROLLER]["errors"] == []
    done = ReviewResult.model_validate(only(events, "done").data)
    assert done.file_errors == [expected_error]
    assert [comment.path for comment in done.comments] == [CONTROLLER]


async def test_pr_with_nothing_to_review_finishes_without_file_events(
    make_service: ServiceFactory,
) -> None:
    service = await make_service(FakeGitHub(pr_files=[]), quiet_reviewer())

    events = await collect(service)

    assert names(events) == ["accepted", "pr_loaded", "indexed", "verified", "done"]
    assert only(events, "done").data["comments"] == []


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(PrNotFoundError("Pull request not found"), id="not-found"),
        pytest.param(PrivateRepoError("Repository is private"), id="private"),
        pytest.param(RateLimitError(None), id="rate-limit"),
    ],
)
async def test_github_error_yields_an_error_event_with_its_type_and_message(
    make_service: ServiceFactory, error: Exception
) -> None:
    service = await make_service(BrokenGitHub(error), quiet_reviewer())

    events = await collect(service)

    assert names(events) == ["accepted", "error"]
    assert events[-1].data == {"type": type(error).__name__, "message": str(error)}


async def test_repo_too_large_yields_an_error_event_after_the_pr_loaded(
    qdrant: AsyncQdrantClient,
    qdrant_store: QdrantStore,
    embedder: FakeEmbedder,
    github: FakeGitHub,
) -> None:
    context = make_context(github, qdrant_store, embedder, quiet_reviewer())
    settings = make_settings().model_copy(update={"max_index_files": 1})
    service = ReviewService(
        graph=build_review_graph(InMemorySaver()),
        context=dataclasses.replace(context, settings=settings),
        qdrant=qdrant,
        langfuse=None,
    )

    events = await collect(service)

    assert names(events) == ["accepted", "pr_loaded", "error"]
    assert events[-1].data == {
        "type": RepoTooLargeError.__name__,
        "message": f"{REPO} has 2 C# files; the index limit is 1",
    }


async def test_malformed_url_yields_an_error_event_instead_of_raising(
    make_service: ServiceFactory, github: FakeGitHub
) -> None:
    service = await make_service(github, quiet_reviewer())

    events = await collect(service, "https://example.com/not-a-pr")

    assert names(events) == ["accepted", "error"]
    assert events[-1].data["type"] == "InvalidPrUrlError"


async def test_unexpected_exception_yields_internal_error_without_the_original_message(
    make_service: ServiceFactory, caplog: pytest.LogCaptureFixture
) -> None:
    service = await make_service(BrokenGitHub(RuntimeError(SECRET_DETAIL)), quiet_reviewer())

    with caplog.at_level(logging.ERROR, logger="code_raqeeb.review.service"):
        events = await collect(service)

    assert names(events) == ["accepted", "error"]
    assert events[-1].data == {
        "type": "InternalError",
        "message": "The review failed unexpectedly.",
    }
    assert SECRET_DETAIL not in json.dumps([event.data for event in events])


async def test_unexpected_exception_is_logged_with_its_traceback(
    make_service: ServiceFactory, caplog: pytest.LogCaptureFixture
) -> None:
    service = await make_service(BrokenGitHub(RuntimeError(SECRET_DETAIL)), quiet_reviewer())

    with caplog.at_level(logging.ERROR, logger="code_raqeeb.review.service"):
        await collect(service)

    records = [r for r in caplog.records if r.name == "code_raqeeb.review.service"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert PR_URL in records[0].getMessage()
    assert records[0].exc_info is not None
    assert SECRET_DETAIL in str(records[0].exc_info[1])


async def test_run_lets_github_errors_reach_the_caller(make_service: ServiceFactory) -> None:
    service = await make_service(BrokenGitHub(PrNotFoundError("gone")), quiet_reviewer())

    with pytest.raises(PrNotFoundError, match="gone"):
        await service.run(PR_URL)


async def test_aclose_closes_qdrant_and_shuts_langfuse_down(
    qdrant_store: QdrantStore, embedder: FakeEmbedder, github: FakeGitHub
) -> None:
    fake_qdrant, fake_langfuse = FakeQdrant(), FakeLangfuse()
    service = ReviewService(
        graph=build_review_graph(InMemorySaver()),
        context=make_context(github, qdrant_store, embedder, quiet_reviewer()),
        qdrant=cast(AsyncQdrantClient, fake_qdrant),
        langfuse=cast(Langfuse, fake_langfuse),
    )

    await service.aclose()

    assert fake_qdrant.closed
    assert fake_langfuse.shut_down


async def test_aclose_works_without_langfuse(
    qdrant_store: QdrantStore, embedder: FakeEmbedder, github: FakeGitHub
) -> None:
    fake_qdrant = FakeQdrant()
    service = ReviewService(
        graph=build_review_graph(InMemorySaver()),
        context=make_context(github, qdrant_store, embedder, quiet_reviewer()),
        qdrant=cast(AsyncQdrantClient, fake_qdrant),
        langfuse=None,
    )

    await service.aclose()

    assert fake_qdrant.closed
