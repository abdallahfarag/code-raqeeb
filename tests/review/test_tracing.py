# pyright: reportPrivateUsage=false

import uuid
from typing import Any, cast, override
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from code_raqeeb.config import Settings
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review import service as service_module
from code_raqeeb.review import tracing
from code_raqeeb.review.graph import _config, build_review_graph, run_review, stream_review
from code_raqeeb.review.reviewer import FileReviewer
from code_raqeeb.review.service import ReviewService
from code_raqeeb.review.tracing import init_langfuse, tracing_callbacks
from tests.rag.fakes import FakeEmbedder
from tests.review.conftest import make_context
from tests.review.fakes import PR_URL, REPO, FakeGitHub, ScriptedChatModel

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

TAGS = [REPO, "pr-2"]


class RecordingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.metadata.append(metadata or {})


class FakeLangfuse:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeCallbackHandler(BaseCallbackHandler):
    pass


def tracing_settings(public_key: str | None, secret_key: str | None) -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        langfuse_host="http://langfuse.test:3000",  # pyright: ignore[reportArgumentType]
        langfuse_public_key=public_key,
        langfuse_secret_key=SecretStr(secret_key) if secret_key is not None else None,
    )


def idle_reviewer() -> FileReviewer:
    return FileReviewer(ScriptedChatModel(replies=[]), max_tool_rounds=0)


def session_ids(handler: RecordingHandler) -> set[str]:
    return {data["langfuse_session_id"] for data in handler.metadata}


@pytest.mark.parametrize(
    ("public_key", "secret_key"),
    [
        pytest.param(None, None, id="no-keys"),
        pytest.param("pk-test", None, id="only-public-key"),
        pytest.param(None, "sk-test", id="only-secret-key"),
    ],
)
def test_init_langfuse_returns_none_without_both_keys(
    monkeypatch: pytest.MonkeyPatch, public_key: str | None, secret_key: str | None
) -> None:
    monkeypatch.setattr(tracing, "Langfuse", FakeLangfuse)

    assert init_langfuse(tracing_settings(public_key, secret_key)) is None


def test_init_langfuse_builds_a_client_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "Langfuse", FakeLangfuse)

    client = init_langfuse(tracing_settings("pk-test", "sk-test"))

    assert isinstance(client, FakeLangfuse)
    assert client.kwargs == {
        "public_key": "pk-test",
        "secret_key": "sk-test",
        "host": "http://langfuse.test:3000/",
    }


def test_tracing_callbacks_is_empty_without_a_client() -> None:
    assert tracing_callbacks(None) == []


def test_tracing_callbacks_returns_a_handler_with_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "CallbackHandler", FakeCallbackHandler)

    callbacks = tracing_callbacks(cast(Langfuse, FakeLangfuse()))

    assert [type(callback) for callback in callbacks] == [FakeCallbackHandler]


def test_config_contains_the_callbacks_and_the_langfuse_metadata() -> None:
    handler = RecordingHandler()

    config = _config(4, [handler], TAGS)

    assert config.get("callbacks") == [handler]
    assert config.get("max_concurrency") == 4
    metadata = config.get("metadata")
    assert metadata is not None
    assert metadata["langfuse_tags"] == TAGS


def test_config_uses_one_new_uuid_as_thread_and_session_id() -> None:
    first = _config(1, [], [])
    second = _config(1, [], [])

    first_configurable = first.get("configurable")
    first_metadata = first.get("metadata")
    second_configurable = second.get("configurable")
    assert first_configurable is not None
    assert first_metadata is not None
    assert second_configurable is not None
    thread_id = first_configurable["thread_id"]
    assert uuid.UUID(thread_id)
    assert first_metadata["langfuse_session_id"] == thread_id
    assert second_configurable["thread_id"] != thread_id


async def test_run_review_hands_callbacks_and_metadata_to_the_graph(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    handler = RecordingHandler()

    await run_review(
        build_review_graph(InMemorySaver()),
        PR_URL,
        make_context(github, store, embedder, idle_reviewer()),
        max_concurrency=2,
        callbacks=[handler],
        tags=TAGS,
    )

    assert handler.metadata
    assert all(data["langfuse_tags"] == TAGS for data in handler.metadata)
    assert len(session_ids(handler)) == 1


async def test_stream_review_hands_callbacks_and_metadata_to_the_graph(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    handler = RecordingHandler()

    updates = [
        node
        async for node, _ in stream_review(
            build_review_graph(InMemorySaver()),
            PR_URL,
            make_context(github, store, embedder, idle_reviewer()),
            max_concurrency=2,
            callbacks=[handler],
            tags=TAGS,
        )
    ]

    assert updates[0] == "load_pr"
    assert updates[-1] == "summarize"
    assert all(data["langfuse_tags"] == TAGS for data in handler.metadata)
    assert len(session_ids(handler)) == 1


async def test_service_traces_run_and_stream_with_the_repo_and_pr_tags(
    monkeypatch: pytest.MonkeyPatch,
    github: FakeGitHub,
    store: QdrantStore,
    embedder: FakeEmbedder,
) -> None:
    handler = RecordingHandler()

    def recording_callbacks(client: Langfuse | None) -> list[BaseCallbackHandler]:
        return [handler]

    monkeypatch.setattr(service_module, "tracing_callbacks", recording_callbacks)
    service = ReviewService(
        graph=build_review_graph(InMemorySaver()),
        context=make_context(github, store, embedder, idle_reviewer()),
        qdrant=cast(AsyncQdrantClient, object()),
        langfuse=None,
    )

    await service.run(PR_URL)
    _ = [event async for event in service.stream(PR_URL)]

    assert all(data["langfuse_tags"] == TAGS for data in handler.metadata)
    assert len(session_ids(handler)) == 2
