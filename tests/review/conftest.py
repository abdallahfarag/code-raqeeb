from collections.abc import AsyncIterator
from typing import cast

import pytest
from qdrant_client import AsyncQdrantClient

from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.service import PullRequestBundle, load_pull_request
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.context import ReviewContext
from code_raqeeb.review.reviewer import FileReviewer
from tests.github.factories import make_settings
from tests.rag.fakes import FakeEmbedder
from tests.review.fakes import PR_URL, FakeGitHub, ScriptedChatModel


def make_context(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, reviewer: FileReviewer
) -> ReviewContext:
    return ReviewContext(
        settings=make_settings(),
        github=cast(GitHubClient, github),
        store=store,
        embedder=embedder,
        reviewer=reviewer,
    )


def idle_reviewer() -> FileReviewer:
    return FileReviewer(ScriptedChatModel(replies=[]), max_tool_rounds=0)


# Keeps a tracing setting in a developer's shell from reaching the network
@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def store(embedder: FakeEmbedder) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(location=":memory:")
    qdrant_store = QdrantStore(client, "test_code", "test_rules")
    await qdrant_store.ensure_collections(embedder.dense_size)
    yield qdrant_store
    await client.close()


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
async def bundle(github: FakeGitHub) -> PullRequestBundle:
    return await load_pull_request(cast(GitHubClient, github), PR_URL, make_settings())
