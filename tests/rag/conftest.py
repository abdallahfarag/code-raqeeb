from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from code_raqeeb.rag.models import CodeChunk
from code_raqeeb.rag.store import QdrantStore
from tests.rag.fakes import FakeEmbedder

CODE_COLLECTION = "test_code"
RULES_COLLECTION = "test_rules"


def make_chunk(
    symbol: str,
    *,
    repo: str = "octo/hello",
    sha: str = "sha1",
    path: str = "src/Foo.cs",
    text: str = "void Foo() { }",
) -> CodeChunk:
    return CodeChunk(
        repo=repo,
        sha=sha,
        path=path,
        symbol=symbol,
        kind="method",
        start_line=1,
        end_line=1,
        text=text,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def empty_store(qdrant: AsyncQdrantClient) -> QdrantStore:
    return QdrantStore(qdrant, CODE_COLLECTION, RULES_COLLECTION)


@pytest.fixture
async def store(empty_store: QdrantStore, embedder: FakeEmbedder) -> QdrantStore:
    await empty_store.ensure_collections(embedder.dense_size)
    return empty_store
