from collections.abc import Iterable, Sequence
from typing import cast

import pytest
from qdrant_client import AsyncQdrantClient

from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.models import PrRef
from code_raqeeb.rag.embeddings import Embedding
from code_raqeeb.rag.indexer import IndexReport, RepoTooLargeError, index_rules, index_snapshot
from code_raqeeb.rag.models import CodeChunk
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.rules.loader import get_rules
from code_raqeeb.rules.models import ReviewRule
from tests.rag.conftest import CODE_COLLECTION, RULES_COLLECTION
from tests.rag.fakes import FakeEmbedder

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

REF = PrRef(owner="octo", repo="hello", number=7)
SHA = "sha1"
SCROLL_LIMIT = 1000

ORDERS_CS = """\
class Orders
{
    void Place() { }
    void Cancel() { }
}
"""
MONEY_CS = "record Money(decimal Amount);\n"
SOURCE_FILES = {"src/Orders.cs": ORDERS_CS, "src/Money.cs": MONEY_CS}
REPO_FILES = {
    **SOURCE_FILES,
    "src/bin/Debug/Generated.cs": "class Generated { void Skipped() { } }\n",
    "README.md": "# Hello\n",
}


class FakeGitHub:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.tree_shas: list[str] = []
        self.requested: list[str] = []

    async def list_tree(self, ref: PrRef, sha: str) -> list[str]:
        self.tree_shas.append(sha)
        return list(self.files)

    async def get_files_content(self, ref: PrRef, paths: Iterable[str], sha: str) -> dict[str, str]:
        wanted = list(paths)
        self.requested.extend(wanted)
        return {path: self.files[path] for path in wanted}


class FailsOnBatch(FakeEmbedder):
    def __init__(self, failing_batch: int) -> None:
        super().__init__()
        self.failing_batch = failing_batch
        self.armed = True

    async def embed_documents(self, texts: Sequence[str]) -> list[Embedding]:
        if self.armed and len(self.document_batches) + 1 == self.failing_batch:
            self.armed = False
            raise RuntimeError("embedding service unavailable")
        return await super().embed_documents(texts)


async def run_index(
    github: FakeGitHub,
    store: QdrantStore,
    embedder: FakeEmbedder,
    *,
    sha: str = SHA,
    max_files: int = 10,
) -> IndexReport:
    return await index_snapshot(
        cast(GitHubClient, github),
        store,
        embedder,
        REF,
        sha,
        max_files=max_files,
        max_chunk_lines=100,
    )


async def stored_chunks(qdrant: AsyncQdrantClient) -> list[CodeChunk]:
    points, _ = await qdrant.scroll(CODE_COLLECTION, limit=SCROLL_LIMIT, with_payload=True)
    return [CodeChunk.model_validate(point.payload) for point in points]


def many_methods(count: int) -> str:
    methods = "".join(f"    void M{n}() {{ }}\n" for n in range(count))
    return f"class Big\n{{\n{methods}}}\n"


async def test_only_reviewable_csharp_files_are_fetched_and_indexed(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(REPO_FILES)

    await run_index(github, store, embedder)

    assert sorted(github.requested) == sorted(SOURCE_FILES)
    assert {chunk.path for chunk in await stored_chunks(qdrant)} == set(SOURCE_FILES)


async def test_report_counts_files_and_chunks(store: QdrantStore, embedder: FakeEmbedder) -> None:
    report = await run_index(FakeGitHub(REPO_FILES), store, embedder)

    assert report == IndexReport(repo="octo/hello", sha=SHA, skipped=False, files=2, chunks=3)


async def test_stored_chunks_match_the_report_and_snapshot(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await run_index(FakeGitHub(REPO_FILES), store, embedder)

    chunks = await stored_chunks(qdrant)

    assert {chunk.symbol for chunk in chunks} == {"Orders.Place", "Orders.Cancel", "Money"}
    assert {(chunk.repo, chunk.sha) for chunk in chunks} == {("octo/hello", SHA)}


async def test_second_call_is_skipped_without_fetching_or_embedding(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(REPO_FILES)
    await run_index(github, store, embedder)
    requested = list(github.requested)
    batches = list(embedder.document_batches)

    report = await run_index(github, store, embedder)

    assert report == IndexReport(repo="octo/hello", sha=SHA, skipped=True)
    assert github.tree_shas == [SHA]
    assert github.requested == requested
    assert embedder.document_batches == batches
    assert (await qdrant.count(CODE_COLLECTION)).count == 3


async def test_a_new_sha_of_the_same_repo_is_indexed_again(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(REPO_FILES)
    await run_index(github, store, embedder, sha="sha1")

    report = await run_index(github, store, embedder, sha="sha2")

    assert not report.skipped
    assert github.tree_shas == ["sha1", "sha2"]
    assert (await qdrant.count(CODE_COLLECTION)).count == 6


async def test_more_files_than_max_files_raises_before_fetching_anything(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(REPO_FILES)

    with pytest.raises(RepoTooLargeError, match="octo/hello has 2 C# files"):
        await run_index(github, store, embedder, max_files=1)

    assert github.requested == []
    assert (await qdrant.count(CODE_COLLECTION)).count == 0


async def test_skipped_files_do_not_count_towards_max_files(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    # Four paths in the tree, but only two of them are C# sources
    report = await run_index(FakeGitHub(REPO_FILES), store, embedder, max_files=2)

    assert report.files == 2


async def test_chunks_spanning_several_embedding_batches_are_all_stored(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    report = await run_index(FakeGitHub({"src/Big.cs": many_methods(65)}), store, embedder)

    assert report.chunks == 65
    assert [len(batch) for batch in embedder.document_batches] == [64, 1]
    assert (await qdrant.count(CODE_COLLECTION)).count == 65


async def test_index_rules_indexes_every_rule(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    rules = get_rules()

    count = await index_rules(store, embedder, rules)

    points, _ = await qdrant.scroll(RULES_COLLECTION, limit=SCROLL_LIMIT, with_payload=True)
    stored = [ReviewRule.model_validate(point.payload) for point in points]
    assert count == len(rules.rules)
    assert sorted(stored, key=lambda rule: rule.id) == sorted(rules.rules, key=lambda r: r.id)


async def test_failed_index_is_not_treated_as_complete_on_retry(
    qdrant: AsyncQdrantClient, store: QdrantStore
) -> None:
    github = FakeGitHub({"src/Big.cs": many_methods(65)})
    embedder = FailsOnBatch(failing_batch=2)
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        await run_index(github, store, embedder)

    report = await run_index(github, store, embedder)

    assert not report.skipped
    assert (await qdrant.count(CODE_COLLECTION)).count == 65
