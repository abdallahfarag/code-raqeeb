from typing import get_args

import pytest
from qdrant_client import AsyncQdrantClient

from code_raqeeb.rag.models import CodeChunk
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.rules.loader import get_rules
from code_raqeeb.rules.models import Category
from tests.rag.conftest import CODE_COLLECTION, RULES_COLLECTION, make_chunk
from tests.rag.fakes import FakeEmbedder

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

ORDER_METHODS = [
    make_chunk(
        "OrderService.PlaceOrder",
        text="public void PlaceOrder(Order order) { _repository.Save(order); }",
    ),
    make_chunk(
        "OrderService.CancelOrder",
        text="public void CancelOrder(Order order) { _repository.Delete(order); }",
    ),
    make_chunk(
        "OrderService.CancelOrderAsync",
        text="public Task CancelOrderAsync(Order order) { return _repository.DeleteAsync(order); }",
    ),
    make_chunk(
        "OrderService.RefundOrder",
        text="public void RefundOrder(Order order) { _payments.Refund(order); }",
    ),
    make_chunk("Clock.Now", text="public DateTime Now() { return DateTime.UtcNow; }"),
]


async def upsert(store: QdrantStore, embedder: FakeEmbedder, chunks: list[CodeChunk]) -> None:
    embeddings = await embedder.embed_documents([c.embedding_text for c in chunks])
    await store.upsert_chunks(chunks, embeddings)


async def upsert_real_rules(store: QdrantStore, embedder: FakeEmbedder) -> None:
    rules = get_rules().rules
    embeddings = await embedder.embed_documents([rule.description for rule in rules])
    await store.upsert_rules(rules, embeddings)


async def test_ensure_collections_creates_both_collections(
    qdrant: AsyncQdrantClient, empty_store: QdrantStore
) -> None:
    await empty_store.ensure_collections(16)

    assert await qdrant.collection_exists(CODE_COLLECTION)
    assert await qdrant.collection_exists(RULES_COLLECTION)


async def test_ensure_collections_is_idempotent_and_keeps_data(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await upsert(store, embedder, ORDER_METHODS)

    await store.ensure_collections(embedder.dense_size)

    assert (await qdrant.count(CODE_COLLECTION)).count == len(ORDER_METHODS)


async def test_upserting_same_chunks_twice_keeps_point_count(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await upsert(store, embedder, ORDER_METHODS)
    await upsert(store, embedder, ORDER_METHODS)

    assert (await qdrant.count(CODE_COLLECTION)).count == len(ORDER_METHODS)


async def test_upserting_same_rules_twice_keeps_point_count(
    qdrant: AsyncQdrantClient, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await upsert_real_rules(store, embedder)
    await upsert_real_rules(store, embedder)

    assert (await qdrant.count(RULES_COLLECTION)).count == len(get_rules().rules)


async def test_has_snapshot_is_true_only_after_snapshot_is_marked_indexed(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    assert not await store.has_snapshot("octo/hello", "sha1")

    await upsert(store, embedder, [make_chunk("Foo.Bar")])
    assert not await store.has_snapshot("octo/hello", "sha1")

    await store.mark_snapshot_indexed("octo/hello", "sha1")
    assert await store.has_snapshot("octo/hello", "sha1")


async def test_marked_snapshot_is_still_searchable(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await upsert(store, embedder, [make_chunk("Foo.Bar")])
    await store.mark_snapshot_indexed("octo/hello", "sha1")

    hits = await store.search_code("octo/hello", "sha1", await embedder.embed_query("Foo Bar"))

    assert [hit.chunk for hit in hits] == [make_chunk("Foo.Bar")]


@pytest.mark.parametrize(
    ("repo", "sha"),
    [
        pytest.param("octo/hello", "other-sha", id="other-sha"),
        pytest.param("octo/other", "sha1", id="other-repo"),
    ],
)
async def test_has_snapshot_needs_both_repo_and_sha_to_match(
    store: QdrantStore, embedder: FakeEmbedder, repo: str, sha: str
) -> None:
    await upsert(
        store,
        embedder,
        [
            make_chunk("Foo.Bar"),
            make_chunk("Foo.Bar", sha="other-sha"),
            make_chunk("Foo.Bar", repo="octo/other"),
        ],
    )
    await store.mark_snapshot_indexed("octo/hello", "sha1")

    assert not await store.has_snapshot(repo, sha)


@pytest.mark.parametrize(
    ("repo", "sha"),
    [
        pytest.param("octo/hello", "sha1", id="requested-pair"),
        pytest.param("octo/hello", "sha2", id="same-repo-other-sha"),
        pytest.param("octo/other", "sha1", id="other-repo-same-sha"),
    ],
)
async def test_search_code_only_returns_requested_repo_and_sha(
    store: QdrantStore, embedder: FakeEmbedder, repo: str, sha: str
) -> None:
    chunks = [
        make_chunk("Foo.Bar", repo="octo/hello", sha="sha1"),
        make_chunk("Foo.Bar", repo="octo/hello", sha="sha2"),
        make_chunk("Foo.Bar", repo="octo/other", sha="sha1"),
    ]
    await upsert(store, embedder, chunks)

    hits = await store.search_code(repo, sha, await embedder.embed_query("Foo Bar"))

    assert [hit.chunk for hit in hits] == [make_chunk("Foo.Bar", repo=repo, sha=sha)]


async def test_search_code_finds_nothing_for_unknown_snapshot(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await upsert(store, embedder, ORDER_METHODS)

    hits = await store.search_code("octo/hello", "unknown", await embedder.embed_query("Order"))

    assert hits == []


async def test_search_code_respects_limit(store: QdrantStore, embedder: FakeEmbedder) -> None:
    await upsert(store, embedder, ORDER_METHODS)

    query = await embedder.embed_query("Order")
    hits = await store.search_code("octo/hello", "sha1", query, limit=2)

    assert len(hits) == 2


@pytest.mark.parametrize(
    "target", ORDER_METHODS, ids=[target.symbol.split(".")[-1] for target in ORDER_METHODS]
)
async def test_exact_method_name_in_query_ranks_that_chunk_first(
    store: QdrantStore, embedder: FakeEmbedder, target: CodeChunk
) -> None:
    await upsert(store, embedder, ORDER_METHODS)
    name = target.symbol.split(".")[-1]

    query = await embedder.embed_query(f"where is {name} implemented")
    hits = await store.search_code("octo/hello", "sha1", query)

    assert hits[0].chunk == target


@pytest.mark.parametrize("category", get_args(Category))
async def test_search_rules_category_filter_returns_only_that_category(
    store: QdrantStore, embedder: FakeEmbedder, category: Category
) -> None:
    await upsert_real_rules(store, embedder)
    expected = {rule.id for rule in get_rules().rules if rule.category == category}
    assert expected

    query = await embedder.embed_query("avoid unsafe query in async code")
    hits = await store.search_rules(query, limit=100, category=category)

    assert {hit.rule.id for hit in hits} == expected


async def test_search_rules_without_category_searches_every_rule(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await upsert_real_rules(store, embedder)

    query = await embedder.embed_query("avoid unsafe query in async code")
    hits = await store.search_rules(query, limit=100)

    assert {hit.rule.id for hit in hits} == {rule.id for rule in get_rules().rules}
