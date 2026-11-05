from typing import cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import AsyncQdrantClient

from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import PrNotFoundError
from code_raqeeb.github.models import PrRef, PullRequest
from code_raqeeb.guardrails.secrets import REDACTED
from code_raqeeb.rag.indexer import index_snapshot
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.graph import build_review_graph
from code_raqeeb.review.reviewer import FileReviewer
from code_raqeeb.review.service import ReviewService
from code_raqeeb.rules.loader import get_rules
from tests.rag.fakes import FakeEmbedder
from tests.review.conftest import idle_reviewer, make_context
from tests.review.fakes import (
    BASE_SHA,
    BASE_TREE,
    HEAD_SHA,
    ORDER_PATH,
    PAYMENT_CLIENT_PATH,
    PR_URL,
    REPO,
    FakeGitHub,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
API_KEY = "hunter2hunter2"
LEAKY_CLIENT = f"""\
namespace Orders.Api.Services;

public sealed class PaymentClient
{{
    public Task ChargeAsync(int orderId, decimal amount, CancellationToken cancellationToken)
    {{
        var awsKey = "{AWS_KEY}";
        var apiKey = "{API_KEY}";
        return Task.CompletedTask;
    }}
}}
"""


class MissingPrGitHub(FakeGitHub):
    async def get_pull_request(self, ref: PrRef) -> PullRequest:
        raise PrNotFoundError("Pull request not found")


def make_service(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, reviewer: FileReviewer
) -> ReviewService:
    return ReviewService(
        graph=build_review_graph(InMemorySaver()),
        context=make_context(github, store, embedder, reviewer),
        qdrant=cast(AsyncQdrantClient, object()),
        langfuse=None,
    )


@pytest.fixture
def service(github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder) -> ReviewService:
    return make_service(github, store, embedder, idle_reviewer())


async def test_search_code_indexes_the_base_commit_first(
    service: ReviewService, github: FakeGitHub
) -> None:
    assert github.tree_shas == []

    await service.search_code(PR_URL, "ChargeAsync", 5)

    assert github.tree_shas == [BASE_SHA]


async def test_search_code_returns_hits_from_the_prs_repo_and_base_sha(
    service: ReviewService,
) -> None:
    hits = await service.search_code(PR_URL, "ChargeAsync", 10)

    assert {hit.chunk.path for hit in hits} == {PAYMENT_CLIENT_PATH, ORDER_PATH}
    assert {(hit.chunk.repo, hit.chunk.sha) for hit in hits} == {(REPO, BASE_SHA)}
    assert "PaymentClient.ChargeAsync" in {hit.chunk.symbol for hit in hits}


async def test_search_code_ignores_snapshots_of_other_commits(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(head_files=BASE_TREE)
    service = make_service(github, store, embedder, idle_reviewer())
    await index_snapshot(
        cast(GitHubClient, github),
        store,
        embedder,
        PrRef.from_url(PR_URL),
        HEAD_SHA,
        max_files=500,
        max_chunk_lines=150,
    )

    hits = await service.search_code(PR_URL, "ChargeAsync", 10)

    assert hits
    assert {hit.chunk.sha for hit in hits} == {BASE_SHA}


async def test_search_code_respects_the_limit(service: ReviewService) -> None:
    hits = await service.search_code(PR_URL, "ChargeAsync", 1)

    assert len(hits) == 1


async def test_search_code_reuses_the_index_on_the_next_call(
    service: ReviewService, github: FakeGitHub
) -> None:
    await service.search_code(PR_URL, "ChargeAsync", 5)
    await service.search_code(PR_URL, "Order", 5)

    assert github.tree_shas == [BASE_SHA]


async def test_search_code_redacts_secrets_in_chunk_text(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(base_files={PAYMENT_CLIENT_PATH: LEAKY_CLIENT})
    service = make_service(github, store, embedder, idle_reviewer())

    hits = await service.search_code(PR_URL, "ChargeAsync", 10)

    leaky = next(hit for hit in hits if hit.chunk.symbol == "PaymentClient.ChargeAsync")
    assert leaky.chunk.text.count(REDACTED) == 2
    assert "return Task.CompletedTask;" in leaky.chunk.text
    for hit in hits:
        assert AWS_KEY not in hit.chunk.text
        assert API_KEY not in hit.chunk.text


async def test_search_code_leaves_clean_chunk_text_unchanged(service: ReviewService) -> None:
    hits = await service.search_code(PR_URL, "ChargeAsync", 10)

    assert all(REDACTED not in hit.chunk.text for hit in hits)
    method = next(hit for hit in hits if hit.chunk.symbol == "PaymentClient.ChargeAsync")
    assert "return Task.CompletedTask;" in method.chunk.text


async def test_search_code_lets_github_errors_reach_the_caller(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    service = make_service(MissingPrGitHub(), store, embedder, idle_reviewer())

    with pytest.raises(PrNotFoundError, match="Pull request not found"):
        await service.search_code(PR_URL, "ChargeAsync", 5)


async def test_find_rules_returns_rules_from_the_rule_set(service: ReviewService) -> None:
    hits = await service.find_rules("async", 5)

    assert len(hits) == 5
    known = {rule.id: rule for rule in get_rules().rules}
    assert all(known[hit.rule.id] == hit.rule for hit in hits)


async def test_find_rules_indexes_each_rule_only_once(service: ReviewService) -> None:
    await service.find_rules("async", 5)

    hits = await service.find_rules("async", 1000)

    assert sorted(hit.rule.id for hit in hits) == sorted(rule.id for rule in get_rules().rules)
