# pyright: reportUnknownMemberType=false

from collections.abc import Sequence
from typing import cast, override

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.models import PrFile
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.graph import build_review_graph, run_review
from code_raqeeb.review.models import DraftComment, FileReview
from code_raqeeb.review.reviewer import FileReviewer
from code_raqeeb.review.state import ReviewState
from code_raqeeb.rules.loader import get_rules
from code_raqeeb.rules.models import Severity
from tests.rag.fakes import FakeEmbedder
from tests.review.conftest import make_context
from tests.review.fakes import (
    BASE_SHA,
    HEAD_SHA,
    PAYMENT_CLIENT_PATH,
    PR_URL,
    REPO,
    FakeGitHub,
    ScriptedChatModel,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

CONTROLLER = "src/Orders.Api/Controllers/OrdersController.cs"
REQUEST = "src/Orders.Api/Models/PayOrderRequest.cs"
SERVICE = "src/Orders.Api/Services/OrderService.cs"
REVIEWABLE = {CONTROLLER, REQUEST, SERVICE}

# Hunks: OrdersController.cs 19-30, PayOrderRequest.cs 1-3, OrderService.cs 13-19 and 22-31
IN_CONTROLLER_AND_SERVICE = 27
IN_CONTROLLER_ONLY = 20
OUTSIDE_EVERY_HUNK = 500

SKIPPED_FILES = [
    PrFile(filename="README.md", status="modified", additions=1, deletions=0, patch="@@"),
    PrFile(filename="src/bin/App.cs", status="modified", additions=1, deletions=0, patch="@@"),
    PrFile(filename="src/Gone.cs", status="removed", additions=0, deletions=1, patch="@@"),
]


def draft(line: int, severity: Severity = "high") -> DraftComment:
    return DraftComment(line=line, severity=severity, category="bug", message=f"Problem at {line}")


def review_of(*comments: DraftComment) -> FileReview:
    return FileReview(comments=list(comments))


def permalink(path: str, line: int) -> str:
    return f"https://github.com/{REPO}/blob/{HEAD_SHA}/{path}#L{line}"


def reviewed_paths(model: ScriptedChatModel) -> list[str]:
    paths: list[str] = []
    for messages in model.structured_calls:
        human = next(m for m in messages if isinstance(m, HumanMessage))
        assert isinstance(human.content, str)
        paths.append(next(p for p in REVIEWABLE if f"File: {p} " in human.content))
    return paths


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


async def run(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, reviewer: FileReviewer
) -> ReviewState:
    graph = build_review_graph(InMemorySaver())
    context = make_context(github, store, embedder, reviewer)
    return await run_review(graph, PR_URL, context, max_concurrency=3)


async def run_with_review(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, review: FileReview
) -> ReviewState:
    reviewer = FileReviewer(ScriptedChatModel(replies=[], review=review), max_tool_rounds=0)
    return await run(github, store, embedder, reviewer)


async def node_order(github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder) -> list[str]:
    reviewer = FileReviewer(ScriptedChatModel(replies=[]), max_tool_rounds=0)
    order: list[str] = []
    async for update in build_review_graph(InMemorySaver()).astream(
        ReviewState(pr_url=PR_URL, comments=[], file_errors=[]),
        config={"configurable": {"thread_id": "order"}},
        context=make_context(github, store, embedder, reviewer),
        stream_mode="updates",
    ):
        order.extend(cast(dict[str, object], update))
    return order


async def test_nodes_run_in_order_with_one_review_per_file(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    assert await node_order(github, store, embedder) == [
        "load_pr",
        "index_base",
        "review_file",
        "review_file",
        "review_file",
        "verify",
        "summarize",
    ]


async def test_nodes_skip_review_when_every_file_is_skipped(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    github = FakeGitHub(pr_files=SKIPPED_FILES)

    assert await node_order(github, store, embedder) == [
        "load_pr",
        "index_base",
        "verify",
        "summarize",
    ]


async def test_one_review_per_reviewable_file(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    model = ScriptedChatModel(replies=[])

    state = await run(github, store, embedder, FileReviewer(model, max_tool_rounds=0))

    assert sorted(reviewed_paths(model)) == sorted(REVIEWABLE)
    assert state["file_errors"] == []


async def test_comment_inside_the_diff_is_kept_with_the_right_permalink(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    line = IN_CONTROLLER_AND_SERVICE

    state = await run_with_review(github, store, embedder, review_of(draft(line)))

    kept = {c.path: c.permalink for c in state.get("verified_comments", [])}
    assert kept == {CONTROLLER: permalink(CONTROLLER, line), SERVICE: permalink(SERVICE, line)}
    assert state.get("dropped_comments") == 1


async def test_comment_outside_any_hunk_is_dropped_and_counted(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    review = review_of(draft(OUTSIDE_EVERY_HUNK))

    state = await run_with_review(github, store, embedder, review)

    assert state.get("verified_comments") == []
    assert state.get("dropped_comments") == len(REVIEWABLE)
    assert len(state["comments"]) == len(REVIEWABLE)


async def test_comment_in_the_gap_between_two_hunks_is_dropped(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    state = await run_with_review(github, store, embedder, review_of(draft(IN_CONTROLLER_ONLY)))

    assert [c.path for c in state.get("verified_comments", [])] == [CONTROLLER]
    assert state.get("dropped_comments") == 2


async def test_reviewer_failure_is_recorded_and_other_files_are_still_reviewed(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    model = ScriptedChatModel(replies=[], review=review_of(draft(IN_CONTROLLER_AND_SERVICE)))

    state = await run(github, store, embedder, FailsForPath(model, SERVICE))

    assert state["file_errors"] == [f"{SERVICE}: RuntimeError: model unavailable"]
    assert sorted(reviewed_paths(model)) == sorted({CONTROLLER, REQUEST})
    assert [c.path for c in state.get("verified_comments", [])] == [CONTROLLER]
    assert state.get("summary") == (
        "1 comment(s): 1 high, 0 medium, 0 low; 1 file(s) could not be reviewed"
    )


async def test_pr_with_only_skipped_files_goes_straight_to_verify(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    model = ScriptedChatModel(replies=[], review=review_of(draft(1)))

    state = await run(
        FakeGitHub(pr_files=SKIPPED_FILES), store, embedder, FileReviewer(model, max_tool_rounds=0)
    )

    assert model.calls == []
    assert state.get("verified_comments") == []
    assert state.get("dropped_comments") == 0
    assert state.get("summary") == "0 comment(s): 0 high, 0 medium, 0 low"
    bundle = state.get("bundle")
    assert bundle is not None
    assert [(s.path, s.reason) for s in bundle.skipped] == [
        ("README.md", "not a C# file"),
        ("src/bin/App.cs", "build output"),
        ("src/Gone.cs", "deleted"),
    ]


async def test_summary_counts_each_severity_of_kept_comments_only(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    review = review_of(
        draft(24, "high"),
        draft(25, "medium"),
        draft(26, "medium"),
        draft(27, "low"),
        draft(28, "low"),
        draft(29, "low"),
        draft(OUTSIDE_EVERY_HUNK, "high"),
    )

    state = await run_with_review(github, store, embedder, review)

    assert state.get("summary") == "12 comment(s): 2 high, 4 medium, 6 low"
    assert state.get("dropped_comments") == 9


async def test_base_snapshot_is_indexed_and_reused_by_the_next_run(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    first = await run_with_review(github, store, embedder, review_of())
    second = await run_with_review(github, store, embedder, review_of())

    first_report = first.get("index_report")
    second_report = second.get("index_report")
    assert first_report is not None
    assert (first_report.sha, first_report.skipped, first_report.files) == (BASE_SHA, False, 2)
    assert second_report is not None
    assert second_report.skipped
    assert github.tree_shas == [BASE_SHA]


async def test_every_rule_is_indexed_for_the_reviewer(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    await run_with_review(github, store, embedder, review_of())

    hits = await store.search_rules(await embedder.embed_query("rule"), limit=100)

    assert {hit.rule.id for hit in hits} == {rule.id for rule in get_rules().rules}


async def test_reviewer_tools_search_the_indexed_base_code(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder
) -> None:
    asks = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_code", "args": {"query": "ChargeAsync"}, "id": f"call-{n}"}
            ],
        )
        for n in range(len(REVIEWABLE))
    ]
    model = ScriptedChatModel(replies=asks)

    state = await run(github, store, embedder, FileReviewer(model, max_tool_rounds=1))

    assert state["file_errors"] == []
    assert len(model.structured_calls) == len(REVIEWABLE)
    for messages in model.structured_calls:
        last = messages[-1]
        assert isinstance(last, ToolMessage)
        assert f"{PAYMENT_CLIENT_PATH}:5-8 (PaymentClient.ChargeAsync)" in str(last.content)
