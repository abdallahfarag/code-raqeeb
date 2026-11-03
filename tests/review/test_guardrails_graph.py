# pyright: reportUnknownMemberType=false

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast, override

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import TypeAdapter

from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.models import PrFile, PullRequest
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.guardrails.injection import InjectionScanner
from code_raqeeb.guardrails.secrets import REDACTED
from code_raqeeb.rag.indexer import index_snapshot
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.graph import build_review_graph, run_review
from code_raqeeb.review.models import DraftComment, FileReview, ReviewComment
from code_raqeeb.review.prompts import build_file_prompt
from code_raqeeb.review.reviewer import FileReviewer
from code_raqeeb.review.state import ReviewState
from code_raqeeb.review.tools import build_tools
from tests.evals.models import ExpectedReview
from tests.guardrails.fakes import AWS_KEY, FakeInjectionScanner, make_file
from tests.rag.fakes import FakeEmbedder
from tests.review.conftest import idle_reviewer, make_context
from tests.review.fakes import BASE_SHA, HEAD_SHA, FakeGitHub, ScriptedChatModel

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "prs"
PAGINATION = "05-pagination"
SHIPPING = "04-shipping-quotes"
ORDER_SERVICE = "src/Orders.Api/Services/OrderService.cs"
SHIPPING_CLIENT = "src/Orders.Api/Services/ShippingClient.cs"
FAKE_KEY = "demo-shipping-key-7f3a9c2e5b1d4086"
INJECTION_LINES = (11, 12)
FILE_PATH = "src/Foo.cs"

_PR_FILES = TypeAdapter(list[PrFile])


def recorded_github(name: str) -> FakeGitHub:
    folder = FIXTURES_DIR / name
    github = FakeGitHub(
        pr_files=_PR_FILES.validate_json((folder / "files.json").read_text(encoding="utf-8"))
    )
    github.pull_request = PullRequest.model_validate_json(
        (folder / "pr.json").read_text(encoding="utf-8")
    )
    return github


def recorded_url(name: str) -> str:
    text = (FIXTURES_DIR / name / "expected.json").read_text(encoding="utf-8")
    return ExpectedReview.model_validate_json(text).pr_url


class AlwaysFails(FileReviewer):
    def __init__(self) -> None:
        super().__init__(ScriptedChatModel(replies=[]), max_tool_rounds=0)

    @override
    async def review(
        self, bundle: PullRequestBundle, file: ChangedFile, tools: Sequence[BaseTool]
    ) -> list[DraftComment]:
        raise RuntimeError("model unavailable")


async def run_recorded(
    name: str,
    store: QdrantStore,
    embedder: FakeEmbedder,
    reviewer: FileReviewer,
    scanner: InjectionScanner | None = None,
) -> ReviewState:
    context = replace(
        make_context(recorded_github(name), store, embedder, reviewer),
        injection_scanner=scanner,
    )
    return await run_review(
        build_review_graph(InMemorySaver()), recorded_url(name), context, max_concurrency=3
    )


def security_findings(state: ReviewState, rule_id: str) -> list[ReviewComment]:
    return [c for c in state.get("verified_comments", []) if c.rule_id == rule_id]


def human_prompts(model: ScriptedChatModel) -> list[str]:
    prompts: list[str] = []
    for messages in model.calls:
        for message in messages:
            if isinstance(message, HumanMessage):
                assert isinstance(message.content, str)
                prompts.append(message.content)
    return prompts


async def test_sec_03_is_verified_when_the_model_returns_no_comments(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    model = ScriptedChatModel(replies=[])

    state = await run_recorded(PAGINATION, store, embedder, FileReviewer(model, max_tool_rounds=0))

    findings = security_findings(state, "SEC-03")
    assert [(c.path, c.line) for c in findings] == [(ORDER_SERVICE, n) for n in INJECTION_LINES]
    assert state["file_errors"] == []
    assert len(model.structured_calls) == 3


async def test_sec_03_is_verified_when_the_reviewer_raises(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    state = await run_recorded(PAGINATION, store, embedder, AlwaysFails())

    findings = security_findings(state, "SEC-03")
    assert [(c.path, c.line) for c in findings] == [(ORDER_SERVICE, n) for n in INJECTION_LINES]
    assert len(state["file_errors"]) == 3
    assert state.get("summary") == (
        "2 comment(s): 2 high, 0 medium, 0 low; 3 file(s) could not be reviewed"
    )


async def test_scanner_finding_wins_over_a_weaker_duplicate_from_the_model(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    review = FileReview(
        comments=[
            DraftComment(
                line=INJECTION_LINES[0],
                severity="low",
                category="bug",
                rule_id="SEC-03",
                message="From the model",
            )
        ]
    )
    model = ScriptedChatModel(replies=[], review=review)

    state = await run_recorded(PAGINATION, store, embedder, FileReviewer(model, max_tool_rounds=0))

    [first, _] = security_findings(state, "SEC-03")
    assert (first.line, first.severity) == (INJECTION_LINES[0], "high")
    assert "From the model" not in first.message


async def test_context_scanner_reaches_the_scan(store: QdrantStore, embedder: FakeEmbedder) -> None:
    scanner = FakeInjectionScanner()

    await run_recorded(SHIPPING, store, embedder, idle_reviewer(), scanner)

    assert any("https://shipping.example.com/" in text for text in scanner.calls)


async def test_prompt_sent_to_the_model_hides_the_fake_key(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    model = ScriptedChatModel(replies=[])

    await run_recorded(SHIPPING, store, embedder, FileReviewer(model, max_tool_rounds=0))

    prompts = human_prompts(model)
    [shipping_prompt] = [p for p in prompts if f"File: {SHIPPING_CLIENT} " in p]
    assert f'private const string ApiKey = "{REDACTED}";' in shipping_prompt
    assert all(FAKE_KEY not in prompt for prompt in prompts)


async def test_sec_02_is_verified_without_the_fake_key(
    store: QdrantStore, embedder: FakeEmbedder
) -> None:
    state = await run_recorded(SHIPPING, store, embedder, idle_reviewer())

    [finding] = security_findings(state, "SEC-02")
    assert (finding.path, finding.line) == (SHIPPING_CLIENT, 7)
    assert FAKE_KEY not in finding.model_dump_json()


@pytest.mark.parametrize(
    "separator",
    [pytest.param("\x0c", id="form-feed"), pytest.param("\N{LINE SEPARATOR}", id="line-separator")],
)
def test_prompt_hides_a_secret_after_a_unicode_line_break(
    bundle: PullRequestBundle, separator: str
) -> None:
    file = make_file(f"@@ -1,1 +1,2 @@\n first\n+// fine{separator}{AWS_KEY}")

    assert AWS_KEY[1:] not in build_file_prompt(bundle, file)


def tools_for(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> dict[str, BaseTool]:
    context = make_context(github, store, embedder, idle_reviewer())
    return {item.name: item for item in build_tools(context, bundle)}


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("../x", id="parent"),
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("src\\Foo.cs", id="backslash"),
    ],
)
async def test_read_file_rejects_unsafe_paths_without_reading(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle, path: str
) -> None:
    github = FakeGitHub(head_files={"x": "content\n"})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = str(await read.ainvoke({"path": path}))

    assert output == "Path rejected: use a relative path inside the repository."
    assert github.content_reads == []


async def test_read_file_reads_the_normalized_path(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: "content\n"})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = str(await read.ainvoke({"path": f"./{FILE_PATH}"}))

    assert output == "     1  content"
    assert github.content_reads == [(FILE_PATH, HEAD_SHA)]


async def test_read_file_redacts_secrets(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: f'var value = "{AWS_KEY}";\n'})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    assert str(await read.ainvoke({"path": FILE_PATH})) == f'     1  var value = "{REDACTED}";'


async def test_search_code_redacts_secrets(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    source = f'class Keys\n{{\n    void Run()\n    {{\n        Use("{AWS_KEY}");\n    }}\n}}\n'
    github = FakeGitHub(base_files={FILE_PATH: source})
    await index_snapshot(
        cast(GitHubClient, github),
        store,
        embedder,
        bundle.ref,
        BASE_SHA,
        max_files=10,
        max_chunk_lines=100,
    )
    search = tools_for(github, store, embedder, bundle)["search_code"]

    output = str(await search.ainvoke({"query": "Run"}))

    assert REDACTED in output
    assert AWS_KEY not in output
