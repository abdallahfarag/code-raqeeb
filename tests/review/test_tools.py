import re
from typing import cast, override

import pytest
from langchain_core.tools import BaseTool

from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import GitHubApiError, GitHubError, PrNotFoundError, RateLimitError
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.rag.embeddings import Embedding
from code_raqeeb.rag.indexer import index_rules, index_snapshot
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.tools import build_tools
from code_raqeeb.rules.loader import get_rules
from code_raqeeb.rules.models import ReviewRule, RuleSet
from tests.rag.fakes import FakeEmbedder
from tests.review.conftest import idle_reviewer, make_context
from tests.review.fakes import (
    BASE_SHA,
    BASE_TREE,
    HEAD_SHA,
    PAYMENT_CLIENT_PATH,
    FakeGitHub,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

TRUNCATED = "\n[truncated]"
MAX_TOOL_TEXT = 4000
FILE_PATH = "src/Foo.cs"


def tools_for(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> dict[str, BaseTool]:
    ctx = make_context(github, store, embedder, idle_reviewer())
    return {item.name: item for item in build_tools(ctx, bundle)}


async def call(tool: BaseTool, **args: object) -> str:
    return str(await tool.ainvoke(args))


async def index_base(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    await index_snapshot(
        cast(GitHubClient, github),
        store,
        embedder,
        bundle.ref,
        BASE_SHA,
        max_files=10,
        max_chunk_lines=100,
    )


async def test_build_tools_returns_the_three_tools(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    assert list(tools_for(github, store, embedder, bundle)) == [
        "search_code",
        "get_rules",
        "read_file",
    ]


async def test_search_code_returns_path_range_symbol_and_code(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    await index_base(github, store, embedder, bundle)
    search = tools_for(github, store, embedder, bundle)["search_code"]

    output = await call(search, query="ChargeAsync")

    first = output.split("\n\n")[0]
    assert first.splitlines()[0] == f"{PAYMENT_CLIENT_PATH}:5-8 (PaymentClient.ChargeAsync)"
    assert "return Task.CompletedTask;" in first


async def test_search_code_only_sees_the_base_snapshot(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files=BASE_TREE)
    await index_snapshot(
        cast(GitHubClient, github),
        store,
        embedder,
        bundle.ref,
        HEAD_SHA,
        max_files=10,
        max_chunk_lines=100,
    )
    search = tools_for(github, store, embedder, bundle)["search_code"]

    assert await call(search, query="ChargeAsync") == "No matching code found."


async def test_search_code_output_is_clipped(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    body = "\n".join(f"        Step{n}({'x' * 500});" for n in range(20))
    big = f"class Big\n{{\n    void Run()\n    {{\n{body}\n    }}\n}}\n"
    await index_snapshot(
        cast(GitHubClient, FakeGitHub(base_files={"src/Big.cs": big})),
        store,
        embedder,
        bundle.ref,
        BASE_SHA,
        max_files=10,
        max_chunk_lines=10_000,
    )
    search = tools_for(github, store, embedder, bundle)["search_code"]

    output = await call(search, query="Run")

    assert output.endswith(TRUNCATED)
    assert len(output) == MAX_TOOL_TEXT + len(TRUNCATED)


async def test_get_rules_lists_rule_ids_with_details_under_the_topic(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    rules = get_rules()
    await index_rules(store, embedder, rules)
    get_rules_tool = tools_for(github, store, embedder, bundle)["get_rules"]

    output = await call(get_rules_tool, topics=["RES-01"])

    [heading, *lines] = output.splitlines()
    assert heading == "Topic: RES-01"
    assert 1 <= len(lines) <= 3
    ids = [re.match(r"- ([A-Z]{3}-\d{2}) ", line) for line in lines]
    assert all(match is not None and rules.get(match[1]) is not None for match in ids)
    rule = rules.get("RES-01")
    assert rule is not None
    assert f"- RES-01 ({rule.category}, {rule.severity}): {rule.title}. {rule.description}" in lines


async def test_get_rules_says_so_when_a_topic_has_no_rules(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    get_rules_tool = tools_for(github, store, embedder, bundle)["get_rules"]

    output = await call(get_rules_tool, topics=["anything"])

    assert output == "Topic: anything\n- No rules found."


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    @override
    async def embed_query(self, text: str) -> Embedding:
        self.queries.append(text)
        return await super().embed_query(text)


def topic_rule(rule_id: str, title: str, description: str) -> ReviewRule:
    return ReviewRule(
        id=rule_id, category="bug", severity="medium", title=title, description=description
    )


TOPIC_RULES = RuleSet(
    language="csharp",
    rules=(
        topic_rule("TST-01", "SqlCommand concatenation", "SqlCommand text joined with input."),
        topic_rule("TST-02", "Thread Sleep", "Thread Sleep blocks a request thread."),
        topic_rule("TST-03", "Empty catch", "Exceptions swallowed by an empty catch."),
        topic_rule("TST-04", "Magic number", "Unnamed numeric literal in logic."),
        topic_rule("TST-05", "DateTime Now", "DateTime Now used where UTC is expected."),
    ),
)


@pytest.fixture
async def recording(store: QdrantStore) -> RecordingEmbedder:
    recorder = RecordingEmbedder()
    await index_rules(store, recorder, TOPIC_RULES)
    return recorder


def sections(output: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for block in output.split("\n\n"):
        [heading, *lines] = block.splitlines()
        grouped[heading.removeprefix("Topic: ")] = lines
    return grouped


def listed_ids(lines: list[str]) -> list[str]:
    return [line.removeprefix("- ").split(" ", 1)[0] for line in lines]


async def test_get_rules_groups_several_topics_in_one_call(
    github: FakeGitHub, store: QdrantStore, recording: RecordingEmbedder, bundle: PullRequestBundle
) -> None:
    get_rules_tool = tools_for(github, store, recording, bundle)["get_rules"]

    output = await call(get_rules_tool, topics=["SqlCommand", "Thread Sleep"])

    grouped = sections(output)
    assert list(grouped) == ["SqlCommand", "Thread Sleep"]
    assert recording.queries == ["SqlCommand", "Thread Sleep"]
    assert all(1 <= len(lines) <= 3 for lines in grouped.values())
    assert listed_ids(grouped["SqlCommand"])[0] == "TST-01"
    assert listed_ids(grouped["Thread Sleep"])[0] == "TST-02"


async def test_get_rules_describes_a_rule_found_by_two_topics_once(
    github: FakeGitHub, store: QdrantStore, recording: RecordingEmbedder, bundle: PullRequestBundle
) -> None:
    get_rules_tool = tools_for(github, store, recording, bundle)["get_rules"]

    output = await call(get_rules_tool, topics=["SqlCommand", "SqlCommand text"])

    grouped = sections(output)
    first, second = grouped["SqlCommand"], grouped["SqlCommand text"]
    assert "TST-01" in listed_ids(first) and "TST-01" in listed_ids(second)
    assert "- TST-01 (listed above)" in second
    assert output.count("TST-01 (bug, medium): SqlCommand concatenation.") == 1
    # Every rule is described in full exactly once across all topics
    described = [line for line in output.splitlines() if "(bug, medium)" in line]
    assert len(described) == len(set(listed_ids(first + second)))


async def test_get_rules_searches_at_most_six_topics(
    github: FakeGitHub, store: QdrantStore, recording: RecordingEmbedder, bundle: PullRequestBundle
) -> None:
    get_rules_tool = tools_for(github, store, recording, bundle)["get_rules"]
    topics = [f"topic{n}" for n in range(1, 9)]

    output = await call(get_rules_tool, topics=topics)

    assert list(sections(output)) == topics[:6]
    assert recording.queries == topics[:6]


async def test_get_rules_ignores_blank_topics_and_strips_the_rest(
    github: FakeGitHub, store: QdrantStore, recording: RecordingEmbedder, bundle: PullRequestBundle
) -> None:
    get_rules_tool = tools_for(github, store, recording, bundle)["get_rules"]
    topics = ["", "   ", "  Thread Sleep  ", "\t", "a", "b", "c", "d", "e", "f"]

    output = await call(get_rules_tool, topics=topics)

    # Blank topics do not use up the six-topic budget
    expected = ["Thread Sleep", "a", "b", "c", "d", "e"]
    assert list(sections(output)) == expected
    assert recording.queries == expected


@pytest.mark.parametrize(
    "topics",
    [pytest.param([], id="empty"), pytest.param(["", "  ", "\n"], id="only-blank")],
)
async def test_get_rules_asks_for_a_topic_without_searching(
    github: FakeGitHub,
    store: QdrantStore,
    recording: RecordingEmbedder,
    bundle: PullRequestBundle,
    topics: list[str],
) -> None:
    get_rules_tool = tools_for(github, store, recording, bundle)["get_rules"]

    assert await call(get_rules_tool, topics=topics) == "Give at least one topic."
    assert recording.queries == []


async def test_get_rules_input_schema_is_a_list_of_strings(
    github: FakeGitHub, store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    get_rules_tool = tools_for(github, store, embedder, bundle)["get_rules"]

    schema = get_rules_tool.get_input_jsonschema()

    assert list(schema["properties"]) == ["topics"]
    assert schema["properties"]["topics"]["type"] == "array"
    assert schema["properties"]["topics"]["items"] == {"type": "string"}
    assert schema["required"] == ["topics"]


def numbered_file(count: int) -> str:
    return "".join(f"line{n}\n" for n in range(1, count + 1))


async def test_read_file_numbers_the_requested_lines_from_head_sha(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: numbered_file(10)})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = await call(read, path=FILE_PATH, start_line=2, end_line=3)

    assert output == "     2  line2\n     3  line3"
    assert github.content_reads == [(FILE_PATH, HEAD_SHA)]


async def test_read_file_defaults_to_the_first_120_lines(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: numbered_file(150)})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    lines = (await call(read, path=FILE_PATH)).splitlines()

    assert (lines[0], lines[-1], len(lines)) == ("     1  line1", "   120  line120", 120)


@pytest.mark.parametrize(
    ("start_line", "first", "last"),
    [pytest.param(1, 1, 200, id="from-start"), pytest.param(5, 5, 204, id="from-middle")],
)
async def test_read_file_caps_the_range_at_200_lines(
    store: QdrantStore,
    embedder: FakeEmbedder,
    bundle: PullRequestBundle,
    start_line: int,
    first: int,
    last: int,
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: numbered_file(300)})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    lines = (await call(read, path=FILE_PATH, start_line=start_line, end_line=999)).splitlines()

    assert len(lines) == 200
    assert lines[0].split() == [str(first), f"line{first}"]
    assert lines[-1].split() == [str(last), f"line{last}"]


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [
        pytest.param(0, 5, id="zero-start"),
        pytest.param(-1, 5, id="negative-start"),
        pytest.param(5, 4, id="end-before-start"),
    ],
)
async def test_read_file_rejects_invalid_ranges_without_reading(
    store: QdrantStore,
    embedder: FakeEmbedder,
    bundle: PullRequestBundle,
    start_line: int,
    end_line: int,
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: numbered_file(10)})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = await call(read, path=FILE_PATH, start_line=start_line, end_line=end_line)

    assert output == "Invalid line range."
    assert github.content_reads == []


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(PrNotFoundError("gone"), id="not-found"),
        pytest.param(RateLimitError(None), id="rate-limited"),
        pytest.param(GitHubApiError("boom", status_code=500), id="api-error"),
    ],
)
async def test_read_file_returns_a_message_on_github_error(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle, failure: GitHubError
) -> None:
    github = FakeGitHub(failures={FILE_PATH: failure})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = await call(read, path=FILE_PATH)

    assert output == f"Could not read {FILE_PATH}: {failure}"


async def test_read_file_output_is_clipped(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: "x" * 500 + "\n" + ("y" * 500 + "\n") * 20})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = await call(read, path=FILE_PATH)

    assert output.endswith(TRUNCATED)
    assert len(output) == MAX_TOOL_TEXT + len(TRUNCATED)


async def test_short_output_is_not_clipped(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: numbered_file(2)})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    assert TRUNCATED not in await call(read, path=FILE_PATH)


@pytest.mark.parametrize(
    "separator",
    [pytest.param("\x0c", id="form-feed"), pytest.param("\N{LINE SEPARATOR}", id="line-separator")],
)
async def test_read_file_numbers_are_not_shifted_by_a_unicode_line_break(
    store: QdrantStore, embedder: FakeEmbedder, bundle: PullRequestBundle, separator: str
) -> None:
    github = FakeGitHub(head_files={FILE_PATH: f"// a{separator}b\nsecond\n"})
    read = tools_for(github, store, embedder, bundle)["read_file"]

    output = await call(read, path=FILE_PATH, start_line=2, end_line=2)

    assert output == "     2  second"
