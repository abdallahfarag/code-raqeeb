import re
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError
from pydantic import TypeAdapter

from code_raqeeb.github.errors import PrNotFoundError
from code_raqeeb.guardrails.input import MAX_URL_LENGTH
from code_raqeeb.mcp_server import dependencies
from code_raqeeb.mcp_server.app import create_server
from code_raqeeb.rag.indexer import RepoTooLargeError
from code_raqeeb.rag.models import CodeChunk, CodeHit, RuleHit
from code_raqeeb.review.models import ReviewComment
from code_raqeeb.review.service import ReviewEvent, ReviewResult
from code_raqeeb.rules.models import ReviewRule

type McpClient = Client[FastMCPTransport]

PR_URL = "https://github.com/octo/hello/pull/7"
PULL_PREFIX = "https://github.com/octo/hello/pull/"
FILES = ["a.cs", "b.cs", "c.cs"]

RESULT = ReviewResult(
    pr_url=PR_URL,
    head_sha="head111",
    summary="1 comment(s): 1 high, 0 medium, 0 low",
    comments=[
        ReviewComment(
            path="a.cs",
            line=3,
            severity="high",
            category="bug",
            rule_id="ASY-01",
            message="Charge is not awaited",
            suggestion="await the call",
            permalink="https://github.com/octo/hello/blob/head111/a.cs#L3",
        )
    ],
    file_errors=[],
    dropped_comments=2,
)
PR_LOADED = ReviewEvent(name="pr_loaded", data={"title": "Pay endpoint", "files": FILES})
DONE = ReviewEvent(name="done", data=RESULT.model_dump(mode="json"))
ERROR = ReviewEvent(name="error", data={"type": "PrNotFoundError", "message": "Not found"})

CODE_HITS = [
    CodeHit(
        chunk=CodeChunk(
            repo="octo/hello",
            sha="base000",
            path="a.cs",
            symbol="A.Run",
            kind="method",
            start_line=1,
            end_line=3,
            text="void Run() { }",
        ),
        score=0.9,
    )
]
RULE_HITS = [
    RuleHit(
        rule=ReviewRule(
            id="ASY-01",
            category="bug",
            severity="high",
            title="Await tasks",
            description="Do not drop a task.",
        ),
        score=0.8,
    )
]

INVALID_URLS = [
    pytest.param("not a url", id="not-a-url"),
    pytest.param("https://gitlab.com/octo/hello/pull/7", id="other-host"),
    pytest.param(PULL_PREFIX + "1" * (MAX_URL_LENGTH + 1 - len(PULL_PREFIX)), id="too-long"),
]
TOOLS_TAKING_A_URL = [
    pytest.param("review_pr", {}, id="review_pr"),
    pytest.param("search_code", {"query": "charge"}, id="search_code"),
]
TOOLS_TAKING_A_LIMIT = [
    pytest.param("search_code", {"pr_url": PR_URL, "query": "charge"}, id="search_code"),
    pytest.param("get_rules", {"query": "async"}, id="get_rules"),
]


class FakeReviewService:
    def __init__(self) -> None:
        self.events: list[ReviewEvent] = [
            ReviewEvent(name="accepted", data={"pr_url": PR_URL}),
            PR_LOADED,
            *[ReviewEvent(name="file_reviewed", data={"file": file}) for file in FILES],
            ReviewEvent(name="verified", data={"kept": 1, "dropped": 2}),
            DONE,
        ]
        self.search_error: Exception | None = None
        self.calls: list[tuple[object, ...]] = []
        self.lookups = 0

    async def stream(self, pr_url: str) -> AsyncIterator[ReviewEvent]:
        self.calls.append(("stream", pr_url))
        for event in self.events:
            yield event

    async def search_code(self, pr_url: str, query: str, limit: int) -> list[CodeHit]:
        self.calls.append(("search_code", pr_url, query, limit))
        if self.search_error is not None:
            raise self.search_error
        return CODE_HITS

    async def find_rules(self, query: str, limit: int) -> list[RuleHit]:
        self.calls.append(("find_rules", query, limit))
        return RULE_HITS


def result_list(structured: dict[str, Any] | None) -> Any:
    assert structured is not None
    return structured["result"]


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> FakeReviewService:
    fake = FakeReviewService()

    async def get_service() -> FakeReviewService:
        fake.lookups += 1
        return fake

    monkeypatch.setattr(dependencies, "get_service", get_service)
    return fake


@pytest.fixture
async def mcp(service: FakeReviewService) -> AsyncIterator[McpClient]:
    async with Client(create_server()) as client:
        yield client


async def schemas(mcp: McpClient) -> dict[str, dict[str, Any]]:
    return {tool.name: tool.input_schema for tool in await mcp.list_tools()}


async def test_list_tools_returns_exactly_the_three_tools(mcp: McpClient) -> None:
    tools = await mcp.list_tools()

    assert sorted(tool.name for tool in tools) == ["get_rules", "review_pr", "search_code"]


async def test_every_tool_is_read_only(mcp: McpClient) -> None:
    for tool in await mcp.list_tools():
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True


async def test_every_tool_has_a_description(mcp: McpClient) -> None:
    for tool in await mcp.list_tools():
        assert tool.description


async def test_review_pr_schema_takes_only_the_url(mcp: McpClient) -> None:
    schema = (await schemas(mcp))["review_pr"]

    assert list(schema["properties"]) == ["pr_url"]
    assert schema["properties"]["pr_url"]["type"] == "string"
    assert schema["required"] == ["pr_url"]


async def test_search_code_schema_bounds_the_limit(mcp: McpClient) -> None:
    schema = (await schemas(mcp))["search_code"]

    assert sorted(schema["properties"]) == ["limit", "pr_url", "query"]
    assert schema["properties"]["pr_url"]["type"] == "string"
    limit = schema["properties"]["limit"]
    assert (limit["type"], limit["minimum"], limit["maximum"]) == ("integer", 1, 10)
    assert sorted(schema["required"]) == ["pr_url", "query"]


async def test_get_rules_schema_bounds_the_limit(mcp: McpClient) -> None:
    schema = (await schemas(mcp))["get_rules"]

    assert sorted(schema["properties"]) == ["limit", "query"]
    limit = schema["properties"]["limit"]
    assert (limit["type"], limit["minimum"], limit["maximum"]) == ("integer", 1, 10)
    assert schema["required"] == ["query"]


async def test_no_schema_exposes_the_context_parameter(mcp: McpClient) -> None:
    for schema in (await schemas(mcp)).values():
        assert "ctx" not in schema["properties"]


async def test_review_pr_returns_structured_data_that_validates_as_review_result(
    mcp: McpClient, service: FakeReviewService
) -> None:
    result = await mcp.call_tool("review_pr", {"pr_url": PR_URL})

    assert ReviewResult.model_validate(result.structured_content) == RESULT
    assert service.calls == [("stream", PR_URL)]


async def test_review_pr_passes_the_stripped_url_to_the_service(
    mcp: McpClient, service: FakeReviewService
) -> None:
    await mcp.call_tool("review_pr", {"pr_url": f"  {PR_URL}\n"})

    assert service.calls == [("stream", PR_URL)]


async def test_review_pr_reports_progress_once_per_reviewed_file(mcp: McpClient) -> None:
    reports: list[tuple[float, float | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total))

    await mcp.call_tool("review_pr", {"pr_url": PR_URL}, progress_handler=on_progress)

    assert reports == [(1, 3), (2, 3), (3, 3)]


async def test_review_pr_reports_no_progress_when_no_file_is_reviewed(
    mcp: McpClient, service: FakeReviewService
) -> None:
    service.events = [ReviewEvent(name="pr_loaded", data={"title": "Docs", "files": []}), DONE]
    reports: list[float] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append(progress)

    await mcp.call_tool("review_pr", {"pr_url": PR_URL}, progress_handler=on_progress)

    assert reports == []


@pytest.mark.parametrize("url", INVALID_URLS)
@pytest.mark.parametrize(("tool", "arguments"), TOOLS_TAKING_A_URL)
async def test_invalid_url_raises_tool_error_and_the_service_is_never_called(
    mcp: McpClient,
    service: FakeReviewService,
    tool: str,
    arguments: dict[str, Any],
    url: str,
) -> None:
    with pytest.raises(ToolError):
        await mcp.call_tool(tool, {"pr_url": url, **arguments})

    assert service.calls == []


@pytest.mark.parametrize(("tool", "arguments"), TOOLS_TAKING_A_URL)
async def test_invalid_url_does_not_build_the_service(
    mcp: McpClient, service: FakeReviewService, tool: str, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ToolError):
        await mcp.call_tool(tool, {"pr_url": "not a url", **arguments})

    assert service.lookups == 0


async def test_error_event_becomes_a_tool_error_with_its_type_and_message(
    mcp: McpClient, service: FakeReviewService
) -> None:
    service.events = [PR_LOADED, ERROR]

    with pytest.raises(ToolError) as caught:
        await mcp.call_tool("review_pr", {"pr_url": PR_URL})

    assert str(caught.value) == "PrNotFoundError: Not found"


async def test_stream_that_ends_without_done_raises_tool_error(
    mcp: McpClient, service: FakeReviewService
) -> None:
    service.events = [PR_LOADED]

    with pytest.raises(ToolError, match="ended without a result"):
        await mcp.call_tool("review_pr", {"pr_url": PR_URL})


async def test_search_code_returns_the_fake_results(
    mcp: McpClient, service: FakeReviewService
) -> None:
    result = await mcp.call_tool("search_code", {"pr_url": PR_URL, "query": "charge", "limit": 3})

    hits = TypeAdapter(list[CodeHit]).validate_python(result_list(result.structured_content))
    assert hits == CODE_HITS
    assert service.calls == [("search_code", PR_URL, "charge", 3)]


async def test_get_rules_returns_the_fake_results(
    mcp: McpClient, service: FakeReviewService
) -> None:
    result = await mcp.call_tool("get_rules", {"query": "async", "limit": 3})

    hits = TypeAdapter(list[RuleHit]).validate_python(result_list(result.structured_content))
    assert hits == RULE_HITS
    assert service.calls == [("find_rules", "async", 3)]


async def test_search_code_uses_the_default_limit_and_stripped_url(
    mcp: McpClient, service: FakeReviewService
) -> None:
    await mcp.call_tool("search_code", {"pr_url": f" {PR_URL} ", "query": "charge"})

    assert service.calls == [("search_code", PR_URL, "charge", 5)]


async def test_get_rules_uses_the_default_limit(mcp: McpClient, service: FakeReviewService) -> None:
    await mcp.call_tool("get_rules", {"query": "async"})

    assert service.calls == [("find_rules", "async", 5)]


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(PrNotFoundError("Pull request not found"), id="github-error"),
        pytest.param(RepoTooLargeError("Too many C# files"), id="repo-too-large"),
    ],
)
async def test_search_code_turns_expected_errors_into_tool_errors(
    mcp: McpClient, service: FakeReviewService, error: Exception
) -> None:
    service.search_error = error

    with pytest.raises(ToolError) as caught:
        await mcp.call_tool("search_code", {"pr_url": PR_URL, "query": "charge"})

    assert str(caught.value) == str(error)


@pytest.mark.parametrize("limit", [0, 11])
@pytest.mark.parametrize(("tool", "arguments"), TOOLS_TAKING_A_LIMIT)
async def test_limit_out_of_range_is_rejected_without_calling_the_service(
    mcp: McpClient, service: FakeReviewService, tool: str, arguments: dict[str, Any], limit: int
) -> None:
    with pytest.raises(ToolError):
        await mcp.call_tool(tool, {**arguments, "limit": limit})

    assert service.calls == []


@pytest.mark.parametrize("limit", [1, 10])
@pytest.mark.parametrize(("tool", "arguments"), TOOLS_TAKING_A_LIMIT)
async def test_limit_at_the_bounds_is_accepted(
    mcp: McpClient, service: FakeReviewService, tool: str, arguments: dict[str, Any], limit: int
) -> None:
    await mcp.call_tool(tool, {**arguments, "limit": limit})

    assert service.calls[0][-1] == limit


@pytest.mark.parametrize("query", ["x", "x" * 201])
@pytest.mark.parametrize(("tool", "arguments"), TOOLS_TAKING_A_LIMIT)
async def test_query_with_a_bad_length_is_rejected(
    mcp: McpClient, service: FakeReviewService, tool: str, arguments: dict[str, Any], query: str
) -> None:
    with pytest.raises(ToolError):
        await mcp.call_tool(tool, {**arguments, "query": query})

    assert service.calls == []


async def test_nothing_is_written_to_stdout_during_calls(
    mcp: McpClient, service: FakeReviewService, capsys: pytest.CaptureFixture[str]
) -> None:
    await mcp.call_tool("review_pr", {"pr_url": PR_URL})
    await mcp.call_tool("search_code", {"pr_url": PR_URL, "query": "charge"})
    await mcp.call_tool("get_rules", {"query": "async"})
    service.events = [ERROR]
    with pytest.raises(ToolError):
        await mcp.call_tool("review_pr", {"pr_url": PR_URL})

    assert capsys.readouterr().out == ""


def schema_descriptions(node: object) -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in cast(dict[str, object], node).items():
            if key == "description" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(schema_descriptions(value))
    elif isinstance(node, list):
        for item in cast(list[object], node):
            found.extend(schema_descriptions(item))
    return found


async def test_no_tool_or_parameter_description_says_public(mcp: McpClient) -> None:
    descriptions: list[str] = []
    for tool in await mcp.list_tools():
        assert tool.description is not None
        descriptions.append(tool.description)
        descriptions.extend(schema_descriptions(tool.input_schema))

    assert any("pull request URL" in text for text in descriptions)
    assert [text for text in descriptions if re.search(r"\bpublic\b", text, re.I)] == []


def test_server_instructions_do_not_say_public() -> None:
    instructions = create_server().instructions

    assert instructions is not None
    assert re.search(r"\bpublic\b", instructions, re.I) is None
