import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.review.models import DraftComment, FileReview
from code_raqeeb.review.prompts import SYSTEM_PROMPT
from code_raqeeb.review.reviewer import FileReviewer
from tests.review.fakes import ScriptedChatModel

REVIEW = FileReview(
    comments=[DraftComment(line=27, severity="high", category="bug", message="Blocks on async")]
)


def asks_for(name: str = "lookup", call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"query": "q"}, "id": call_id}])


class Recorder:
    def __init__(self) -> None:
        self.queries: list[str] = []

        @tool(description="Look something up.")
        async def lookup(query: str) -> str:
            self.queries.append(query)
            return f"result for {query}"

        self.tool: BaseTool = lookup


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()


async def test_one_tool_round_drops_text_reply_and_ends_with_tool_message(
    bundle: PullRequestBundle, recorder: Recorder
) -> None:
    model = ScriptedChatModel(replies=[asks_for(), AIMessage(content="final text")], review=REVIEW)
    reviewer = FileReviewer(model, max_tool_rounds=2)

    comments = await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    assert comments == REVIEW.comments
    assert recorder.queries == ["q"]
    [structured] = model.structured_calls
    last = structured[-1]
    assert isinstance(last, ToolMessage)
    assert last.content == "result for q"
    assert last.tool_call_id == "call-1"
    assert not any(message.content == "final text" for message in structured)
    assert len(model.calls) == 3


async def test_text_reply_without_tools_is_dropped_before_structured_call(
    bundle: PullRequestBundle, recorder: Recorder
) -> None:
    model = ScriptedChatModel(replies=[AIMessage(content="looks fine")], review=REVIEW)
    reviewer = FileReviewer(model, max_tool_rounds=2)

    await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    [structured] = model.structured_calls
    assert [type(message) for message in structured] == [SystemMessage, HumanMessage]
    assert recorder.queries == []


@pytest.mark.parametrize("max_tool_rounds", [1, 2, 3])
async def test_round_cap_stops_a_model_that_always_asks_for_tools(
    bundle: PullRequestBundle, recorder: Recorder, max_tool_rounds: int
) -> None:
    replies = [asks_for(call_id=f"call-{n}") for n in range(10)]
    model = ScriptedChatModel(replies=replies, review=REVIEW)
    reviewer = FileReviewer(model, max_tool_rounds=max_tool_rounds)

    comments = await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    assert comments == REVIEW.comments
    assert len(recorder.queries) == max_tool_rounds
    assert len(model.calls) == max_tool_rounds + 1
    assert len(model.structured_calls) == 1
    assert isinstance(model.structured_calls[0][-1], ToolMessage)


async def test_unknown_tool_name_produces_a_tool_message(
    bundle: PullRequestBundle, recorder: Recorder
) -> None:
    model = ScriptedChatModel(replies=[asks_for(name="missing"), AIMessage(content="done")])
    reviewer = FileReviewer(model, max_tool_rounds=2)

    await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    last = model.structured_calls[0][-1]
    assert isinstance(last, ToolMessage)
    assert last.content == "Unknown tool: missing"
    assert recorder.queries == []


async def test_every_tool_call_in_a_reply_gets_its_own_tool_message(
    bundle: PullRequestBundle, recorder: Recorder
) -> None:
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "lookup", "args": {"query": "one"}, "id": "call-1"},
            {"name": "lookup", "args": {"query": "two"}, "id": "call-2"},
        ],
    )
    model = ScriptedChatModel(replies=[both, AIMessage(content="done")])
    reviewer = FileReviewer(model, max_tool_rounds=2)

    await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    tool_messages = [m for m in model.structured_calls[0] if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in tool_messages] == [
        ("call-1", "result for one"),
        ("call-2", "result for two"),
    ]


async def test_zero_rounds_makes_exactly_one_structured_call(
    bundle: PullRequestBundle, recorder: Recorder
) -> None:
    model = ScriptedChatModel(replies=[asks_for()], review=REVIEW)
    reviewer = FileReviewer(model, max_tool_rounds=0)

    comments = await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    assert comments == REVIEW.comments
    assert len(model.calls) == 1
    assert len(model.structured_calls) == 1
    assert model.bound_tools == []
    assert recorder.queries == []


async def test_no_tools_skips_the_tool_rounds(bundle: PullRequestBundle) -> None:
    model = ScriptedChatModel(replies=[], review=REVIEW)
    reviewer = FileReviewer(model, max_tool_rounds=2)

    await reviewer.review(bundle, bundle.files[0], [])

    assert len(model.calls) == 1
    assert model.bound_tools == []


async def test_tools_are_bound_by_name(bundle: PullRequestBundle, recorder: Recorder) -> None:
    model = ScriptedChatModel(replies=[AIMessage(content="done")])
    reviewer = FileReviewer(model, max_tool_rounds=1)

    await reviewer.review(bundle, bundle.files[0], [recorder.tool])

    assert model.bound_tools == [["lookup"]]


async def test_prompt_holds_system_rules_and_the_numbered_diff(bundle: PullRequestBundle) -> None:
    controller = next(file for file in bundle.files if file.path.endswith("OrdersController.cs"))
    model = ScriptedChatModel(replies=[])

    await FileReviewer(model, max_tool_rounds=0).review(bundle, controller, [])

    [system, human] = model.structured_calls[0]
    assert system.content == SYSTEM_PROMPT
    assert isinstance(human.content, str)
    assert f"File: {controller.path} (modified)" in human.content
    assert "Pull request: Add pay endpoint and payment receipts" in human.content
    assert "    27 +         var paid = orders.PayAsync" in human.content
