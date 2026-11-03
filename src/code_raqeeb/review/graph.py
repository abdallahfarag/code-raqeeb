# pyright: reportUnknownMemberType=false

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from code_raqeeb.review.context import ReviewContext
from code_raqeeb.review.nodes import fan_out, index_base, load_pr, review_file, summarize, verify
from code_raqeeb.review.state import ReviewState

ReviewGraph = CompiledStateGraph[ReviewState, ReviewContext, ReviewState, ReviewState]


def build_review_graph(checkpointer: InMemorySaver | None = None) -> ReviewGraph:
    builder = StateGraph(ReviewState, context_schema=ReviewContext)

    builder.add_node("load_pr", load_pr)
    builder.add_node("index_base", index_base)
    builder.add_node("review_file", review_file)
    builder.add_node("verify", verify)
    builder.add_node("summarize", summarize)

    builder.add_edge(START, "load_pr")
    builder.add_edge("load_pr", "index_base")
    builder.add_conditional_edges("index_base", fan_out, ["review_file", "verify"])
    builder.add_edge("review_file", "verify")
    builder.add_edge("verify", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile(checkpointer=checkpointer)


def _initial_state(pr_url: str) -> ReviewState:
    return ReviewState(pr_url=pr_url, comments=[], file_errors=[], reviewed_files=[])


def _config(
    max_concurrency: int,
    callbacks: Sequence[BaseCallbackHandler],
    tags: Sequence[str],
) -> RunnableConfig:
    thread_id = str(uuid.uuid4())
    return {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": max_concurrency,
        "callbacks": list(callbacks),
        "metadata": {"langfuse_session_id": thread_id, "langfuse_tags": list(tags)},
    }


async def run_review(
    graph: ReviewGraph,
    pr_url: str,
    context: ReviewContext,
    *,
    max_concurrency: int,
    callbacks: Sequence[BaseCallbackHandler] = (),
    tags: Sequence[str] = (),
) -> ReviewState:
    result = await graph.ainvoke(
        _initial_state(pr_url),
        config=_config(max_concurrency, callbacks, tags),
        context=context,
    )
    return cast(ReviewState, result)


async def stream_review(
    graph: ReviewGraph,
    pr_url: str,
    context: ReviewContext,
    *,
    max_concurrency: int,
    callbacks: Sequence[BaseCallbackHandler] = (),
    tags: Sequence[str] = (),
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    async for chunk in graph.astream(
        _initial_state(pr_url),
        config=_config(max_concurrency, callbacks, tags),
        context=context,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            yield node, update if isinstance(update, dict) else {}
