# pyright: reportUnknownMemberType=false

import uuid
from typing import cast

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


async def run_review(
    graph: ReviewGraph, pr_url: str, context: ReviewContext, *, max_concurrency: int
) -> ReviewState:
    result = await graph.ainvoke(
        ReviewState(pr_url=pr_url, comments=[], file_errors=[]),
        config={
            "configurable": {"thread_id": str(uuid.uuid4())},
            "max_concurrency": max_concurrency,
        },
        context=context,
    )
    return cast(ReviewState, result)
