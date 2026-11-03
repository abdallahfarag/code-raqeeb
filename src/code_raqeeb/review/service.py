import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from code_raqeeb.config import Settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import GitHubError
from code_raqeeb.github.models import PrRef
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.guardrails.injection import InjectionScanner
from code_raqeeb.rag.embeddings import FastEmbedEmbedder
from code_raqeeb.rag.indexer import IndexReport, RepoTooLargeError
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.context import ReviewContext
from code_raqeeb.review.graph import ReviewGraph, build_review_graph, run_review, stream_review
from code_raqeeb.review.llm import build_chat_model
from code_raqeeb.review.models import ReviewComment
from code_raqeeb.review.reviewer import FileReviewer
from code_raqeeb.review.tracing import init_langfuse, tracing_callbacks

logger = logging.getLogger(__name__)

EXPECTED_ERRORS = (GitHubError, RepoTooLargeError)


class ReviewEvent(BaseModel):
    name: str
    data: dict[str, Any]


class ReviewResult(BaseModel):
    pr_url: str
    head_sha: str
    summary: str
    comments: list[ReviewComment]
    file_errors: list[str]
    dropped_comments: int


class ReviewService:
    def __init__(
        self,
        *,
        graph: ReviewGraph,
        context: ReviewContext,
        qdrant: AsyncQdrantClient,
        langfuse: Langfuse | None,
    ) -> None:
        self._graph = graph
        self._context = context
        self._qdrant = qdrant
        self._langfuse = langfuse

    async def run(self, pr_url: str) -> ReviewResult:
        state = await run_review(
            self._graph,
            pr_url,
            self._context,
            max_concurrency=self._context.settings.review_concurrency,
            callbacks=tracing_callbacks(self._langfuse),
            tags=_tags(pr_url),
        )
        bundle = state.get("bundle")
        return ReviewResult(
            pr_url=pr_url,
            head_sha=bundle.head_sha if bundle else "",
            summary=state.get("summary", ""),
            comments=state.get("verified_comments", []),
            file_errors=state["file_errors"],
            dropped_comments=state.get("dropped_comments", 0),
        )

    async def stream(self, pr_url: str) -> AsyncIterator[ReviewEvent]:
        yield ReviewEvent(name="accepted", data={"pr_url": pr_url})
        head_sha = ""
        comments: list[ReviewComment] = []
        file_errors: list[str] = []
        dropped = 0
        try:
            async for node, update in stream_review(
                self._graph,
                pr_url,
                self._context,
                max_concurrency=self._context.settings.review_concurrency,
                callbacks=tracing_callbacks(self._langfuse),
                tags=_tags(pr_url),
            ):
                if node == "load_pr":
                    bundle: PullRequestBundle = update["bundle"]
                    head_sha = bundle.head_sha
                    yield ReviewEvent(
                        name="pr_loaded",
                        data={
                            "title": bundle.pull_request.title,
                            "head_sha": head_sha,
                            "files": [file.path for file in bundle.files],
                            "skipped": len(bundle.skipped),
                        },
                    )
                elif node == "index_base":
                    report: IndexReport = update["index_report"]
                    yield ReviewEvent(name="indexed", data=report.model_dump())
                elif node == "review_file":
                    errors: list[str] = update.get("file_errors", [])
                    file_errors.extend(errors)
                    reviewed: list[str] = update.get("reviewed_files", [])
                    yield ReviewEvent(
                        name="file_reviewed",
                        data={
                            "file": reviewed[0] if reviewed else "",
                            "comments": len(update.get("comments", [])),
                            "errors": errors,
                        },
                    )
                elif node == "verify":
                    comments = update["verified_comments"]
                    dropped = update["dropped_comments"]
                    yield ReviewEvent(
                        name="verified", data={"kept": len(comments), "dropped": dropped}
                    )
                elif node == "summarize":
                    result = ReviewResult(
                        pr_url=pr_url,
                        head_sha=head_sha,
                        summary=update["summary"],
                        comments=comments,
                        file_errors=file_errors,
                        dropped_comments=dropped,
                    )
                    yield ReviewEvent(name="done", data=result.model_dump(mode="json"))
        except EXPECTED_ERRORS as exc:
            yield ReviewEvent(name="error", data={"type": type(exc).__name__, "message": str(exc)})
        except Exception:
            logger.exception("Review of %s failed", pr_url)
            yield ReviewEvent(
                name="error",
                data={"type": "InternalError", "message": "The review failed unexpectedly."},
            )

    async def aclose(self) -> None:
        await self._qdrant.close()
        if self._langfuse is not None:
            await asyncio.to_thread(self._langfuse.shutdown)


def _tags(pr_url: str) -> list[str]:
    ref = PrRef.from_url(pr_url)
    return [ref.full_name, f"pr-{ref.number}"]


async def build_review_service(settings: Settings, http: httpx.AsyncClient) -> ReviewService:
    embedder = await asyncio.to_thread(
        FastEmbedEmbedder, settings.dense_embedding_model, settings.sparse_embedding_model
    )
    qdrant = AsyncQdrantClient(url=str(settings.qdrant_url))
    store = QdrantStore(qdrant, settings.qdrant_code_collection, settings.qdrant_rules_collection)
    await store.ensure_collections(embedder.dense_size)

    injection_scanner: InjectionScanner | None = None
    if settings.ml_injection_scanner:
        from code_raqeeb.guardrails.ml_injection import GorgetInjectionScanner

        injection_scanner = await asyncio.to_thread(
            GorgetInjectionScanner, settings.injection_threshold
        )

    context = ReviewContext(
        settings=settings,
        github=GitHubClient(http, settings),
        store=store,
        embedder=embedder,
        reviewer=FileReviewer(build_chat_model(settings), max_tool_rounds=settings.max_tool_rounds),
        injection_scanner=injection_scanner,
    )
    return ReviewService(
        graph=build_review_graph(InMemorySaver()),
        context=context,
        qdrant=qdrant,
        langfuse=init_langfuse(settings),
    )
