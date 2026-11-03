import asyncio
import sys

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import AsyncQdrantClient

from code_raqeeb.config import get_settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import GitHubError
from code_raqeeb.guardrails.ml_injection import GorgetInjectionScanner
from code_raqeeb.rag.embeddings import FastEmbedEmbedder
from code_raqeeb.rag.indexer import RepoTooLargeError
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.context import ReviewContext
from code_raqeeb.review.graph import build_review_graph, run_review
from code_raqeeb.review.llm import build_chat_model
from code_raqeeb.review.reviewer import FileReviewer

USAGE = "Usage: uv run python scripts/review_pr.py <PR URL>"


async def main(url: str) -> int:
    settings = get_settings()
    embedder = await asyncio.to_thread(
        FastEmbedEmbedder, settings.dense_embedding_model, settings.sparse_embedding_model
    )
    qdrant = AsyncQdrantClient(url=str(settings.qdrant_url))
    store = QdrantStore(qdrant, settings.qdrant_code_collection, settings.qdrant_rules_collection)
    await store.ensure_collections(embedder.dense_size)
    reviewer = FileReviewer(build_chat_model(settings), max_tool_rounds=settings.max_tool_rounds)
    graph = build_review_graph(InMemorySaver())

    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
            context = ReviewContext(
                settings=settings,
                github=GitHubClient(http, settings),
                store=store,
                embedder=embedder,
                reviewer=reviewer,
                injection_scanner=(
                    await asyncio.to_thread(GorgetInjectionScanner, settings.injection_threshold)
                    if settings.ml_injection_scanner
                    else None
                ),
            )
            result = await run_review(
                graph, url, context, max_concurrency=settings.review_concurrency
            )
    except (GitHubError, RepoTooLargeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        await qdrant.close()

    print(result.get("summary", ""), "\n")
    for comment in result.get("verified_comments", []):
        rule = f" [{comment.rule_id}]" if comment.rule_id else ""
        print(f"{comment.severity.upper():<6} {comment.path}:{comment.line}{rule}")
        print(f"       {comment.message}")
        if comment.suggestion:
            print(f"       Suggestion: {comment.suggestion}")
        print(f"       {comment.permalink}\n")
    for error in result["file_errors"]:
        print(f"ERROR  {error}")
    dropped = result.get("dropped_comments", 0)
    if dropped:
        print(f"(dropped {dropped} comment(s) pointing outside the diff)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(USAGE)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
