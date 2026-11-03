import asyncio
from collections import Counter
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Send

from code_raqeeb.github.service import PullRequestBundle, load_pull_request
from code_raqeeb.guardrails.findings import scan_added_lines
from code_raqeeb.guardrails.output import check_comments
from code_raqeeb.rag.indexer import index_rules, index_snapshot
from code_raqeeb.review.context import ReviewContext
from code_raqeeb.review.models import to_comment
from code_raqeeb.review.state import FileTask, ReviewState
from code_raqeeb.review.tools import build_tools
from code_raqeeb.rules.loader import get_rules


def _bundle(state: ReviewState) -> PullRequestBundle:
    bundle = state.get("bundle")
    if bundle is None:
        raise RuntimeError("load_pr must run before this node")
    return bundle


async def load_pr(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    ctx = runtime.context
    bundle = await load_pull_request(ctx.github, state["pr_url"], ctx.settings)
    return {"bundle": bundle}


async def index_base(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    ctx = runtime.context
    bundle = _bundle(state)
    report = await index_snapshot(
        ctx.github,
        ctx.store,
        ctx.embedder,
        bundle.ref,
        bundle.base_sha,
        max_files=ctx.settings.max_index_files,
        max_chunk_lines=ctx.settings.max_chunk_lines,
    )
    await index_rules(ctx.store, ctx.embedder, get_rules())
    return {"index_report": report}


def fan_out(state: ReviewState) -> list[Send] | Literal["verify"]:
    bundle = _bundle(state)
    if not bundle.files:
        return "verify"
    return [Send("review_file", FileTask(bundle=bundle, file=file)) for file in bundle.files]


async def review_file(state: FileTask, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    ctx = runtime.context
    bundle, file = state["bundle"], state["file"]
    repo, head_sha = bundle.ref.full_name, bundle.head_sha
    findings = await asyncio.to_thread(
        scan_added_lines,
        file,
        repo=repo,
        head_sha=head_sha,
        injection_scanner=ctx.injection_scanner,
    )
    try:
        drafts = await ctx.reviewer.review(bundle, file, build_tools(ctx, bundle))
    except Exception as exc:
        return {
            "comments": findings,
            "file_errors": [f"{file.path}: {type(exc).__name__}: {exc}"],
            "reviewed_files": [file.path],
        }
    comments = [
        to_comment(draft, repo=bundle.ref.full_name, head_sha=bundle.head_sha, path=file.path)
        for draft in drafts
    ]
    return {"comments": findings + comments, "reviewed_files": [file.path]}


def verify(state: ReviewState, runtime: Runtime[ReviewContext]) -> dict[str, Any]:
    settings = runtime.context.settings
    check = check_comments(
        state["comments"],
        files={file.path: file for file in _bundle(state).files},
        rules=get_rules(),
        max_per_file=settings.max_comments_per_file,
        max_chars=settings.max_comment_chars,
    )
    return {"verified_comments": list(check.kept), "dropped_comments": len(check.dropped)}


def summarize(state: ReviewState) -> dict[str, Any]:
    comments = state.get("verified_comments", [])
    by_severity = Counter(comment.severity for comment in comments)
    summary = (
        f"{len(comments)} comment(s): {by_severity['high']} high, "
        f"{by_severity['medium']} medium, {by_severity['low']} low"
    )
    if state["file_errors"]:
        summary += f"; {len(state['file_errors'])} file(s) could not be reviewed"
    return {"summary": summary}
