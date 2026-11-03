import asyncio
import sys

import httpx

from code_raqeeb.config import get_settings
from code_raqeeb.github.errors import GitHubError
from code_raqeeb.rag.indexer import RepoTooLargeError
from code_raqeeb.review.service import ReviewResult, build_review_service

USAGE = "Usage: uv run python scripts/review_pr.py <PR URL>"


async def main(url: str) -> int:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
        service = await build_review_service(settings, http)
        try:
            result = await service.run(url)
        except (GitHubError, RepoTooLargeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            await service.aclose()
    print_result(result)
    return 0


def print_result(result: ReviewResult) -> None:
    print(result.summary, "\n")
    for comment in result.comments:
        rule = f" [{comment.rule_id}]" if comment.rule_id else ""
        print(f"{comment.severity.upper():<6} {comment.path}:{comment.line}{rule}")
        print(f"{comment.message}")
        if comment.suggestion:
            print(f"Suggestion: {comment.suggestion}")
        print(f"{comment.permalink}\n")
    for error in result.file_errors:
        print(f"ERROR  {error}")
    if result.dropped_comments:
        print(f"(dropped {result.dropped_comments} comment(s) pointing outside the diff)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(USAGE)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
