import asyncio
import sys

import httpx

from code_raqeeb.config import get_settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import GitHubError
from code_raqeeb.github.service import load_pull_request

USAGE = "Usage: uv run python scripts/inspect_pr.py https://github.com/OWNER/REPO/pull/NUMBER"


async def main(url: str) -> int:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
        client = GitHubClient(http, settings)
        try:
            bundle = await load_pull_request(client, url, settings)
        except GitHubError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    pr = bundle.pull_request
    print(f"{pr.title}  (#{pr.number}, head {bundle.head_sha[:7]})")
    for file in bundle.files:
        added = sorted(file.added_lines)
        print(f"  REVIEW {file.path}: {len(file.hunks)} hunk(s), added lines {added}")
    for skipped in bundle.skipped:
        print(f"  SKIP   {skipped.path}: {skipped.reason}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(USAGE)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
