import asyncio
import json
import sys
from pathlib import Path

import httpx
from pydantic import BaseModel

from code_raqeeb.config import get_settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.errors import GitHubError
from code_raqeeb.github.models import PrRef

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "prs"


class _ExpectedUrl(BaseModel):
    pr_url: str


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def record(client: GitHubClient, folder: Path) -> None:
    expected_text = (folder / "expected.json").read_text(encoding="utf-8")
    ref = PrRef.from_url(_ExpectedUrl.model_validate_json(expected_text).pr_url)

    pull_request = await client.get_pull_request(ref)
    files = await client.list_pull_request_files(ref)

    write_json(folder / "pr.json", pull_request.model_dump(mode="json"))
    write_json(folder / "files.json", [file.model_dump(mode="json") for file in files])
    print(f"Recorded {folder.name}: {len(files)} file(s), head {pull_request.head.sha[:7]}")


async def main() -> int:
    folders = sorted(
        folder for folder in FIXTURES_DIR.iterdir() if (folder / "expected.json").is_file()
    )
    if not folders:
        print(f"No expected.json files under {FIXTURES_DIR}", file=sys.stderr)
        return 1

    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
        client = GitHubClient(http, settings)
        try:
            await asyncio.gather(*(record(client, folder) for folder in folders))
        except GitHubError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
