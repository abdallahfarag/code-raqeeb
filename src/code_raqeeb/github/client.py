import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter

from code_raqeeb.config import Settings
from code_raqeeb.github.errors import GitHubApiError, PrNotFoundError, RateLimitError
from code_raqeeb.github.models import GitTree, PrFile, PrRef, PullRequest

_PER_PAGE = 100
_MAX_PAGES = 30  # GitHub returns at most 3,000 files per PR
_FILE_LIST = TypeAdapter(list[PrFile])

QueryParams = dict[str, str | int]


class GitHubClient:
    def __init__(self, http: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http
        self._base_url = str(settings.github_api_url).rstrip("/")
        self._semaphore = asyncio.Semaphore(settings.github_concurrency)
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "code-raqeeb",
        }
        if settings.github_token is not None:
            token = settings.github_token.get_secret_value()
            self._headers["Authorization"] = f"Bearer {token}"

    async def get_pull_request(self, ref: PrRef) -> PullRequest:
        response = await self._get(f"/repos/{ref.full_name}/pulls/{ref.number}")
        return PullRequest.model_validate(response.json())

    async def list_pull_request_files(self, ref: PrRef) -> list[PrFile]:
        files: list[PrFile] = []
        for page in range(1, _MAX_PAGES + 1):
            response = await self._get(
                f"/repos/{ref.full_name}/pulls/{ref.number}/files",
                params={"per_page": _PER_PAGE, "page": page},
            )
            batch = _FILE_LIST.validate_python(response.json())
            files.extend(batch)
            if len(batch) < _PER_PAGE:
                break
        return files

    async def get_file_content(self, ref: PrRef, path: str, sha: str) -> str:
        response = await self._get(
            f"/repos/{ref.full_name}/contents/{quote(path)}",
            params={"ref": sha},
            accept="application/vnd.github.raw+json",
        )
        return response.text

    async def get_files_content(self, ref: PrRef, paths: Iterable[str], sha: str) -> dict[str, str]:
        async def fetch(path: str) -> tuple[str, str]:
            return path, await self.get_file_content(ref, path, sha)

        results = await asyncio.gather(*(fetch(path) for path in paths))
        return dict(results)

    async def _get(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
        accept: str | None = None,
    ) -> httpx.Response:
        headers = dict(self._headers)
        if accept is not None:
            headers["Accept"] = accept
        async with self._semaphore:
            try:
                response = await self._http.get(
                    f"{self._base_url}{path}", params=params, headers=headers
                )
            except httpx.HTTPError as exc:
                raise GitHubApiError(f"Network error calling GitHub: {exc}") from exc
        _raise_for_status(response)
        return response

    async def list_tree(self, ref: PrRef, sha: str) -> list[str]:
        response = await self._get(
            f"/repos/{ref.full_name}/git/trees/{sha}", params={"recursive": 1}
        )
        tree = GitTree.model_validate(response.json())
        if tree.truncated:
            raise GitHubApiError(f"The file tree of {ref.full_name} is too large to list")
        return [entry.path for entry in tree.tree if entry.type == "blob"]


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    status = response.status_code
    if status == 404:
        raise PrNotFoundError(
            "Not found on GitHub: the PR does not exist, or the repository is private"
        )
    remaining = response.headers.get("x-ratelimit-remaining")
    if status == 429 or (status == 403 and remaining == "0"):
        raise RateLimitError(_reset_time(response))
    raise GitHubApiError(f"GitHub returned {status} for {response.request.url}", status_code=status)


def _reset_time(response: httpx.Response) -> datetime | None:
    reset = response.headers.get("x-ratelimit-reset")
    if reset is None or not reset.isdigit():
        return None
    return datetime.fromtimestamp(int(reset), tz=UTC)
