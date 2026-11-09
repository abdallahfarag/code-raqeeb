from pydantic import SecretStr

from code_raqeeb.config import Settings
from code_raqeeb.github.models import FileStatus, PrRef

REF = PrRef(owner="octo", repo="hello", number=7)
PR_URL = "https://github.com/octo/hello/pull/7"
BASE_SHA = "base000"
HEAD_SHA = "head111"

type Json = dict[str, object]


def make_settings(
    *,
    github_token: str | None = None,
    github_concurrency: int = 5,
    max_review_files: int = 20,
    max_review_changed_lines: int = 800,
    allow_private_repos: bool = False,
) -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        github_token=SecretStr(github_token) if github_token is not None else None,
        github_concurrency=github_concurrency,
        max_review_files=max_review_files,
        max_review_changed_lines=max_review_changed_lines,
        allow_private_repos=allow_private_repos,
    )


def repo_payload(*, private: bool = False) -> Json:
    return {"full_name": "octo/hello", "private": private}


def pr_payload(*, base_repo: Json | None) -> Json:
    return {
        "number": 7,
        "title": "Add greeting",
        "state": "open",
        "html_url": PR_URL,
        "base": {"ref": "main", "sha": BASE_SHA, "repo": base_repo},
        "head": {"ref": "feature", "sha": HEAD_SHA, "repo": None},
        "additions": 3,
        "deletions": 1,
        "changed_files": 1,
    }


def tree_payload(entries: list[tuple[str, str]], *, truncated: bool = False) -> Json:
    return {
        "sha": HEAD_SHA,
        "tree": [{"path": path, "type": kind} for path, kind in entries],
        "truncated": truncated,
    }


def file_payload(
    filename: str,
    *,
    status: FileStatus = "modified",
    additions: int = 1,
    deletions: int = 0,
    patch: str | None = "@@ -1,1 +1,2 @@\n context\n+added",
) -> Json:
    payload: Json = {
        "filename": filename,
        "status": status,
        "additions": additions,
        "deletions": deletions,
    }
    if patch is not None:
        payload["patch"] = patch
    return payload
