import httpx
import pytest

from code_raqeeb.github.diff import DiffHunk
from code_raqeeb.github.errors import InvalidPrUrlError, PrivateRepoError, PrTooLargeError
from code_raqeeb.github.filters import SkippedFile
from code_raqeeb.github.service import load_pull_request
from tests.github.conftest import GitHubClientFactory, SyncHandler
from tests.github.factories import (
    BASE_SHA,
    HEAD_SHA,
    PR_URL,
    REF,
    Json,
    file_payload,
    make_settings,
    pr_payload,
    repo_payload,
)

GREETER_PATCH = (
    "@@ -1,2 +1,3 @@\n using System;\n+using System.Linq;\n class Greeter\n"
    "@@ -10 +11 @@\n-old\n+new"
)


def github_api(pull_request: Json, files: list[Json], requests: list[httpx.Request]) -> SyncHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/repos/octo/hello/pulls/7":
            return httpx.Response(200, json=pull_request)
        if request.url.path == "/repos/octo/hello/pulls/7/files":
            return httpx.Response(200, json=files)
        return httpx.Response(404)

    return handler


async def test_happy_path_splits_files_and_parses_hunks(
    make_github_client: GitHubClientFactory,
) -> None:
    files = [
        file_payload("src/Greeter.cs", additions=2, deletions=1, patch=GREETER_PATCH),
        file_payload("src/packages.lock.json"),
        file_payload("src/Form1.Designer.cs"),
        file_payload("docs/README.md"),
    ]
    requests: list[httpx.Request] = []
    handler = github_api(pr_payload(base_repo=repo_payload()), files, requests)
    settings = make_settings()

    bundle = await load_pull_request(make_github_client(handler, settings), PR_URL, settings)

    assert bundle.ref == REF
    assert bundle.base_sha == BASE_SHA
    assert bundle.head_sha == HEAD_SHA
    [greeter] = bundle.files
    assert greeter.path == "src/Greeter.cs"
    assert greeter.hunks == (
        DiffHunk(start=1, length=3, added_lines=frozenset({2})),
        DiffHunk(start=11, length=1, added_lines=frozenset({11})),
    )
    assert bundle.skipped == (
        SkippedFile(path="src/packages.lock.json", reason="lockfile"),
        SkippedFile(path="src/Form1.Designer.cs", reason="generated code"),
        SkippedFile(path="docs/README.md", reason="not a C# file"),
    )


@pytest.mark.parametrize(
    "base_repo",
    [pytest.param(repo_payload(private=True), id="private"), pytest.param(None, id="repo-none")],
)
async def test_non_public_repo_rejected(
    make_github_client: GitHubClientFactory, base_repo: Json | None
) -> None:
    requests: list[httpx.Request] = []
    handler = github_api(pr_payload(base_repo=base_repo), [], requests)
    settings = make_settings()

    with pytest.raises(PrivateRepoError):
        await load_pull_request(make_github_client(handler, settings), PR_URL, settings)

    assert [r.url.path for r in requests] == ["/repos/octo/hello/pulls/7"]


async def test_too_many_files_rejected(make_github_client: GitHubClientFactory) -> None:
    files = [file_payload("src/A.cs"), file_payload("src/B.cs")]
    requests: list[httpx.Request] = []
    handler = github_api(pr_payload(base_repo=repo_payload()), files, requests)
    settings = make_settings(max_review_files=1)

    with pytest.raises(PrTooLargeError):
        await load_pull_request(make_github_client(handler, settings), PR_URL, settings)


async def test_invalid_url_rejected_before_any_request(
    make_github_client: GitHubClientFactory,
) -> None:
    requests: list[httpx.Request] = []
    handler = github_api(pr_payload(base_repo=repo_payload()), [], requests)
    settings = make_settings()
    client = make_github_client(handler, settings)

    with pytest.raises(InvalidPrUrlError):
        await load_pull_request(client, "https://github.com/octo/hello/issues/7", settings)

    assert requests == []
