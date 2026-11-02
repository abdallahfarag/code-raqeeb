from pydantic import BaseModel, ConfigDict

from code_raqeeb.config import Settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.github.diff import ChangedFile, parse_patch
from code_raqeeb.github.errors import PrivateRepoError
from code_raqeeb.github.filters import SkippedFile, enforce_limits, skip_reason
from code_raqeeb.github.models import PrRef, PullRequest


class PullRequestBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref: PrRef
    pull_request: PullRequest
    files: tuple[ChangedFile, ...]
    skipped: tuple[SkippedFile, ...]

    @property
    def base_sha(self) -> str:
        return self.pull_request.base.sha

    @property
    def head_sha(self) -> str:
        return self.pull_request.head.sha


async def load_pull_request(
    client: GitHubClient, url: str, settings: Settings
) -> PullRequestBundle:
    ref = PrRef.from_url(url)
    pull_request = await client.get_pull_request(ref)

    base_repo = pull_request.base.repo
    if base_repo is None or base_repo.private:
        raise PrivateRepoError(f"{ref.full_name} is not a public repository")

    files: list[ChangedFile] = []
    skipped: list[SkippedFile] = []
    for pr_file in await client.list_pull_request_files(ref):
        reason = skip_reason(pr_file)
        if reason is not None or pr_file.patch is None:
            skipped.append(SkippedFile(path=pr_file.filename, reason=reason or "no diff"))
            continue
        files.append(
            ChangedFile(
                patch=pr_file.patch,
                path=pr_file.filename,
                status=pr_file.status,
                additions=pr_file.additions,
                deletions=pr_file.deletions,
                hunks=parse_patch(pr_file.patch),
            )
        )

    enforce_limits(
        files,
        max_files=settings.max_review_files,
        max_changed_lines=settings.max_review_changed_lines,
    )
    return PullRequestBundle(
        ref=ref, pull_request=pull_request, files=tuple(files), skipped=tuple(skipped)
    )
