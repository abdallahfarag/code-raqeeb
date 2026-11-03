import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from code_raqeeb.github.errors import InvalidPrUrlError

FileStatus = Literal["added", "removed", "modified", "renamed", "copied", "changed", "unchanged"]

_PR_URL = re.compile(
    r"https://github\.com/"
    r"(?P<owner>[A-Za-z0-9-]{1,39})/"
    r"(?P<repo>[A-Za-z0-9._-]{1,100})/"
    r"pull/(?P<number>[1-9][0-9]*)"
    r"(?:/(?:files|commits|checks))?/?"
)


class PrRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    owner: str
    repo: str
    number: int = Field(gt=0)

    @classmethod
    def from_url(cls, url: str) -> Self:
        match = _PR_URL.fullmatch(url.strip())
        if match is None:
            raise InvalidPrUrlError(url)
        return cls(owner=match["owner"], repo=match["repo"], number=int(match["number"]))

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


class GitHubRepo(BaseModel):
    full_name: str
    private: bool


class GitHubBranch(BaseModel):
    ref: str
    sha: str
    repo: GitHubRepo | None


class PullRequest(BaseModel):
    number: int
    title: str
    state: Literal["open", "closed"]
    html_url: str
    base: GitHubBranch
    head: GitHubBranch
    additions: int
    deletions: int
    changed_files: int


class PrFile(BaseModel):
    filename: str
    status: FileStatus
    additions: int
    deletions: int
    patch: str | None = None
    previous_filename: str | None = None


class TreeEntry(BaseModel):
    path: str
    type: Literal["blob", "tree", "commit"]


class GitTree(BaseModel):
    sha: str
    tree: list[TreeEntry]
    truncated: bool
