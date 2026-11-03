from pydantic import BaseModel, ConfigDict, Field

from code_raqeeb.rules.models import Category, Severity


class DraftComment(BaseModel):
    line: int = Field(gt=0, description="New-side line number shown in the numbered diff")
    severity: Severity
    category: Category
    rule_id: str | None = Field(
        default=None, description="A rule ID from get_rules, if one applies"
    )
    message: str = Field(min_length=1, description="What is wrong and why it matters")
    suggestion: str | None = Field(default=None, description="A concrete fix, if there is one")


class FileReview(BaseModel):
    comments: list[DraftComment] = Field(default_factory=list[DraftComment])


class ReviewComment(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    line: int
    severity: Severity
    category: Category
    rule_id: str | None
    message: str
    suggestion: str | None
    permalink: str


def to_comment(draft: DraftComment, *, repo: str, head_sha: str, path: str) -> ReviewComment:
    return ReviewComment(
        path=path,
        line=draft.line,
        severity=draft.severity,
        category=draft.category,
        rule_id=draft.rule_id,
        message=draft.message,
        suggestion=draft.suggestion,
        permalink=permalink(repo, head_sha, path, draft.line),
    )


def permalink(repo: str, head_sha: str, path: str, line: int) -> str:
    return f"https://github.com/{repo}/blob/{head_sha}/{path}#L{line}"
