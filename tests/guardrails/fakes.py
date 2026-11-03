from code_raqeeb.github.diff import ChangedFile, parse_patch
from code_raqeeb.review.models import ReviewComment
from code_raqeeb.rules.models import Category, Severity

GITHUB_TOKEN = "ghp_" + "a1B2" * 9
GITHUB_PAT = "github_pat_" + "A1b2C3d4E5f6G7h8I9j0K1"
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
GOOGLE_KEY = "AIza" + "SyD3" * 8 + "abc"
SLACK_TOKEN = "xoxb-" + "1234567890-abcdefghij"
JWT = "eyJ" + "hbGciOiJIUzI1NiJ9" + ".eyJ" + "zdWIiOiIxMjM0NTYifQ" + ".dozjgNryP4J3jVmNHl0w5N"

REPO = "octo/hello"
HEAD_SHA = "head111"
PATH = "src/Foo.cs"


class FakeInjectionScanner:
    def __init__(self, *, flagged: bool = False) -> None:
        self.flagged = flagged
        self.calls: list[str] = []

    def is_injection(self, text: str) -> bool:
        self.calls.append(text)
        return self.flagged


def make_file(patch: str, path: str = PATH) -> ChangedFile:
    return ChangedFile(
        patch=patch,
        path=path,
        status="modified",
        additions=0,
        deletions=0,
        hunks=parse_patch(patch),
    )


def make_comment(
    *,
    path: str = PATH,
    line: int = 1,
    severity: Severity = "medium",
    category: Category = "bug",
    rule_id: str | None = None,
    message: str = "Something is wrong",
    suggestion: str | None = None,
) -> ReviewComment:
    return ReviewComment(
        path=path,
        line=line,
        severity=severity,
        category=category,
        rule_id=rule_id,
        message=message,
        suggestion=suggestion,
        permalink=f"https://github.com/{REPO}/blob/{HEAD_SHA}/{path}#L{line}",
    )
