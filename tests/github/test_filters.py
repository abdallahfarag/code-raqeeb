import pytest

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.errors import PrTooLargeError
from code_raqeeb.github.filters import enforce_limits, skip_reason
from code_raqeeb.github.models import FileStatus, PrFile

PATCH = "@@ -1,1 +1,2 @@\n context\n+added"


def pr_file(filename: str, *, status: FileStatus = "modified", patch: str | None = PATCH) -> PrFile:
    return PrFile(filename=filename, status=status, additions=1, deletions=0, patch=patch)


@pytest.mark.parametrize(
    ("file", "reason"),
    [
        pytest.param(pr_file("src/Foo.cs", status="removed"), "deleted", id="removed"),
        pytest.param(pr_file("src/packages.lock.json"), "lockfile", id="lockfile"),
        pytest.param(pr_file("src/App/bin/Debug/Foo.cs"), "build output", id="bin"),
        pytest.param(pr_file("src/App/obj/Foo.cs"), "build output", id="obj"),
        pytest.param(pr_file("README.md"), "not a C# file", id="non-cs"),
        pytest.param(pr_file("src/Form1.Designer.cs"), "generated code", id="designer"),
        pytest.param(pr_file("src/View.g.cs"), "generated code", id="g-cs"),
        pytest.param(
            pr_file("src/Foo.cs", patch=None), "no diff (binary or too large)", id="no-patch"
        ),
    ],
)
def test_skip_reason(file: PrFile, reason: str) -> None:
    assert skip_reason(file) == reason


def test_normal_cs_file_is_reviewed() -> None:
    assert skip_reason(pr_file("src/App/Services/Greeter.cs")) is None


def changed(additions: int, deletions: int = 0) -> ChangedFile:
    return ChangedFile(
        path="src/Foo.cs", status="modified", additions=additions, deletions=deletions, hunks=()
    )


@pytest.mark.parametrize("count", [1, 2], ids=["below", "at"])
def test_file_count_within_limit(count: int) -> None:
    enforce_limits([changed(1)] * count, max_files=2, max_changed_lines=100)


def test_file_count_above_limit() -> None:
    with pytest.raises(PrTooLargeError, match="3 reviewable files"):
        enforce_limits([changed(1)] * 3, max_files=2, max_changed_lines=100)


@pytest.mark.parametrize(
    "files",
    [
        pytest.param([changed(4, 5)], id="below"),
        pytest.param([changed(4, 2), changed(3, 1)], id="at"),
    ],
)
def test_changed_lines_within_limit(files: list[ChangedFile]) -> None:
    enforce_limits(files, max_files=10, max_changed_lines=10)


def test_changed_lines_above_limit() -> None:
    with pytest.raises(PrTooLargeError, match="11 changed lines"):
        enforce_limits([changed(4, 2), changed(3, 2)], max_files=10, max_changed_lines=10)
