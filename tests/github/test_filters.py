import pytest

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.errors import PrTooLargeError
from code_raqeeb.github.filters import enforce_limits, path_skip_reason, skip_reason
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


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        pytest.param("packages.lock.json", "lockfile", id="nuget-lock"),
        pytest.param("web/package-lock.json", "lockfile", id="npm-lock"),
        pytest.param("web/yarn.lock", "lockfile", id="yarn-lock"),
        pytest.param("web/pnpm-lock.yaml", "lockfile", id="pnpm-lock"),
        pytest.param("bin/Foo.cs", "build output", id="bin-at-root"),
        pytest.param("src/App/bin/Debug/Foo.cs", "build output", id="bin-nested"),
        pytest.param("src/App/obj/Foo.cs", "build output", id="obj-nested"),
        pytest.param("src/App/obj/project.assets.json", "build output", id="build-non-cs"),
        pytest.param("README.md", "not a C# file", id="markdown"),
        pytest.param("src/App.csproj", "not a C# file", id="csproj"),
        pytest.param("src/Form1.Designer.cs", "generated code", id="designer"),
        pytest.param("src/View.g.cs", "generated code", id="g-cs"),
        pytest.param("src/View.g.i.cs", "generated code", id="g-i-cs"),
        pytest.param("src/App.AssemblyInfo.cs", "generated code", id="assembly-info"),
        pytest.param("Migrations/AppModelSnapshot.cs", "generated code", id="model-snapshot"),
    ],
)
def test_path_skip_reason(path: str, reason: str) -> None:
    assert path_skip_reason(path) == reason


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("Foo.cs", id="root"),
        pytest.param("src/App/Services/Greeter.cs", id="nested"),
        pytest.param("src/binary/Foo.cs", id="folder-containing-bin"),
        pytest.param("src/bin.cs", id="file-named-bin"),
    ],
)
def test_path_skip_reason_keeps_source_files(path: str) -> None:
    assert path_skip_reason(path) is None


def test_skip_reason_reports_deleted_before_path_reason() -> None:
    assert skip_reason(pr_file("README.md", status="removed")) == "deleted"


def test_skip_reason_reports_path_reason_before_missing_patch() -> None:
    assert skip_reason(pr_file("README.md", patch=None)) == "not a C# file"


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
