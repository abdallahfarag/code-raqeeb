from collections.abc import Sequence
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.errors import PrTooLargeError
from code_raqeeb.github.models import PrFile

_REVIEWED_SUFFIX = ".cs"
_LOCKFILES = frozenset({"packages.lock.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"})
_BUILD_FOLDERS = frozenset({"bin", "obj"})
_GENERATED_SUFFIXES = (".Designer.cs", ".g.cs", ".g.i.cs", ".AssemblyInfo.cs", "ModelSnapshot.cs")


class SkippedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    reason: str


def path_skip_reason(path_text: str) -> str | None:
    path = PurePosixPath(path_text)
    if path.name in _LOCKFILES:
        return "lockfile"
    if _BUILD_FOLDERS.intersection(path.parts):
        return "build output"
    if path.suffix != _REVIEWED_SUFFIX:
        return "not a C# file"
    if path.name.endswith(_GENERATED_SUFFIXES):
        return "generated code"
    return None


def skip_reason(file: PrFile) -> str | None:
    if file.status == "removed":
        return "deleted"
    reason = path_skip_reason(file.filename)
    if reason is not None:
        return reason
    if file.patch is None:
        return "no diff (binary or too large)"
    return None


def enforce_limits(files: Sequence[ChangedFile], *, max_files: int, max_changed_lines: int) -> None:
    if len(files) > max_files:
        raise PrTooLargeError(f"{len(files)} reviewable files; the limit is {max_files}")
    changed = sum(file.additions + file.deletions for file in files)
    if changed > max_changed_lines:
        raise PrTooLargeError(
            f"{changed} changed lines in reviewable files; the limit is {max_changed_lines}"
        )
