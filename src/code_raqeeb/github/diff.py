import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from code_raqeeb.github.models import FileStatus

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<length>\d+))? @@")


class DiffHunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    length: int
    added_lines: frozenset[int]

    @property
    def lines(self) -> range:
        return range(self.start, self.start + self.length)


class ChangedFile(BaseModel):
    model_config = ConfigDict(frozen=True)
    patch: str
    path: str
    status: FileStatus
    additions: int
    deletions: int
    hunks: tuple[DiffHunk, ...]

    @property
    def added_lines(self) -> frozenset[int]:
        added: set[int] = set()
        for hunk in self.hunks:
            added |= hunk.added_lines
        return frozenset(added)

    def is_commentable(self, line: int) -> bool:
        return any(line in hunk.lines for hunk in self.hunks)


@dataclass
class _HunkBuilder:
    start: int
    length: int
    next_line: int
    added: list[int] = field(default_factory=list[int])

    def build(self) -> DiffHunk:
        return DiffHunk(start=self.start, length=self.length, added_lines=frozenset(self.added))


def parse_patch(patch: str) -> tuple[DiffHunk, ...]:
    hunks: list[DiffHunk] = []
    current: _HunkBuilder | None = None

    for raw_line in patch.splitlines():
        header = _HUNK_HEADER.match(raw_line)
        if header is not None:
            if current is not None:
                hunks.append(current.build())
            start = int(header["start"])
            length = int(header["length"]) if header["length"] is not None else 1
            current = _HunkBuilder(start=start, length=length, next_line=start)
            continue

        if current is None:
            continue
        if raw_line.startswith("+"):
            current.added.append(current.next_line)
            current.next_line += 1
        elif raw_line.startswith(("-", "\\")):
            pass
        else:
            current.next_line += 1

    if current is not None:
        hunks.append(current.build())
    return tuple(hunks)


def render_numbered_patch(patch: str) -> str:
    rendered: list[str] = []
    next_line = 0
    for raw_line in patch.splitlines():
        header = _HUNK_HEADER.match(raw_line)
        if header is not None:
            next_line = int(header["start"])
            rendered.append(raw_line)
        elif raw_line.startswith("+"):
            rendered.append(f"{next_line:>6} + {raw_line[1:]}")
            next_line += 1
        elif raw_line.startswith("-"):
            rendered.append(f"{'':>6} - {raw_line[1:]}")
        elif raw_line.startswith("\\"):
            continue
        else:
            rendered.append(f"{next_line:>6}   {raw_line[1:]}")
            next_line += 1
    return "\n".join(rendered)
