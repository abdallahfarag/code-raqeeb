from pathlib import Path

import pytest
from pydantic import TypeAdapter

from code_raqeeb.github.diff import added_lines_with_text, map_patch_lines, parse_patch
from code_raqeeb.github.models import PrFile
from code_raqeeb.helper.text import split_lines

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "prs"
FOLDERS = sorted(path for path in FIXTURES_DIR.iterdir() if path.is_dir())

_PR_FILES = TypeAdapter(list[PrFile])

TWO_HUNKS = """\
@@ -1,3 +1,4 @@ class Foo
 context one
-removed
+new two
+new three
 context four
@@ -20,2 +21,3 @@ void Bar()
 context 21
+new 22
 context 23"""

WITH_MARKERS = """\
@@ -1,2 +1,2 @@ class Foo
 context
-old
\\ No newline at end of file
+new
\\ No newline at end of file"""

unicode_line_breaks = pytest.mark.parametrize(
    "separator",
    [pytest.param("\x0c", id="form-feed"), pytest.param("\N{LINE SEPARATOR}", id="line-separator")],
)


def test_added_lines_have_numbers_and_texts_across_two_hunks() -> None:
    assert added_lines_with_text(TWO_HUNKS) == [
        (2, "new two"),
        (3, "new three"),
        (22, "new 22"),
    ]


def test_added_lines_skip_the_no_newline_marker() -> None:
    assert added_lines_with_text(WITH_MARKERS) == [(2, "new")]


def test_added_lines_keep_blank_lines_and_marker_like_code() -> None:
    patch = "@@ -0,0 +1,3 @@\n+\n+-1;\n+@@ text"

    assert added_lines_with_text(patch) == [(1, ""), (2, "-1;"), (3, "@@ text")]


def test_added_lines_of_an_empty_patch() -> None:
    assert added_lines_with_text("") == []


@pytest.mark.parametrize("folder", FOLDERS, ids=[path.name for path in FOLDERS])
def test_added_lines_agree_with_parse_patch_on_recorded_patches(folder: Path) -> None:
    files = _PR_FILES.validate_json((folder / "files.json").read_text(encoding="utf-8"))

    for file in files:
        patch = file.patch or ""
        expected = {line for hunk in parse_patch(patch) for line in hunk.added_lines}
        assert {line for line, _ in added_lines_with_text(patch)} == expected


@unicode_line_breaks
def test_added_lines_are_not_shifted_by_a_unicode_line_break(separator: str) -> None:
    patch = f"@@ -1,2 +1,4 @@\n first\n+// a{separator}b\n+next\n last"

    assert added_lines_with_text(patch) == [(2, f"// a{separator}b"), (3, "next")]


def test_map_patch_lines_with_identity_returns_the_same_patch() -> None:
    assert map_patch_lines(WITH_MARKERS, lambda text: text) == WITH_MARKERS


def test_map_patch_lines_keeps_headers_no_newline_markers_and_line_count() -> None:
    mapped = map_patch_lines(WITH_MARKERS, str.upper)

    assert mapped.splitlines() == [
        "@@ -1,2 +1,2 @@ class Foo",
        " CONTEXT",
        "-OLD",
        "\\ No newline at end of file",
        "+NEW",
        "\\ No newline at end of file",
    ]


def test_map_patch_lines_keeps_the_line_count_across_two_hunks() -> None:
    mapped = map_patch_lines(TWO_HUNKS, str.upper)

    assert len(split_lines(mapped)) == len(split_lines(TWO_HUNKS))
    assert parse_patch(mapped) == parse_patch(TWO_HUNKS)


def test_map_patch_lines_passes_text_without_its_marker() -> None:
    seen: list[str] = []

    def record(text: str) -> str:
        seen.append(text)
        return text

    map_patch_lines(TWO_HUNKS, record)

    assert seen == [
        "context one",
        "removed",
        "new two",
        "new three",
        "context four",
        "context 21",
        "new 22",
        "context 23",
    ]


def test_map_patch_lines_transforms_added_code_that_looks_like_a_header() -> None:
    assert map_patch_lines("@@ -0,0 +1 @@\n+@@ text", str.upper) == "@@ -0,0 +1 @@\n+@@ TEXT"


def test_map_patch_lines_keeps_blank_lines() -> None:
    assert map_patch_lines("@@ -1,2 +1,2 @@\n\n a", str.upper) == "@@ -1,2 +1,2 @@\n\n A"


@unicode_line_breaks
def test_map_patch_lines_keeps_the_line_count_with_a_unicode_line_break(separator: str) -> None:
    patch = f"@@ -1,1 +1,2 @@\n first\n+// a{separator}b"

    assert map_patch_lines(patch, lambda text: text) == patch
