import json
from pathlib import Path

import pytest

from code_raqeeb.github.diff import parse_patch, render_numbered_patch

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "prs" / "02-pay-endpoint"


def fixture_patch(path: str) -> str:
    files = json.loads((FIXTURE / "files.json").read_text(encoding="utf-8"))
    return next(file["patch"] for file in files if file["filename"] == path)


def test_added_and_context_lines_get_their_new_side_number() -> None:
    patch = "@@ -10,2 +10,3 @@ class Foo\n context\n+added\n context two"

    assert render_numbered_patch(patch).splitlines() == [
        "@@ -10,2 +10,3 @@ class Foo",
        "    10   context",
        "    11 + added",
        "    12   context two",
    ]


def test_removed_lines_have_no_number_and_do_not_advance_numbering() -> None:
    patch = "@@ -1,3 +1,2 @@\n first\n-gone\n last"

    assert render_numbered_patch(patch).splitlines() == [
        "@@ -1,3 +1,2 @@",
        "     1   first",
        "       - gone",
        "     2   last",
    ]


def test_no_newline_marker_is_skipped() -> None:
    patch = "@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file"

    assert render_numbered_patch(patch).splitlines() == [
        "@@ -1 +1 @@",
        "       - old",
        "     1 + new",
    ]


def test_two_hunks_restart_numbering() -> None:
    patch = "@@ -1,1 +1,2 @@\n one\n+two\n@@ -20,1 +21,2 @@\n twenty\n+new"

    assert render_numbered_patch(patch).splitlines() == [
        "@@ -1,1 +1,2 @@",
        "     1   one",
        "     2 + two",
        "@@ -20,1 +21,2 @@",
        "    21   twenty",
        "    22 + new",
    ]


def test_new_file_numbers_every_line_from_one() -> None:
    assert render_numbered_patch("@@ -0,0 +1,2 @@\n+a\n+b").splitlines() == [
        "@@ -0,0 +1,2 @@",
        "     1 + a",
        "     2 + b",
    ]


def test_blank_context_line_keeps_its_number() -> None:
    assert render_numbered_patch("@@ -1,2 +1,2 @@\n \n next").splitlines() == [
        "@@ -1,2 +1,2 @@",
        "     1   ",
        "     2   next",
    ]


def test_code_that_looks_like_a_marker_keeps_its_text() -> None:
    assert render_numbered_patch("@@ -1,0 +1,2 @@\n+-1;\n+@@ text").splitlines() == [
        "@@ -1,0 +1,2 @@",
        "     1 + -1;",
        "     2 + @@ text",
    ]


def test_empty_patch_renders_nothing() -> None:
    assert render_numbered_patch("") == ""


@pytest.mark.parametrize(
    ("path", "line", "code"),
    [
        pytest.param(
            "src/Orders.Api/Controllers/OrdersController.cs", 27, ".Result", id="blocking-call"
        ),
        pytest.param("src/Orders.Api/Services/OrderService.cs", 26, "async void", id="async-void"),
    ],
)
def test_recorded_patch_numbers_match_expected_findings(path: str, line: int, code: str) -> None:
    rendered = render_numbered_patch(fixture_patch(path)).splitlines()

    [numbered] = [text for text in rendered if text.startswith(f"{line:>6} ")]

    assert code in numbered


@pytest.mark.parametrize(
    "separator",
    [pytest.param("\x0c", id="form-feed"), pytest.param("\N{LINE SEPARATOR}", id="line-separator")],
)
def test_numbering_is_not_shifted_by_a_unicode_line_break_inside_a_line(separator: str) -> None:
    patch = f"@@ -1,2 +1,3 @@\n first\n+// a{separator}b\n last"

    assert render_numbered_patch(patch).splitlines()[-1] == "     3   last"


@pytest.mark.parametrize(
    "separator",
    [pytest.param("\x0c", id="form-feed"), pytest.param("\N{LINE SEPARATOR}", id="line-separator")],
)
def test_added_lines_are_not_shifted_by_a_unicode_line_break_inside_a_line(separator: str) -> None:
    patch = f"@@ -1,2 +1,4 @@\n first\n // a{separator}b\n+added\n last"

    [hunk] = parse_patch(patch)

    assert hunk.added_lines == frozenset({3})
