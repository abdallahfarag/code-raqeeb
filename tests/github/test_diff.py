import pytest

from code_raqeeb.github.diff import ChangedFile, DiffHunk, parse_patch


def hunk(start: int, length: int, added: set[int]) -> DiffHunk:
    return DiffHunk(start=start, length=length, added_lines=frozenset(added))


def changed_file(*hunks: DiffHunk) -> ChangedFile:
    return ChangedFile(
        patch="", path="src/Foo.cs", status="modified", additions=0, deletions=0, hunks=hunks
    )


SINGLE_HUNK = """\
@@ -10,4 +10,5 @@ public class Foo
 context
-removed
+added one
+added two
 context
 context"""


def test_single_hunk_tracks_head_line_numbers() -> None:
    assert parse_patch(SINGLE_HUNK) == (hunk(10, 5, {11, 12}),)


def test_two_hunks() -> None:
    patch = "@@ -1,2 +1,3 @@\n line1\n+new2\n line3\n@@ -20,2 +21,2 @@\n-old\n+new21\n ctx"

    assert parse_patch(patch) == (hunk(1, 3, {2}), hunk(21, 2, {21}))


def test_new_file() -> None:
    assert parse_patch("@@ -0,0 +1,3 @@\n+a\n+b\n+c") == (hunk(1, 3, {1, 2, 3}),)


def test_header_without_counts_defaults_length_to_one() -> None:
    assert parse_patch("@@ -1 +1 @@\n-old\n+new") == (hunk(1, 1, {1}),)


def test_deletion_only_hunk_has_no_commentable_lines() -> None:
    hunks = parse_patch("@@ -5,2 +4,0 @@\n-a\n-b")

    assert hunks == (hunk(4, 0, set()),)
    assert list(hunks[0].lines) == []
    file = changed_file(*hunks)
    assert file.added_lines == frozenset()
    assert not any(file.is_commentable(line) for line in range(1, 10))


def test_no_newline_marker_is_ignored() -> None:
    patch = (
        "@@ -1,1 +1,1 @@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file"
    )

    assert parse_patch(patch) == (hunk(1, 1, {1}),)


def test_text_before_first_header_is_ignored() -> None:
    patch = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,1 +1,2 @@\n ctx\n+new"

    assert parse_patch(patch) == (hunk(1, 2, {2}),)


def test_changed_file_added_lines_merges_hunks() -> None:
    file = changed_file(hunk(1, 3, {2}), hunk(21, 2, {21}))

    assert file.added_lines == frozenset({2, 21})


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param(10, True, id="leading-context"),
        pytest.param(11, True, id="added"),
        pytest.param(14, True, id="trailing-context"),
        pytest.param(9, False, id="before-hunk"),
        pytest.param(15, False, id="after-hunk"),
    ],
)
def test_is_commentable(line: int, expected: bool) -> None:
    file = changed_file(*parse_patch(SINGLE_HUNK))

    assert file.is_commentable(line) is expected
