from collections.abc import Sequence

import pytest

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.guardrails.output import OutputCheck, check_comments
from code_raqeeb.guardrails.secrets import REDACTED
from code_raqeeb.review.models import ReviewComment
from code_raqeeb.rules.models import Category, ReviewRule, RuleSet
from tests.guardrails.fakes import AWS_KEY, PATH, make_comment, make_file

OTHER_PATH = "src/Bar.cs"
PATCH = "@@ -0,0 +1,20 @@\n" + "\n".join(f"+line {n}" for n in range(1, 21))
FILES: dict[str, ChangedFile] = {
    PATH: make_file(PATCH, PATH),
    OTHER_PATH: make_file(PATCH, OTHER_PATH),
}


def rule(rule_id: str, category: Category) -> ReviewRule:
    return ReviewRule(
        id=rule_id, category=category, severity="high", title="Title", description="Description"
    )


RULES = RuleSet(language="csharp", rules=(rule("BUG-01", "bug"), rule("SEC-02", "security")))


def check(
    comments: Sequence[ReviewComment], *, max_per_file: int = 8, max_chars: int = 200
) -> OutputCheck:
    return check_comments(
        comments, files=FILES, rules=RULES, max_per_file=max_per_file, max_chars=max_chars
    )


def test_valid_comment_is_kept_unchanged() -> None:
    comment = make_comment(line=3, rule_id="BUG-01", suggestion="Fix it")

    result = check([comment])

    assert result.kept == (comment,)
    assert result.dropped == ()


def test_no_comments_gives_an_empty_result() -> None:
    assert check([]) == OutputCheck(kept=(), dropped=())


def test_unknown_file_is_dropped_with_a_reason() -> None:
    result = check([make_comment(path="src/Other.cs", line=3)])

    assert result.kept == ()
    assert [(d.path, d.line, d.reason) for d in result.dropped] == [
        ("src/Other.cs", 3, "file is not part of this PR")
    ]


def test_line_outside_the_diff_is_dropped_with_a_reason() -> None:
    result = check([make_comment(line=21)])

    assert result.kept == ()
    assert [(d.path, d.line, d.reason) for d in result.dropped] == [
        (PATH, 21, "line is outside the diff")
    ]


def test_unknown_rule_id_is_cleared_and_the_comment_is_kept() -> None:
    [kept] = check([make_comment(rule_id="XXX-99", category="style")]).kept

    assert kept.rule_id is None
    assert kept.category == "style"


def test_category_is_corrected_from_the_rule() -> None:
    [kept] = check([make_comment(rule_id="SEC-02", category="bug")]).kept

    assert kept.category == "security"


def test_secrets_in_message_and_suggestion_are_redacted() -> None:
    [kept] = check(
        [make_comment(message=f"Leaked {AWS_KEY}", suggestion=f"Rotate {AWS_KEY} today")]
    ).kept

    assert kept.message == f"Leaked {REDACTED}"
    assert kept.suggestion == f"Rotate {REDACTED} today"


def test_missing_suggestion_stays_none() -> None:
    [kept] = check([make_comment(suggestion=None)]).kept

    assert kept.suggestion is None


def test_long_message_and_suggestion_are_cut_to_max_chars_with_an_ellipsis() -> None:
    [kept] = check([make_comment(message="m" * 80, suggestion="s" * 80)], max_chars=50).kept

    assert kept.message == "m" * 49 + "…"
    assert kept.suggestion == "s" * 49 + "…"


def test_text_of_exactly_max_chars_is_not_cut() -> None:
    [kept] = check([make_comment(message="m" * 50)], max_chars=50).kept

    assert kept.message == "m" * 50


def test_duplicates_are_removed_keeping_the_higher_severity() -> None:
    low = make_comment(line=3, severity="low", rule_id="BUG-01", message="Low one")
    high = make_comment(line=3, severity="high", rule_id="BUG-01", message="High one")
    medium = make_comment(line=3, severity="medium", rule_id="BUG-01", message="Medium one")

    result = check([low, high, medium])

    assert [comment.message for comment in result.kept] == ["High one"]
    assert [d.reason for d in result.dropped] == ["duplicate", "duplicate"]


def test_comments_without_a_rule_are_duplicates_within_one_category() -> None:
    result = check(
        [
            make_comment(line=3, severity="low", category="bug"),
            make_comment(line=3, severity="high", category="bug"),
            make_comment(line=3, severity="low", category="style"),
        ]
    )

    assert [(c.category, c.severity) for c in result.kept] == [("bug", "high"), ("style", "low")]


def test_different_rules_on_one_line_are_not_duplicates() -> None:
    result = check([make_comment(line=3, rule_id="BUG-01"), make_comment(line=3, rule_id="SEC-02")])

    assert {comment.rule_id for comment in result.kept} == {"BUG-01", "SEC-02"}
    assert result.dropped == ()


def test_same_rule_on_other_line_or_file_is_not_a_duplicate() -> None:
    result = check(
        [
            make_comment(line=3, rule_id="BUG-01"),
            make_comment(line=4, rule_id="BUG-01"),
            make_comment(path=OTHER_PATH, line=3, rule_id="BUG-01"),
        ]
    )

    assert len(result.kept) == 3


def test_max_per_file_keeps_the_most_severe() -> None:
    result = check(
        [
            make_comment(line=1, severity="low"),
            make_comment(line=2, severity="high"),
            make_comment(line=3, severity="medium"),
        ],
        max_per_file=2,
    )

    assert [(c.line, c.severity) for c in result.kept] == [(2, "high"), (3, "medium")]
    assert [(d.line, d.reason) for d in result.dropped] == [(1, "too many comments for this file")]


def test_max_per_file_counts_each_file_separately() -> None:
    result = check(
        [make_comment(line=1), make_comment(line=2), make_comment(path=OTHER_PATH, line=1)],
        max_per_file=1,
    )

    assert [(c.path, c.line) for c in result.kept] == [(OTHER_PATH, 1), (PATH, 1)]


def test_kept_is_sorted_by_path_and_line() -> None:
    result = check(
        [
            make_comment(path=PATH, line=9, severity="high"),
            make_comment(path=PATH, line=2, severity="low"),
            make_comment(path=OTHER_PATH, line=5, severity="medium"),
            make_comment(path=OTHER_PATH, line=1, severity="low"),
        ]
    )

    assert [(c.path, c.line) for c in result.kept] == [
        (OTHER_PATH, 1),
        (OTHER_PATH, 5),
        (PATH, 2),
        (PATH, 9),
    ]


@pytest.mark.parametrize(
    "comment",
    [
        pytest.param(make_comment(path="src/Other.cs"), id="unknown-file"),
        pytest.param(make_comment(line=99), id="outside-diff"),
    ],
)
def test_every_input_comment_is_either_kept_or_dropped(comment: ReviewComment) -> None:
    result = check([comment, make_comment(line=1)])

    assert (len(result.kept), len(result.dropped)) == (1, 1)
