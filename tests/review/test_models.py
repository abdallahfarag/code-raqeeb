from typing import Any

import pytest
from pydantic import ValidationError

from code_raqeeb.review.models import DraftComment, FileReview, ReviewComment, to_comment


def draft_payload(**overrides: Any) -> dict[str, Any]:
    return {
        "line": 27,
        "severity": "high",
        "category": "bug",
        "message": "Blocks on async code",
        **overrides,
    }


def test_to_comment_builds_the_exact_permalink() -> None:
    comment = to_comment(
        DraftComment.model_validate(draft_payload()),
        repo="octo/hello",
        head_sha="abc123",
        path="src/Orders/OrdersController.cs",
    )

    assert comment.permalink == (
        "https://github.com/octo/hello/blob/abc123/src/Orders/OrdersController.cs#L27"
    )


def test_to_comment_copies_every_draft_field() -> None:
    draft = DraftComment.model_validate(
        draft_payload(severity="low", category="style", rule_id="BUG-01", suggestion="Use await")
    )

    comment = to_comment(draft, repo="octo/hello", head_sha="abc123", path="src/Foo.cs")

    assert comment == ReviewComment(
        path="src/Foo.cs",
        line=27,
        severity="low",
        category="style",
        rule_id="BUG-01",
        message="Blocks on async code",
        suggestion="Use await",
        permalink="https://github.com/octo/hello/blob/abc123/src/Foo.cs#L27",
    )


def test_draft_comment_defaults_rule_and_suggestion_to_none() -> None:
    draft = DraftComment.model_validate(draft_payload())

    assert (draft.rule_id, draft.suggestion) == (None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("line", 0, id="line-zero"),
        pytest.param("line", -1, id="negative-line"),
        pytest.param("severity", "critical", id="unknown-severity"),
        pytest.param("category", "naming", id="unknown-category"),
        pytest.param("message", "", id="empty-message"),
    ],
)
def test_draft_comment_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as exc_info:
        DraftComment.model_validate(draft_payload(**{field: value}))

    assert [error["loc"] for error in exc_info.value.errors()] == [(field,)]


def test_file_review_defaults_to_no_comments() -> None:
    assert FileReview().comments == []
