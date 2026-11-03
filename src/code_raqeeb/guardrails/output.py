from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.guardrails.text_guard import TextGuards, build_text_guards, clean_text
from code_raqeeb.review.models import ReviewComment
from code_raqeeb.rules.models import RuleSet

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


class DroppedComment(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    line: int
    reason: str


class OutputCheck(BaseModel):
    model_config = ConfigDict(frozen=True)
    kept: tuple[ReviewComment, ...]
    dropped: tuple[DroppedComment, ...]


def check_comments(
    comments: Sequence[ReviewComment],
    *,
    files: Mapping[str, ChangedFile],
    rules: RuleSet,
    max_per_file: int,
    max_chars: int,
) -> OutputCheck:
    guards = build_text_guards(max_chars)
    dropped: list[DroppedComment] = []

    located: list[ReviewComment] = []
    for comment in comments:
        file = files.get(comment.path)
        if file is None:
            dropped.append(_drop(comment, "file is not part of this PR"))
        elif not file.is_commentable(comment.line):
            dropped.append(_drop(comment, "line is outside the diff"))
        else:
            located.append(_clean(comment, rules, guards))

    unique: dict[tuple[str, int, str], ReviewComment] = {}
    for comment in sorted(located, key=_rank):
        key = (comment.path, comment.line, comment.rule_id or comment.category)
        if key in unique:
            dropped.append(_drop(comment, "duplicate"))
        else:
            unique[key] = comment

    kept: list[ReviewComment] = []
    per_file: dict[str, int] = {}
    for comment in unique.values():
        count = per_file.get(comment.path, 0)
        if count >= max_per_file:
            dropped.append(_drop(comment, "too many comments for this file"))
            continue
        per_file[comment.path] = count + 1
        kept.append(comment)

    kept.sort(key=lambda comment: (comment.path, comment.line))
    return OutputCheck(kept=tuple(kept), dropped=tuple(dropped))


def _clean(comment: ReviewComment, rules: RuleSet, guards: TextGuards) -> ReviewComment:
    rule = rules.get(comment.rule_id) if comment.rule_id else None
    suggestion = comment.suggestion
    return comment.model_copy(
        update={
            "rule_id": rule.id if rule else None,
            "category": rule.category if rule else comment.category,
            "message": clean_text(guards, comment.message),
            "suggestion": clean_text(guards, suggestion) if suggestion else None,
        }
    )


def _rank(comment: ReviewComment) -> tuple[int, int]:
    return (_SEVERITY_RANK[comment.severity], comment.line)


def _drop(comment: ReviewComment, reason: str) -> DroppedComment:
    return DroppedComment(path=comment.path, line=comment.line, reason=reason)
