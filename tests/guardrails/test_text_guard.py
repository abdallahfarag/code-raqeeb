# pyright: reportPrivateUsage=false

import pytest
from guardrails.types import OnFailAction
from guardrails.utils.hub_telemetry_utils import HubTelemetry
from guardrails.validators import FailResult, PassResult

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.guardrails import output as output_module
from code_raqeeb.guardrails.output import check_comments
from code_raqeeb.guardrails.secrets import REDACTED
from code_raqeeb.guardrails.text_guard import (
    MaxChars,
    NoSecrets,
    TextGuards,
    build_text_guards,
    clean_text,
)
from code_raqeeb.rules.models import ReviewRule, RuleSet
from tests.guardrails.fakes import AWS_KEY, PATH, make_comment, make_file

FILES: dict[str, ChangedFile] = {PATH: make_file("@@ -0,0 +1 @@\n+line", PATH)}
RULES = RuleSet(
    language="csharp",
    rules=(ReviewRule(id="BUG-01", category="bug", severity="high", title="T", description="D"),),
)


def no_secrets() -> NoSecrets:
    return NoSecrets(on_fail=OnFailAction.FIX)


def max_chars(limit: int) -> MaxChars:
    return MaxChars(max_chars=limit, on_fail=OnFailAction.FIX)


def test_no_secrets_passes_clean_text() -> None:
    assert isinstance(no_secrets().validate("Nothing to hide here", {}), PassResult)


def test_no_secrets_fails_with_a_redacted_fix_value() -> None:
    result = no_secrets().validate(f"first\nkey {AWS_KEY}", {})

    assert isinstance(result, FailResult)
    assert result.fix_value == f"first\nkey {REDACTED}"
    assert AWS_KEY not in result.error_message


def test_max_chars_passes_short_text() -> None:
    assert isinstance(max_chars(10).validate("short", {}), PassResult)


def test_max_chars_passes_text_of_exactly_the_limit() -> None:
    assert isinstance(max_chars(5).validate("exact", {}), PassResult)


def test_max_chars_fixes_long_text_with_an_ellipsis() -> None:
    result = max_chars(10).validate("x" * 11, {})

    assert isinstance(result, FailResult)
    assert result.fix_value == "x" * 9 + "…"


def test_max_chars_drops_whitespace_before_the_ellipsis() -> None:
    result = max_chars(10).validate("abcd      efgh", {})

    assert isinstance(result, FailResult)
    assert result.fix_value == "abcd…"


def test_clean_text_leaves_clean_short_text_alone() -> None:
    assert clean_text(build_text_guards(50), "fine") == "fine"


def test_clean_text_redacts_a_secret() -> None:
    assert clean_text(build_text_guards(50), f"key {AWS_KEY}") == f"key {REDACTED}"


def test_clean_text_cuts_long_text() -> None:
    assert clean_text(build_text_guards(50), "x" * 60) == "x" * 49 + "…"


def test_clean_text_applies_secrets_before_length() -> None:
    text = f"{'a' * 40} {AWS_KEY}{' tail' * 10}"

    cleaned = clean_text(build_text_guards(46), text)

    assert cleaned == f"{'a' * 40} {REDACTED[:4]}…"


async def test_clean_text_works_inside_a_running_event_loop() -> None:
    assert clean_text(build_text_guards(50), f"key {AWS_KEY}") == f"key {REDACTED}"


def test_clean_text_sends_no_telemetry() -> None:
    clean_text(build_text_guards(50), "fine")

    assert HubTelemetry()._enabled is False


def test_check_comments_uses_the_guards_for_message_and_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits: list[int] = []
    cleaned: list[str] = []

    def record_build(limit: int) -> TextGuards:
        limits.append(limit)
        return build_text_guards(limit)

    def record_clean(guards: TextGuards, text: str) -> str:
        cleaned.append(text)
        return clean_text(guards, text)

    monkeypatch.setattr(output_module, "build_text_guards", record_build)
    monkeypatch.setattr(output_module, "clean_text", record_clean)

    result = check_comments(
        [make_comment(message="The message", suggestion=f"Rotate {AWS_KEY}")],
        files=FILES,
        rules=RULES,
        max_per_file=8,
        max_chars=77,
    )

    assert limits == [77]
    assert cleaned == ["The message", f"Rotate {AWS_KEY}"]
    assert result.kept[0].suggestion == f"Rotate {REDACTED}"
