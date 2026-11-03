# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false

from dataclasses import dataclass
from typing import Any

from guardrails import Guard, OnFailAction
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from code_raqeeb.guardrails.secrets import redact_text


@register_validator(name="code-raqeeb/no-secrets", data_type="string")
class NoSecrets(Validator):
    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        text = str(value)
        redacted = redact_text(text)
        if redacted == text:
            return PassResult()
        return FailResult(error_message="The text contains a secret.", fix_value=redacted)


@register_validator(name="code-raqeeb/max-chars", data_type="string")
class MaxChars(Validator):
    def __init__(self, max_chars: int, **kwargs: Any) -> None:
        super().__init__(max_chars=max_chars, **kwargs)
        self._max_chars = max_chars

    def _validate(self, value: Any, metadata: dict[str, Any]) -> ValidationResult:
        text = str(value)
        if len(text) <= self._max_chars:
            return PassResult()
        return FailResult(
            error_message=f"The text is longer than {self._max_chars} characters.",
            fix_value=text[: self._max_chars - 1].rstrip() + "…",
        )


@dataclass(frozen=True)
class TextGuards:
    secrets: Guard
    length: Guard


def build_text_guards(max_chars: int) -> TextGuards:
    return TextGuards(
        secrets=Guard().use(NoSecrets(on_fail=OnFailAction.FIX)),
        length=Guard().use(MaxChars(max_chars=max_chars, on_fail=OnFailAction.FIX)),
    )


def clean_text(guards: TextGuards, text: str) -> str:
    for guard in (guards.secrets, guards.length):
        outcome = guard.validate(text)
        if isinstance(outcome.validated_output, str):
            text = outcome.validated_output
    return text
