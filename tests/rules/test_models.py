import pytest
from pydantic import ValidationError

from code_raqeeb.rules.models import ReviewRule, RuleSet


def rule_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "SEC-01",
        "category": "security",
        "severity": "high",
        "title": "SQL built from strings",
        "description": "SQL text built with interpolation of input.",
    }
    return data | overrides


def rule_set(*rules: dict[str, object]) -> RuleSet:
    return RuleSet.model_validate({"language": "csharp", "rules": rules})


def test_valid_rule() -> None:
    rule = ReviewRule.model_validate(rule_data())

    assert rule.id == "SEC-01"
    assert rule.category == "security"
    assert rule.severity == "high"
    assert rule.title == "SQL built from strings"
    assert rule.description == "SQL text built with interpolation of input."


@pytest.mark.parametrize(
    "rule_id",
    [
        pytest.param("sec-01", id="lowercase"),
        pytest.param("SE-01", id="too-few-letters"),
        pytest.param("SECU-01", id="too-many-letters"),
        pytest.param("SEC-1", id="too-few-digits"),
        pytest.param("SEC-001", id="too-many-digits"),
        pytest.param("SEC01", id="no-dash"),
    ],
)
def test_invalid_rule_id_is_rejected(rule_id: str) -> None:
    with pytest.raises(ValidationError):
        ReviewRule.model_validate(rule_data(id=rule_id))


def test_unknown_category_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewRule.model_validate(rule_data(category="correctness"))


def test_unknown_severity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ReviewRule.model_validate(rule_data(severity="critical"))


def test_duplicate_rule_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate rule IDs"):
        rule_set(rule_data(), rule_data(title="Same ID, different title"))


def test_get_returns_rule_when_found() -> None:
    rules = rule_set(rule_data(), rule_data(id="BUG-01", category="bug"))

    found = rules.get("BUG-01")

    assert found is not None
    assert found.id == "BUG-01"
    assert found.category == "bug"


def test_get_returns_none_when_not_found() -> None:
    assert rule_set(rule_data()).get("BUG-99") is None
