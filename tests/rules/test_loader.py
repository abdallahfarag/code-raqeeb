from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from code_raqeeb.rules.loader import DEFAULT_RULES_PATH, get_rules, load_rules
from code_raqeeb.rules.models import Category, Severity

DUPLICATE_ID_YAML = """\
language: csharp
rules:
  - id: SEC-01
    category: security
    severity: high
    title: First
    description: First rule.
  - id: SEC-01
    category: bug
    severity: low
    title: Second
    description: Second rule with the same ID.
"""


def test_real_rules_file_loads() -> None:
    rules = load_rules(DEFAULT_RULES_PATH)

    assert rules.language == "csharp"
    assert rules.rules


def test_real_rule_ids_are_unique() -> None:
    ids = [rule.id for rule in load_rules(DEFAULT_RULES_PATH).rules]

    assert len(ids) == len(set(ids))


def test_real_rule_categories_are_valid() -> None:
    valid = set(get_args(Category))

    for rule in load_rules(DEFAULT_RULES_PATH).rules:
        assert rule.category in valid, rule.id


def test_real_rule_severities_are_valid() -> None:
    valid = set(get_args(Severity))

    for rule in load_rules(DEFAULT_RULES_PATH).rules:
        assert rule.severity in valid, rule.id


def test_load_rules_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(DUPLICATE_ID_YAML, encoding="utf-8")

    with pytest.raises(ValidationError, match="Duplicate rule IDs"):
        load_rules(path)


def test_get_rules_returns_same_instance() -> None:
    first = get_rules()
    second = get_rules()

    assert first is second
