from functools import lru_cache
from pathlib import Path

import yaml

from code_raqeeb.rules.models import RuleSet

DEFAULT_RULES_PATH = Path(__file__).parent / "csharp.yaml"


def load_rules(path: Path) -> RuleSet:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RuleSet.model_validate(data)


@lru_cache
def get_rules() -> RuleSet:
    return load_rules(DEFAULT_RULES_PATH)
