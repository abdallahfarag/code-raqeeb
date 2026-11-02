from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["bug", "security", "performance", "style"]
Severity = Literal["high", "medium", "low"]


class ReviewRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[A-Z]{3}-\d{2}$")
    category: Category
    severity: Severity
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RuleSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: Literal["csharp"]
    rules: tuple[ReviewRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [rule.id for rule in self.rules]
        duplicates = sorted({rule_id for rule_id in ids if ids.count(rule_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate rule IDs: {duplicates}")
        return self

    def get(self, rule_id: str) -> ReviewRule | None:
        return next((rule for rule in self.rules if rule.id == rule_id), None)
