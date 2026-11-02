from pydantic import BaseModel, ConfigDict, Field

from code_raqeeb.rules.models import Category


class ExpectedFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    line: int = Field(gt=0)
    category: Category
    rule_id: str


class ExpectedReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    pr_url: str
    injection: bool = False
    findings: tuple[ExpectedFinding, ...] = ()
