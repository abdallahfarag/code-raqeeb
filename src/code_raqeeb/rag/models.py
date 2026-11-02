from typing import Literal

from pydantic import BaseModel, ConfigDict

from code_raqeeb.rules.models import ReviewRule

ChunkKind = Literal["type", "method", "constructor", "property", "file"]


class CodeChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo: str
    sha: str
    path: str
    symbol: str
    kind: ChunkKind
    start_line: int
    end_line: int
    text: str

    @property
    def embedding_text(self) -> str:
        return f"// {self.path}\n// {self.symbol}\n{self.text}"


class CodeHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: CodeChunk
    score: float


class RuleHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: ReviewRule
    score: float
