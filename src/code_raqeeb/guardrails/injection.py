import re
from typing import Protocol

_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:the\s+)?"
        r"(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules)",
        r"\b(?:note|message|instructions?)\s+(?:to|for)\s+(?:the\s+)?"
        r"(?:ai|llm|bots?|(?:code\s+)?reviewers?)\b",
        r"\bdo\s+not\s+(?:report|flag|mention)\s+(?:any\s+)?(?:issues|problems|bugs|findings)",
        r"\breply\s+only\s+with\b",
        r"\byou\s+are\s+now\b",
        r"\b(?:system|developer)\s+prompt\b",
    )
)


def looks_like_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PATTERNS)


class InjectionScanner(Protocol):
    def is_injection(self, text: str) -> bool: ...
