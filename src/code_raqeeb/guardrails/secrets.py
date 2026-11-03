import re
from dataclasses import dataclass

REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class SecretPattern:
    kind: str
    regex: re.Pattern[str]
    group: int = 0


_PATTERNS = (
    SecretPattern(
        "GitHub token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"),
    ),
    SecretPattern("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    SecretPattern("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    SecretPattern("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    SecretPattern(
        "JWT",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    SecretPattern("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    SecretPattern(
        "password in a connection string",
        re.compile(r"(?i)[;\"']\s*(?:password|pwd)\s*=\s*([^;\"'\s]+)"),
        group=1,
    ),
    SecretPattern(
        "hardcoded credential",
        re.compile(
            r"(?i)\b\w*(?:api[_-]?key|secret|token|password|passwd)\w*\s*[:=]\s*[@$]?\"([^\"]{8,})\""
        ),
        group=1,
    ),
)


def redact_line(line: str) -> tuple[str, list[str]]:
    kinds: list[str] = []
    for pattern in _PATTERNS:
        line, found = _redact(pattern, line)
        if found:
            kinds.append(pattern.kind)
    return line, kinds


def redact_text(text: str) -> str:
    return "\n".join(redact_line(line)[0] for line in text.split("\n"))


def _redact(pattern: SecretPattern, line: str) -> tuple[str, bool]:
    found = False

    def replace(match: re.Match[str]) -> str:
        nonlocal found
        found = True
        if pattern.group == 0:
            return REDACTED
        whole = match.group(0)
        start, end = match.span(pattern.group)
        offset = match.start()
        return whole[: start - offset] + REDACTED + whole[end - offset :]

    return pattern.regex.sub(replace, line), found
