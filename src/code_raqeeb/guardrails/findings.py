import re

from code_raqeeb.github.diff import ChangedFile, added_lines_with_text
from code_raqeeb.guardrails.injection import InjectionScanner, looks_like_injection
from code_raqeeb.guardrails.secrets import redact_line
from code_raqeeb.review.models import ReviewComment, permalink

_PROSE = re.compile(r'//|/\*|^\s*\*|"[^"]{20,}"')


def scan_added_lines(
    file: ChangedFile,
    *,
    repo: str,
    head_sha: str,
    injection_scanner: InjectionScanner | None = None,
) -> list[ReviewComment]:
    findings: list[ReviewComment] = []
    for line, text in added_lines_with_text(file.patch):
        _, kinds = redact_line(text)
        if kinds:
            findings.append(
                ReviewComment(
                    path=file.path,
                    line=line,
                    severity="high",
                    category="security",
                    rule_id="SEC-02",
                    message=(
                        f"Possible hardcoded secret ({', '.join(kinds)}). Move it to configuration "
                        "or a secret store, and rotate it if it was ever real."
                    ),
                    suggestion="Read it from IConfiguration, user secrets, or a key vault.",
                    permalink=permalink(repo, head_sha, file.path, line),
                )
            )
        if looks_like_injection(text) or _ml_flags(injection_scanner, text):
            findings.append(
                ReviewComment(
                    path=file.path,
                    line=line,
                    severity="high",
                    category="security",
                    rule_id="SEC-03",
                    message=(
                        "This text tries to give instructions to AI reviewers (prompt injection). "
                        "It was ignored for this review; remove it."
                    ),
                    suggestion=None,
                    permalink=permalink(repo, head_sha, file.path, line),
                )
            )
    return findings


def _ml_flags(scanner: InjectionScanner | None, text: str) -> bool:
    return scanner is not None and _PROSE.search(text) is not None and scanner.is_injection(text)
