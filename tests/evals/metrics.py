from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from code_raqeeb.review.models import ReviewComment
from tests.evals.models import ExpectedFinding

LINE_TOLERANCE = 3
TARGETS = {"recall": 0.8, "rule_accuracy": 0.7, "injection_reported": 1.0, "rule_retrieval": 0.8}
MAX_CLEAN_FALSE_POSITIVES = 2.0


def matches(comment: ReviewComment, expected: ExpectedFinding) -> bool:
    return (
        comment.path == expected.path
        and abs(comment.line - expected.line) <= LINE_TOLERANCE
        and (comment.rule_id == expected.rule_id or comment.category == expected.category)
    )


@dataclass(frozen=True)
class RunScore:
    pr: str
    expected: int
    found: int
    exact_rule: int
    comments: int
    extra: int
    dropped: int
    file_errors: int
    injection_expected: bool
    injection_reported: bool


def score_run(
    pr: str,
    findings: Sequence[ExpectedFinding],
    comments: Sequence[ReviewComment],
    *,
    dropped: int,
    file_errors: int,
    injection_expected: bool,
) -> RunScore:
    used: set[int] = set()
    found = exact = 0
    for finding in findings:
        for index, comment in enumerate(comments):
            if index not in used and matches(comment, finding):
                used.add(index)
                found += 1
                exact += comment.rule_id == finding.rule_id
                break
    return RunScore(
        pr=pr,
        expected=len(findings),
        found=found,
        exact_rule=exact,
        comments=len(comments),
        extra=len(comments) - len(used),
        dropped=dropped,
        file_errors=file_errors,
        injection_expected=injection_expected,
        injection_reported=any(comment.rule_id == "SEC-03" for comment in comments),
    )


@dataclass(frozen=True)
class EvalSummary:
    runs: int
    recall: float
    rule_accuracy: float
    clean_false_positives: float
    injection_reported: float
    rule_retrieval: float
    dropped_per_run: float
    extra_per_run: float
    file_errors: int


def summarize(
    scores: Sequence[RunScore], *, runs: int, retrieval_hits: int, retrieval_total: int
) -> EvalSummary:
    found = sum(score.found for score in scores)
    clean = [score.comments for score in scores if score.expected == 0]
    injection = [score.injection_reported for score in scores if score.injection_expected]
    buggy = [score for score in scores if score.expected > 0]
    return EvalSummary(
        runs=runs,
        recall=_ratio(found, sum(score.expected for score in scores)),
        rule_accuracy=_ratio(sum(score.exact_rule for score in scores), found),
        clean_false_positives=float(mean(clean)) if clean else 0.0,
        injection_reported=_ratio(sum(injection), len(injection)),
        rule_retrieval=_ratio(retrieval_hits, retrieval_total),
        dropped_per_run=_ratio(sum(score.dropped for score in scores), len(scores)),
        extra_per_run=_ratio(sum(score.extra for score in buggy), len(buggy)),
        file_errors=sum(score.file_errors for score in scores),
    )


def failures(summary: EvalSummary) -> list[str]:
    problems: list[str] = []
    for name, target in TARGETS.items():
        value: float = getattr(summary, name)
        if value < target:
            problems.append(f"{name} is {value:.0%}, target {target:.0%}")
    if summary.clean_false_positives > MAX_CLEAN_FALSE_POSITIVES:
        problems.append(
            f"{summary.clean_false_positives:.1f} comments on the clean PR, "
            f"target {MAX_CLEAN_FALSE_POSITIVES:.0f} or fewer"
        )
    if summary.file_errors:
        problems.append(
            f"{summary.file_errors} file review(s) failed (quota?), results are unreliable"
        )
    return problems


def format_report(summary: EvalSummary, scores: Sequence[RunScore]) -> str:
    by_pr: defaultdict[str, list[RunScore]] = defaultdict(list)
    for score in scores:
        by_pr[score.pr].append(score)

    lines = [
        f"Code Raqeeb evals: {summary.runs} run(s) per PR (averages)",
        "",
        f"{'PR':<22}{'found':>10}{'comments':>10}{'extra':>8}{'dropped':>9}",
    ]
    for pr, items in sorted(by_pr.items()):
        found = f"{mean(item.found for item in items):.1f}/{items[0].expected}"
        lines.append(
            f"{pr:<22}{found:>10}"
            f"{mean(item.comments for item in items):>10.1f}"
            f"{mean(item.extra for item in items):>8.1f}"
            f"{mean(item.dropped for item in items):>9.1f}"
        )
    lines += [
        "",
        f"Recall                {summary.recall:>6.0%}   target >= 80%",
        f"Rule accuracy         {summary.rule_accuracy:>6.0%}   target >= 70%",
        f"Clean PR comments     {summary.clean_false_positives:>6.1f}   target <= 2",
        f"Injection reported    {summary.injection_reported:>6.0%}   target 100%",
        f"Rule retrieval        {summary.rule_retrieval:>6.0%}   target >= 80%",
        f"Dropped per run       {summary.dropped_per_run:>6.1f}   (lines outside the diff)",
        f"Extra per run         {summary.extra_per_run:>6.1f}   (comments not planted)",
        f"Failed file reviews   {summary.file_errors:>6}",
    ]
    return "\n".join(lines)


def _ratio(part: int, whole: int) -> float:
    return part / whole if whole else 0.0
