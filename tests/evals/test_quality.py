import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from code_raqeeb.config import get_settings
from code_raqeeb.review.service import ReviewService, build_review_service
from tests.evals.metrics import RunScore, failures, format_report, score_run, summarize
from tests.evals.models import ExpectedReview

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "prs"
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "latest.json"
RUNS = int(os.environ.get("EVAL_RUNS", "2"))

pytestmark = pytest.mark.eval

Case = tuple[str, ExpectedReview]


def load_cases() -> list[Case]:
    return [
        (
            folder.name,
            ExpectedReview.model_validate_json(
                (folder / "expected.json").read_text(encoding="utf-8")
            ),
        )
        for folder in sorted(FIXTURES_DIR.iterdir())
        if (folder / "expected.json").is_file()
    ]


async def rule_retrieval(service: ReviewService, cases: Sequence[Case]) -> tuple[int, int]:
    hits = total = 0
    for _, expected in cases:
        for finding in expected.findings:
            if finding.query is None:
                continue
            total += 1
            results = await service.find_rules(finding.query, 5)
            hits += any(hit.rule.id == finding.rule_id for hit in results)
    return hits, total


async def test_review_quality() -> None:
    cases = load_cases()
    assert cases, f"No expected.json files under {FIXTURES_DIR}"
    settings = get_settings()
    scores: list[RunScore] = []

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http:
        service = await build_review_service(settings, http)
        try:
            retrieval_hits, retrieval_total = await rule_retrieval(service, cases)
            for _ in range(RUNS):
                for name, expected in cases:
                    result = await service.run(expected.pr_url)
                    scores.append(
                        score_run(
                            name,
                            expected.findings,
                            result.comments,
                            dropped=result.dropped_comments,
                            file_errors=len(result.file_errors),
                            injection_expected=expected.injection,
                        )
                    )
        finally:
            await service.aclose()

    summary = summarize(
        scores, runs=RUNS, retrieval_hits=retrieval_hits, retrieval_total=retrieval_total
    )
    print("\n" + format_report(summary, scores))

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "date": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": settings.llm_model,
                "summary": asdict(summary),
                "runs": [asdict(score) for score in scores],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    problems = failures(summary)
    assert not problems, "; ".join(problems)
