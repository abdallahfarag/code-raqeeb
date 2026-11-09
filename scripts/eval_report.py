import json
import sys
from pathlib import Path
from typing import Any


def pct(value: float) -> str:
    return f"{value:.0%}"


def render(data: dict[str, Any]) -> str:
    summary = data["summary"]
    rows = [
        ("Recall", pct(summary["recall"]), "Planted bugs found (line within ±3)"),
        ("Rule accuracy", pct(summary["rule_accuracy"]), "Found bugs tagged with the right rule"),
        (
            "False positives on the clean PR",
            f"{summary['clean_false_positives']:g}",
            "Comments on the PR that has no bugs",
        ),
        (
            "Prompt injection reported",
            pct(summary["injection_reported"]),
            "The attack in PR 5 was reported, not obeyed",
        ),
        ("Rule retrieval", pct(summary["rule_retrieval"]), "RAG returned the right rule"),
        (
            "Extra comments per run",
            f"{summary['extra_per_run']:.1f}",
            "Comments not in the answers",
        ),
        (
            "Dropped per run",
            f"{summary['dropped_per_run']:.2f}",
            "Comments removed by the verifier",
        ),
        ("File errors", f"{summary['file_errors']}", "Files whose review failed"),
    ]
    lines = ["| Metric | Result | Meaning |", "| --- | --- | --- |"]
    lines += [f"| {name} | {value} | {meaning} |" for name, value, meaning in rows]
    lines.append("")
    lines.append(f"*`{data['model']}`, {summary['runs']} runs, {data['date'][:10]}.*")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/eval_report.py <results.json>")
    print(render(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
