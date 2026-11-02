from pathlib import Path

import pytest
from pydantic import TypeAdapter

from code_raqeeb.github.diff import ChangedFile, parse_patch
from code_raqeeb.github.models import PrFile, PrRef, PullRequest
from code_raqeeb.rules.loader import get_rules
from tests.evals.models import ExpectedFinding, ExpectedReview

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "prs"
FOLDERS = sorted(path for path in FIXTURES_DIR.iterdir() if path.is_dir())

_PR_FILES = TypeAdapter(list[PrFile])

every_folder = pytest.mark.parametrize("folder", FOLDERS, ids=[path.name for path in FOLDERS])


def load_expected(folder: Path) -> ExpectedReview:
    text = (folder / "expected.json").read_text(encoding="utf-8")
    return ExpectedReview.model_validate_json(text)


def load_pull_request(folder: Path) -> PullRequest:
    return PullRequest.model_validate_json((folder / "pr.json").read_text(encoding="utf-8"))


def load_files(folder: Path) -> dict[str, PrFile]:
    files = _PR_FILES.validate_json((folder / "files.json").read_text(encoding="utf-8"))
    return {file.filename: file for file in files}


def to_changed_file(file: PrFile) -> ChangedFile:
    return ChangedFile(
        path=file.filename,
        status=file.status,
        additions=file.additions,
        deletions=file.deletions,
        hunks=parse_patch(file.patch or ""),
    )


def describe(finding: ExpectedFinding) -> str:
    return f"{finding.rule_id} at {finding.path}:{finding.line}"


@every_folder
def test_expected_json_is_valid(folder: Path) -> None:
    assert isinstance(load_expected(folder), ExpectedReview)


@every_folder
def test_pr_json_is_valid(folder: Path) -> None:
    assert isinstance(load_pull_request(folder), PullRequest)


@every_folder
def test_files_json_is_valid(folder: Path) -> None:
    assert load_files(folder)


@every_folder
def test_finding_rule_ids_exist(folder: Path) -> None:
    rules = get_rules()

    unknown = [
        describe(finding)
        for finding in load_expected(folder).findings
        if rules.get(finding.rule_id) is None
    ]

    assert not unknown


@every_folder
def test_finding_categories_match_rules(folder: Path) -> None:
    rules = get_rules()

    mismatched: list[str] = []
    for finding in load_expected(folder).findings:
        rule = rules.get(finding.rule_id)
        if rule is not None and rule.category != finding.category:
            mismatched.append(f"{describe(finding)}: {finding.category} != {rule.category}")

    assert not mismatched


@every_folder
def test_finding_paths_have_patches(folder: Path) -> None:
    files = load_files(folder)

    missing: list[str] = []
    for finding in load_expected(folder).findings:
        file = files.get(finding.path)
        if file is None or file.patch is None:
            missing.append(describe(finding))

    assert not missing


@every_folder
def test_finding_lines_are_commentable(folder: Path) -> None:
    files = load_files(folder)

    not_commentable: list[str] = []
    for finding in load_expected(folder).findings:
        file = files.get(finding.path)
        if file is None or not to_changed_file(file).is_commentable(finding.line):
            not_commentable.append(describe(finding))

    assert not not_commentable


@every_folder
def test_pr_url_matches_pr_number(folder: Path) -> None:
    ref = PrRef.from_url(load_expected(folder).pr_url)

    assert ref.number == load_pull_request(folder).number


def test_at_least_one_fixture_has_injection() -> None:
    assert any(load_expected(folder).injection for folder in FOLDERS)


def test_at_least_one_fixture_has_no_findings() -> None:
    assert any(not load_expected(folder).findings for folder in FOLDERS)
