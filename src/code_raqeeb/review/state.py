import operator
from typing import Annotated, NotRequired, TypedDict

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.rag.indexer import IndexReport
from code_raqeeb.review.models import ReviewComment


class ReviewState(TypedDict):
    pr_url: str
    comments: Annotated[list[ReviewComment], operator.add]
    file_errors: Annotated[list[str], operator.add]
    bundle: NotRequired[PullRequestBundle]
    index_report: NotRequired[IndexReport]
    verified_comments: NotRequired[list[ReviewComment]]
    dropped_comments: NotRequired[int]
    summary: NotRequired[str]
    reviewed_files: Annotated[list[str], operator.add]


class FileTask(TypedDict):
    bundle: PullRequestBundle
    file: ChangedFile
