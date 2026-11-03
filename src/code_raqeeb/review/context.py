from dataclasses import dataclass

from code_raqeeb.config import Settings
from code_raqeeb.github.client import GitHubClient
from code_raqeeb.guardrails.injection import InjectionScanner
from code_raqeeb.rag.embeddings import Embedder
from code_raqeeb.rag.store import QdrantStore
from code_raqeeb.review.reviewer import FileReviewer


@dataclass(frozen=True)
class ReviewContext:
    settings: Settings
    github: GitHubClient
    store: QdrantStore
    embedder: Embedder
    reviewer: FileReviewer
    injection_scanner: InjectionScanner | None = None
