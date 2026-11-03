from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, field_validator

from code_raqeeb.github.errors import (
    GitHubError,
    PrivateRepoError,
    PrNotFoundError,
    PrTooLargeError,
    RateLimitError,
)
from code_raqeeb.guardrails.input import check_pr_url
from code_raqeeb.rag.indexer import RepoTooLargeError
from code_raqeeb.review.service import ReviewResult, ReviewService

router = APIRouter(prefix="/reviews", tags=["reviews"])


class ReviewRequest(BaseModel):
    pr_url: str

    @field_validator("pr_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        check_pr_url(value)
        return value.strip()


def get_review_service(request: Request) -> ReviewService:
    service: ReviewService = request.app.state.review_service
    return service


ReviewServiceDep = Annotated[ReviewService, Depends(get_review_service)]


@router.post("", response_class=EventSourceResponse, summary="Review a PR, streaming progress")
async def review_stream(
    body: ReviewRequest, service: ReviewServiceDep
) -> AsyncIterable[ServerSentEvent]:
    async for event in service.stream(body.pr_url):
        yield ServerSentEvent(event=event.name, data=event.data)


@router.post("/sync", summary="Review a PR and return the whole result")
async def review_sync(body: ReviewRequest, service: ReviewServiceDep) -> ReviewResult:
    try:
        return await service.run(body.pr_url)
    except (GitHubError, RepoTooLargeError) as exc:
        raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc


def _status_for(exc: Exception) -> int:
    if isinstance(exc, PrNotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(exc, PrivateRepoError):
        return status.HTTP_403_FORBIDDEN
    if isinstance(exc, PrTooLargeError | RepoTooLargeError):
        return 413
    if isinstance(exc, RateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS
    return status.HTTP_502_BAD_GATEWAY
