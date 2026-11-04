import asyncio
from typing import Annotated

import httpx
from fastmcp.exceptions import ToolError
from pydantic import Field

from code_raqeeb.config import get_settings
from code_raqeeb.guardrails.input import InputRejectedError, check_pr_url
from code_raqeeb.review.service import ReviewService, build_review_service

READ_ONLY = {"readOnlyHint": True, "openWorldHint": True}

PrUrl = Annotated[
    str,
    Field(
        description="A public GitHub pull request URL: https://github.com/{owner}/{repo}/pull/{n}"
    ),
]

_service: ReviewService | None = None
_service_lock = asyncio.Lock()


async def get_service() -> ReviewService:
    global _service
    async with _service_lock:
        if _service is None:
            settings = get_settings()
            http = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
            _service = await build_review_service(settings, http)
        return _service


def checked_url(pr_url: str) -> str:
    try:
        check_pr_url(pr_url)
    except InputRejectedError as exc:
        raise ToolError(str(exc)) from exc
    return pr_url.strip()
