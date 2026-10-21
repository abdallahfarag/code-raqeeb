from typing import Literal

import httpx
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from code_raqeeb.api.dependencies import HttpClientDep, SettingsDep

router = APIRouter(prefix="/health", tags=["health"])

DependencyStatus = Literal["up", "down"]


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: dict[str, DependencyStatus]


@router.get("/live")
async def live() -> LiveResponse:
    return LiveResponse()


@router.get("/ready", responses={503: {"model": ReadyResponse}})
async def ready(
    response: Response,
    settings: SettingsDep,
    client: HttpClientDep,
) -> ReadyResponse:
    dependencies: dict[str, DependencyStatus] = {
        "qdrant": await _check_qdrant(client, str(settings.qdrant_url)),
    }

    if all(state == "up" for state in dependencies.values()):
        return ReadyResponse(status="ready", dependencies=dependencies)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="not_ready", dependencies=dependencies)


async def _check_qdrant(client: httpx.AsyncClient, base_url: str) -> DependencyStatus:
    try:
        result = await client.get(f"{base_url.rstrip('/')}/readyz")
    except httpx.HTTPError:
        return "down"
    return "up" if result.is_success else "down"
