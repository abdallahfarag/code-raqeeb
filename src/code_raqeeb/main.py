from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from code_raqeeb.api.health import router as health_router
from code_raqeeb.api.reviews import router as reviews_router
from code_raqeeb.config import get_settings
from code_raqeeb.review.service import build_review_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        app.state.http_client = client
        service = await build_review_service(settings, client)
        app.state.review_service = service
        try:
            yield
        finally:
            await service.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(reviews_router)
    return app


app = create_app()
