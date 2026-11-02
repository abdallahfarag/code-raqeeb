from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from code_raqeeb.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.gemini_api_key is None:
        raise ValueError("GEMINI_API_KEY is not set in .env")
    limiter = InMemoryRateLimiter(
        requests_per_second=settings.llm_requests_per_minute / 60,
        check_every_n_seconds=0.2,
        max_bucket_size=1,
    )
    return init_chat_model(
        settings.llm_model,
        model_provider="google_genai",
        api_key=settings.gemini_api_key.get_secret_value(),
        max_retries=settings.llm_max_retries,
        rate_limiter=limiter,
    )
