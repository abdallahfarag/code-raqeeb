from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from code_raqeeb.config import Settings


def init_langfuse(settings: Settings) -> Langfuse | None:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=str(settings.langfuse_host),
    )


def tracing_callbacks(client: Langfuse | None) -> list[BaseCallbackHandler]:
    return [] if client is None else [CallbackHandler()]
