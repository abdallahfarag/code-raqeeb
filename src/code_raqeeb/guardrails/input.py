from code_raqeeb.github.errors import InvalidPrUrlError
from code_raqeeb.github.models import PrRef

MAX_URL_LENGTH = 300


class InputRejectedError(ValueError):
    pass


def check_pr_url(url: str) -> PrRef:
    cleaned = url.strip()
    if len(cleaned) > MAX_URL_LENGTH:
        raise InputRejectedError(f"The URL is longer than {MAX_URL_LENGTH} characters")
    try:
        return PrRef.from_url(cleaned)
    except InvalidPrUrlError as exc:
        raise InputRejectedError(str(exc)) from exc
