import pytest
from pydantic import ValidationError

from code_raqeeb.github.errors import InvalidPrUrlError
from code_raqeeb.github.models import PrRef


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://github.com/octo/hello/pull/7", id="plain"),
        pytest.param("https://github.com/octo/hello/pull/7/", id="trailing-slash"),
        pytest.param("https://github.com/octo/hello/pull/7/files", id="files-tab"),
        pytest.param("  https://github.com/octo/hello/pull/7\n", id="surrounding-whitespace"),
    ],
)
def test_from_url_accepts_pr_urls(url: str) -> None:
    assert PrRef.from_url(url) == PrRef(owner="octo", repo="hello", number=7)


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://github.com/octo/hello/issues/7", id="issues-url"),
        pytest.param("https://gitlab.com/octo/hello/pull/7", id="gitlab"),
        pytest.param("https://github.com/octo/hello/pull/", id="missing-number"),
        pytest.param("https://github.com/octo/hello/pull/0", id="number-zero"),
        pytest.param("https://github.com/octo/hello/pull/7/files/extra", id="extra-path"),
    ],
)
def test_from_url_rejects_non_pr_urls(url: str) -> None:
    with pytest.raises(InvalidPrUrlError) as exc_info:
        PrRef.from_url(url)

    assert exc_info.value.url == url


def test_full_name() -> None:
    assert PrRef(owner="octo", repo="hello.world", number=1).full_name == "octo/hello.world"


def test_pr_ref_is_frozen() -> None:
    ref = PrRef(owner="octo", repo="hello", number=7)

    with pytest.raises(ValidationError):
        ref.number = 8  # pyright: ignore[reportAttributeAccessIssue]
