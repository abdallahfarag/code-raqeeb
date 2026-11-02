import pytest

from code_raqeeb.helper.text import split_lines


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", [], id="empty"),
        pytest.param("a", ["a"], id="no-newline"),
        pytest.param("a\nb\n", ["a", "b"], id="trailing-newline"),
        pytest.param("a\nb", ["a", "b"], id="no-trailing-newline"),
        pytest.param("a\r\nb\r\n", ["a", "b"], id="crlf"),
        pytest.param("\n\n", ["", ""], id="blank-lines"),
        pytest.param("a\x0cb\nc", ["a\x0cb", "c"], id="form-feed-stays-in-line"),
        pytest.param(
            "a\N{LINE SEPARATOR}b\nc", ["a\N{LINE SEPARATOR}b", "c"], id="line-separator-stays"
        ),
    ],
)
def test_split_lines(text: str, expected: list[str]) -> None:
    assert split_lines(text) == expected
