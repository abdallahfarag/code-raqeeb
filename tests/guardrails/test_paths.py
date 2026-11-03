import pytest

from code_raqeeb.guardrails.paths import safe_repo_path


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("/etc/passwd", id="absolute"),
        pytest.param("//etc/passwd", id="double-slash-absolute"),
        pytest.param("..", id="parent"),
        pytest.param("../x", id="leading-parent"),
        pytest.param("src/../../x", id="nested-parent"),
        pytest.param("src/..", id="trailing-parent"),
        pytest.param("src\\Foo.cs", id="backslash"),
        pytest.param("..\\x", id="backslash-parent"),
        pytest.param("src/Foo\x00.cs", id="nul"),
        pytest.param("", id="empty"),
        pytest.param("a" * 301, id="too-long"),
    ],
)
def test_unsafe_paths_are_rejected(path: str) -> None:
    assert safe_repo_path(path) is None


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        pytest.param("src/Foo.cs", "src/Foo.cs", id="unchanged"),
        pytest.param("Foo.cs", "Foo.cs", id="top-level"),
        pytest.param("./src/Foo.cs", "src/Foo.cs", id="leading-dot"),
        pytest.param("src/./Foo.cs", "src/Foo.cs", id="inner-dot"),
        pytest.param("src//Foo.cs", "src/Foo.cs", id="double-slash"),
        pytest.param("src/Foo/", "src/Foo", id="trailing-slash"),
        pytest.param("src/a..b/Foo.cs", "src/a..b/Foo.cs", id="dots-inside-a-name"),
    ],
)
def test_normal_paths_are_normalized(path: str, expected: str) -> None:
    assert safe_repo_path(path) == expected


def test_path_of_exactly_the_maximum_length_is_accepted() -> None:
    path = "a" * 300

    assert safe_repo_path(path) == path
