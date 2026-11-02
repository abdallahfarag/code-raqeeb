import json
import subprocess
import sys

import pytest

from code_raqeeb.rag.chunker import chunk_csharp
from code_raqeeb.rag.models import ChunkKind, CodeChunk
from tests.rag.conftest import make_chunk

BLOCK_NAMESPACE = """\
namespace Shop.Orders
{
    public class OrderService
    {
        public void Place()
        {
        }
    }
}
"""

FILE_SCOPED_NAMESPACE = """\
namespace Shop.Orders;

public class OrderService
{
    public void Place()
    {
    }
}
"""

CLASS_WITH_MEMBERS = """\
using System;

namespace Shop;

public class Customer
{
    public Customer(string name)
    {
        Name = name;
    }

    public string Name { get; }

    public string Greet()
    {
        return $"Hello {Name}";
    }
}
"""

RECORD_WITHOUT_BODY = """\
namespace Shop;

public record Money(decimal Amount, string Currency);
"""

ENUM = """\
public enum Color
{
    Red,
    Green,
}
"""

STRUCT_WITH_ONLY_FIELDS = """\
public struct Point
{
    public int X;
}
"""

NESTED_CLASS = """\
class Outer
{
    void Own() { }

    class Inner
    {
        void Method() { }
    }
}
"""

NESTED_CLASS_WITHOUT_OUTER_MEMBERS = """\
class Outer
{
    class Inner
    {
        void Method() { }
    }
}
"""

ATTRIBUTES = """\
class Controller
{
    [HttpGet]
    [Authorize]
    public void Get()
    {
    }
}
"""

DESTRUCTOR = """\
class Handle
{
    ~Handle() { }
}
"""

OPERATORS = """\
class Money
{
    public static Money operator +(Money a, Money b) => a;
    public static Money operator -(Money a, Money b) => a;
}
"""

CONVERSION_OPERATOR = """\
class Money
{
    public void Round() { }
    public static implicit operator decimal(Money money) => 0m;
}
"""

CONDITIONAL_METHOD = """\
class Service
{
#if DEBUG
    void DumpState() { }
#endif
    void Run() { }
}
"""


CONDITIONAL_BRANCHES = """\
class Service
{
#if DEBUG
    void InIf() { }
#elif TRACE
    void InElif() { }
#else
    void InElse() { }
#endif
}
"""


def chunk(source: str, *, max_lines: int = 100) -> list[CodeChunk]:
    return chunk_csharp("src/File.cs", source, repo="octo/hello", sha="sha1", max_lines=max_lines)


def by_symbol(source: str) -> dict[str, CodeChunk]:
    return {c.symbol: c for c in chunk(source)}


def statements(count: int) -> str:
    return "".join(f'Console.WriteLine("{n}");\n' for n in range(1, count + 1))


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(BLOCK_NAMESPACE, id="block"),
        pytest.param(FILE_SCOPED_NAMESPACE, id="file-scoped"),
    ],
)
def test_namespace_is_not_part_of_symbol(source: str) -> None:
    assert [c.symbol for c in chunk(source)] == ["OrderService.Place"]


@pytest.mark.parametrize(
    ("source", "start_line", "end_line"),
    [
        pytest.param(BLOCK_NAMESPACE, 5, 7, id="block"),
        pytest.param(FILE_SCOPED_NAMESPACE, 5, 7, id="file-scoped"),
    ],
)
def test_namespaced_member_lines(source: str, start_line: int, end_line: int) -> None:
    [method] = chunk(source)

    assert (method.start_line, method.end_line) == (start_line, end_line)


def test_class_yields_one_chunk_per_member_with_kind() -> None:
    kinds: dict[str, ChunkKind] = {c.symbol: c.kind for c in chunk(CLASS_WITH_MEMBERS)}

    assert kinds == {
        "Customer.Customer": "constructor",
        "Customer.Name": "property",
        "Customer.Greet": "method",
    }


@pytest.mark.parametrize(
    ("symbol", "start_line", "end_line"),
    [
        pytest.param("Customer.Customer", 7, 10, id="constructor"),
        pytest.param("Customer.Name", 12, 12, id="property"),
        pytest.param("Customer.Greet", 14, 17, id="method"),
    ],
)
def test_exact_one_based_lines(symbol: str, start_line: int, end_line: int) -> None:
    member = by_symbol(CLASS_WITH_MEMBERS)[symbol]

    assert (member.start_line, member.end_line) == (start_line, end_line)


def test_text_is_the_source_of_the_member() -> None:
    greet = by_symbol(CLASS_WITH_MEMBERS)["Customer.Greet"]

    assert greet.text.splitlines() == [
        "    public string Greet()",
        "    {",
        '        return $"Hello {Name}";',
        "    }",
    ]


def test_chunk_carries_repo_sha_and_path() -> None:
    [method] = chunk(BLOCK_NAMESPACE)

    assert (method.repo, method.sha, method.path) == ("octo/hello", "sha1", "src/File.cs")


def test_embedding_text_starts_with_path_and_symbol() -> None:
    item = make_chunk("Foo.Bar", path="src/Foo.cs", text="void Bar() { }")

    assert item.embedding_text == "// src/Foo.cs\n// Foo.Bar\nvoid Bar() { }"


@pytest.mark.parametrize(
    ("source", "symbol", "start_line", "end_line"),
    [
        pytest.param(RECORD_WITHOUT_BODY, "Money", 3, 3, id="record-no-body"),
        pytest.param(ENUM, "Color", 1, 5, id="enum"),
        pytest.param(STRUCT_WITH_ONLY_FIELDS, "Point", 1, 4, id="struct-only-fields"),
    ],
)
def test_type_without_members_is_one_type_chunk(
    source: str, symbol: str, start_line: int, end_line: int
) -> None:
    [only] = chunk(source)

    assert only.kind == "type"
    assert only.symbol == symbol
    assert (only.start_line, only.end_line) == (start_line, end_line)


def test_nested_class_symbol_includes_every_enclosing_type() -> None:
    assert [c.symbol for c in chunk(NESTED_CLASS)] == ["Outer.Own", "Outer.Inner.Method"]


def test_nested_method_inside_class_with_no_own_members_is_found() -> None:
    method = by_symbol(NESTED_CLASS_WITHOUT_OUTER_MEMBERS)["Outer.Inner.Method"]

    assert (method.kind, method.start_line, method.end_line) == ("method", 5, 5)


def test_attributes_above_method_are_inside_the_chunk() -> None:
    [method] = chunk(ATTRIBUTES)

    assert (method.start_line, method.end_line) == (3, 7)
    assert method.text.splitlines()[:3] == [
        "    [HttpGet]",
        "    [Authorize]",
        "    public void Get()",
    ]


def test_destructor_is_a_method_chunk() -> None:
    [destructor] = chunk(DESTRUCTOR)

    assert destructor.kind == "method"
    assert destructor.text == "    ~Handle() { }"


def test_top_level_statements_become_60_line_file_chunks() -> None:
    source = statements(130)
    lines = source.splitlines()

    chunks = chunk(source)

    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (61, 120), (121, 130)]
    assert {c.kind for c in chunks} == {"file"}
    for item in chunks:
        assert item.text == "\n".join(lines[item.start_line - 1 : item.end_line])


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        pytest.param(60, [(1, 60)], id="exactly-one-window"),
        pytest.param(61, [(1, 60), (61, 61)], id="one-line-over"),
    ],
)
def test_file_window_boundaries(count: int, expected: list[tuple[int, int]]) -> None:
    assert [(c.start_line, c.end_line) for c in chunk(statements(count))] == expected


def test_file_chunks_ignore_max_lines() -> None:
    [only] = chunk(statements(40), max_lines=5)

    assert (only.start_line, only.end_line) == (1, 40)


@pytest.mark.parametrize(
    ("max_lines", "end_line"),
    [
        pytest.param(3, 5, id="truncated"),
        pytest.param(4, 6, id="exactly-max-lines"),
        pytest.param(5, 6, id="shorter-than-max-lines"),
    ],
)
def test_long_member_is_truncated_to_max_lines(max_lines: int, end_line: int) -> None:
    lines = "class C\n{\n    void M()\n    {\n        A();\n    }\n}\n".splitlines()

    [method] = chunk("\n".join(lines), max_lines=max_lines)

    assert (method.start_line, method.end_line) == (3, end_line)
    assert method.text == "\n".join(lines[2:end_line])


def test_empty_file_has_no_chunks() -> None:
    assert chunk("") == []


def test_crlf_line_endings_keep_lines_and_strip_carriage_returns() -> None:
    [method] = chunk(ATTRIBUTES.replace("\n", "\r\n"))

    assert (method.start_line, method.end_line) == (3, 7)
    assert "\r" not in method.text


# str.splitlines also splits on these, but tree-sitter rows only count \n
@pytest.mark.parametrize(
    "separator",
    [pytest.param("\x0c", id="form-feed"), pytest.param("\N{LINE SEPARATOR}", id="line-separator")],
)
def test_text_stays_aligned_with_lines_when_a_comment_holds_a_unicode_line_break(
    separator: str,
) -> None:
    source = f"class C\n{{\n    // a{separator}b\n    void M() {{ }}\n}}\n"

    [method] = chunk(source)

    assert method.start_line == 4
    assert method.text == "    void M() { }"


def test_operators_have_distinct_named_symbols() -> None:
    assert [c.symbol for c in chunk(OPERATORS)] == ["Money.operator +", "Money.operator -"]


def test_conversion_operator_is_chunked() -> None:
    conversion = by_symbol(CONVERSION_OPERATOR)["Money.operator decimal"]

    assert conversion.kind == "method"
    assert "implicit operator decimal" in conversion.text


def test_method_inside_preprocessor_block_is_chunked() -> None:
    assert {c.symbol for c in chunk(CONDITIONAL_METHOD)} == {"Service.DumpState", "Service.Run"}


def test_methods_in_every_preprocessor_branch_are_chunked() -> None:
    assert {c.symbol for c in chunk(CONDITIONAL_BRANCHES)} == {
        "Service.InIf",
        "Service.InElif",
        "Service.InElse",
    }


def test_class_with_only_conditional_members_has_no_type_chunk() -> None:
    assert {c.kind for c in chunk(CONDITIONAL_BRANCHES)} == {"method"}


# tree-sitter 0.26.0 over-frees Point rows above 256, CPython's cached small ints. That corrupts
# the heap, so these run in a subprocess: in-process they would take the whole test run down.
_LARGE_CLASS = r"""
methods = "".join(f"    public int M{n}() {{ return {n}; }}\n" for n in range(400))
source = "class Service\n{\n" + methods + "}\n"
"""
_LINES_PAST_256 = (
    "import json\n"
    "from code_raqeeb.rag.chunker import chunk_csharp\n"
    + _LARGE_CLASS
    + r"""
chunks = chunk_csharp("F.cs", source, repo="r", sha="s", max_lines=100)
print(json.dumps({c.symbol: [c.start_line, c.end_line] for c in chunks}))
"""
)
_REPEATED_LARGE_FILES = (
    "from code_raqeeb.rag.chunker import chunk_csharp\n"
    + _LARGE_CLASS
    + r"""
for _ in range(30):
    chunk_csharp("F.cs", source, repo="r", sha="s", max_lines=100)
    [str(n) * 3 for n in range(5000)]
"""
)


def run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False
    )


def test_line_numbers_are_correct_past_line_256() -> None:
    result = run_isolated(_LINES_PAST_256)

    assert result.returncode == 0, result.stderr[-500:]
    expected = {f"Service.M{n}": [n + 3, n + 3] for n in range(400)}
    assert json.loads(result.stdout) == expected


def test_chunking_large_files_repeatedly_does_not_crash_the_process() -> None:
    result = run_isolated(_REPEATED_LARGE_FILES)

    assert result.returncode == 0, result.stderr[-500:]


def test_non_ascii_text_does_not_shift_line_numbers() -> None:
    [method] = chunk('class C\n{\n    // تعليق ✓\n    string S() => "é";\n}\n')

    assert (method.start_line, method.end_line) == (4, 4)
    assert method.text == '    string S() => "é";'
