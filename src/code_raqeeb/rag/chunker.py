from functools import lru_cache

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from code_raqeeb.rag.models import ChunkKind, CodeChunk

_TYPE_NODES = frozenset(
    {
        "class_declaration",
        "struct_declaration",
        "record_declaration",
        "interface_declaration",
        "enum_declaration",
    }
)
_MEMBER_NODES: dict[str, ChunkKind] = {
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "property_declaration": "property",
    "operator_declaration": "method",
    "conversion_operator_declaration": "method",
    "destructor_declaration": "method",
}
_OPERATOR_NAME_FIELDS = {
    "operator_declaration": "operator",
    "conversion_operator_declaration": "type",
}
_PREPROC_BLOCKS = frozenset({"preproc_if", "preproc_elif", "preproc_else"})
_WINDOW_LINES = 60


@lru_cache
def _parser() -> Parser:
    return get_parser("csharp")


def chunk_csharp(path: str, source: str, *, repo: str, sha: str, max_lines: int) -> list[CodeChunk]:
    lines = _split_lines(source)
    tree = _parser().parse(source.encode("utf-8"))
    chunks: list[CodeChunk] = []

    def make(node: Node, symbol: str, kind: ChunkKind) -> CodeChunk:
        start = node.start_point.row + 1
        end = min(node.end_point.row + 1, start + max_lines - 1)
        return CodeChunk(
            repo=repo,
            sha=sha,
            path=path,
            symbol=symbol,
            kind=kind,
            start_line=start,
            end_line=end,
            text="\n".join(lines[start - 1 : end]),
        )

    def visit(node: Node, type_path: tuple[str, ...]) -> None:
        for child in node.named_children:
            if child.type in _TYPE_NODES:
                visit_type(child, type_path)
            elif child.type not in _MEMBER_NODES:
                visit(child, type_path)

    def visit_type(node: Node, type_path: tuple[str, ...]) -> None:
        name_path = (*type_path, _name(node))
        body = node.child_by_field_name("body")
        members = _members(body) if body is not None else []
        for member in members:
            symbol = ".".join((*name_path, _name(member)))
            chunks.append(make(member, symbol, _MEMBER_NODES[member.type]))
        if not members:
            chunks.append(make(node, ".".join(name_path), "type"))
        if body is not None:
            visit(body, name_path)

    visit(tree.root_node, ())

    if not chunks:
        for start in range(1, len(lines) + 1, _WINDOW_LINES):
            end = min(start + _WINDOW_LINES - 1, len(lines))
            chunks.append(
                CodeChunk(
                    repo=repo,
                    sha=sha,
                    path=path,
                    symbol=f"{path}:{start}",
                    kind="file",
                    start_line=start,
                    end_line=end,
                    text="\n".join(lines[start - 1 : end]),
                )
            )
    return chunks


def _split_lines(source: str) -> list[str]:
    lines = [line.removesuffix("\r") for line in source.split("\n")]
    return lines[:-1] if lines[-1] == "" else lines


def _members(node: Node) -> list[Node]:
    members: list[Node] = []
    for child in node.named_children:
        if child.type in _MEMBER_NODES:
            members.append(child)
        elif child.type in _PREPROC_BLOCKS:
            members.extend(_members(child))
    return members


def _name(node: Node) -> str:
    operator_field = _OPERATOR_NAME_FIELDS.get(node.type)
    if operator_field is not None:
        return f"operator {_field_text(node, operator_field)}"
    return _field_text(node, "name")


def _field_text(node: Node, field: str) -> str:
    child = node.child_by_field_name(field)
    if child is None or child.text is None:
        return "<anonymous>"
    return child.text.decode("utf-8")
