from langchain_core.tools import BaseTool, tool

from code_raqeeb.github.errors import GitHubError
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.guardrails.paths import safe_repo_path
from code_raqeeb.guardrails.secrets import redact_text
from code_raqeeb.helper.text import split_lines
from code_raqeeb.review.context import ReviewContext

_MAX_READ_LINES = 200
_MAX_TOOL_TEXT = 4000


def build_tools(ctx: ReviewContext, bundle: PullRequestBundle) -> list[BaseTool]:
    repo = bundle.ref.full_name

    @tool(
        description=(
            "Search the repository as it was before this pull request for code related to a "
            "query: a class, a method name, or a behavior. Returns paths, line ranges, and code."
        )
    )
    async def search_code(query: str) -> str:
        embedding = await ctx.embedder.embed_query(query)
        hits = await ctx.store.search_code(repo, bundle.base_sha, embedding, limit=4)
        if not hits:
            return "No matching code found."
        parts = [
            f"{hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line} "
            f"({hit.chunk.symbol})\n{redact_text(hit.chunk.text)}"
            for hit in hits
        ]
        return _clip("\n\n".join(parts))

    @tool(
        description=(
            "Find code review rules about a topic, for example 'HttpClient', 'async', or 'SQL'. "
            "Returns rule IDs with their category, severity, and description."
        )
    )
    async def get_rules(query: str) -> str:
        embedding = await ctx.embedder.embed_query(query)
        hits = await ctx.store.search_rules(embedding, limit=5)
        lines = [
            f"{hit.rule.id} ({hit.rule.category}, {hit.rule.severity}): "
            f"{hit.rule.title}. {hit.rule.description}"
            for hit in hits
        ]
        return "\n".join(lines) or "No rules found."

    @tool(
        description=(
            "Read lines of a file as it is in this pull request (after the change), "
            "to see the code around the diff."
        )
    )
    async def read_file(path: str, start_line: int = 1, end_line: int = 120) -> str:
        if start_line < 1 or end_line < start_line:
            return "Invalid line range."
        end_line = min(end_line, start_line + _MAX_READ_LINES - 1)
        safe_path = safe_repo_path(path)
        if safe_path is None:
            return "Path rejected: use a relative path inside the repository."
        try:
            content = await ctx.github.get_file_content(bundle.ref, safe_path, bundle.head_sha)
        except GitHubError as exc:
            return f"Could not read {safe_path}: {exc}"
        selected = split_lines(redact_text(content))[start_line - 1 : end_line]
        numbered = [
            f"{number:>6}  {text}" for number, text in enumerate(selected, start=start_line)
        ]
        return _clip("\n".join(numbered))

    return [search_code, get_rules, read_file]


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_TOOL_TEXT else text[:_MAX_TOOL_TEXT] + "\n[truncated]"
