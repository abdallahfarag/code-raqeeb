from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from code_raqeeb.mcp_server import dependencies
from code_raqeeb.mcp_server.dependencies import READ_ONLY, PrUrl, checked_url
from code_raqeeb.rag.models import CodeHit, RuleHit
from code_raqeeb.review.service import EXPECTED_ERRORS, ReviewResult


def register_tools(server: FastMCP) -> None:
    @server.tool(
        description=(
            "Review a public GitHub pull request that changes C# files. Returns verified "
            "comments, each with severity, category, rule ID, message, suggestion, and a "
            "permalink to the exact line."
        ),
        annotations=READ_ONLY,
    )
    async def review_pr(pr_url: PrUrl, ctx: Context) -> ReviewResult:
        url = checked_url(pr_url)
        service = await dependencies.get_service()
        total, reviewed = 1, 0
        async for event in service.stream(url):
            if event.name == "pr_loaded":
                total = max(1, len(event.data["files"]))
                await ctx.info(f"Reviewing {total} file(s): {event.data['title']}")
            elif event.name == "file_reviewed":
                reviewed += 1
                await ctx.report_progress(progress=reviewed, total=total)
            elif event.name == "error":
                raise ToolError(f"{event.data['type']}: {event.data['message']}")
            elif event.name == "done":
                return ReviewResult.model_validate(event.data)
        raise ToolError("The review ended without a result.")

    @server.tool(
        description=(
            "Search the code of a pull request's base commit (indexed automatically) for code "
            "related to a query: a class, a method name, or a behavior."
        ),
        annotations=READ_ONLY,
    )
    async def search_code(
        pr_url: PrUrl,
        query: Annotated[str, Field(min_length=2, max_length=200, description="What to look for")],
        limit: Annotated[int, Field(ge=1, le=10, description="How many results")] = 5,
    ) -> list[CodeHit]:
        url = checked_url(pr_url)
        service = await dependencies.get_service()
        try:
            return await service.search_code(url, query, limit)
        except EXPECTED_ERRORS as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(
        description=(
            "Find C# code review rules about a topic, for example 'HttpClient', 'async', or 'SQL'."
        ),
        annotations=READ_ONLY,
    )
    async def get_rules(
        query: Annotated[str, Field(min_length=2, max_length=200, description="The topic")],
        limit: Annotated[int, Field(ge=1, le=10, description="How many rules")] = 5,
    ) -> list[RuleHit]:
        service = await dependencies.get_service()
        return await service.find_rules(query, limit)
