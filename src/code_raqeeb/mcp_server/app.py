import logging

from fastmcp import FastMCP

from code_raqeeb.config import get_settings
from code_raqeeb.mcp_server.tools import register_tools

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def create_server() -> FastMCP:
    server = FastMCP(
        "Code Raqeeb",
        instructions=(
            "Reviews public GitHub pull requests that change C# code. Every comment links to "
            "the exact line. All tools are read-only and never write to GitHub."
        ),
    )
    register_tools(server)
    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    if settings.mcp_host not in _LOOPBACK_HOSTS:
        raise SystemExit(
            f"Refusing to listen on {settings.mcp_host}: this server has no auth, "
            "so it only runs on localhost."
        )

    create_server().run(transport="http", host=settings.mcp_host, port=settings.mcp_port)
