import runpy
from collections.abc import Callable
from typing import Any

import pytest
from fastmcp import FastMCP

from code_raqeeb.config import Settings
from code_raqeeb.mcp_server import app
from code_raqeeb.mcp_server.app import create_server, main

type HostSetter = Callable[[str], None]


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(self: FastMCP, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(FastMCP, "run", fake_run)
    return calls


@pytest.fixture
def set_host(monkeypatch: pytest.MonkeyPatch) -> HostSetter:
    def install(host: str) -> None:
        settings = Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            mcp_host=host,
            mcp_port=8123,
        )

        def fake_settings() -> Settings:
            return settings

        monkeypatch.setattr(app, "get_settings", fake_settings)

    return install


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com", ""])
def test_main_refuses_a_non_loopback_host_and_never_starts_the_server(
    set_host: HostSetter, run_calls: list[dict[str, Any]], host: str
) -> None:
    set_host(host)

    with pytest.raises(SystemExit) as caught:
        main()

    assert f"Refusing to listen on {host}" in str(caught.value)
    assert run_calls == []


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_main_runs_streamable_http_on_a_loopback_host(
    set_host: HostSetter, run_calls: list[dict[str, Any]], host: str
) -> None:
    set_host(host)

    main()

    assert run_calls == [{"transport": "http", "host": host, "port": 8123}]


def test_create_server_returns_the_named_server() -> None:
    server = create_server()

    assert isinstance(server, FastMCP)
    assert server.name == "Code Raqeeb"


def test_running_the_package_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def fake_main() -> None:
        calls.append(None)

    monkeypatch.setattr(app, "main", fake_main)

    runpy.run_module("code_raqeeb.mcp_server", run_name="__main__")

    assert len(calls) == 1
