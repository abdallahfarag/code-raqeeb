from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, TypeAdapter

from code_raqeeb.github.errors import GitHubError, PrNotFoundError
from code_raqeeb.github.models import PrFile, PrRef, PullRequest
from code_raqeeb.review.models import FileReview

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "prs" / "02-pay-endpoint"
PR_URL = "https://github.com/abdallahfarag/code-raqeeb-demo/pull/2"
REPO = "abdallahfarag/code-raqeeb-demo"
BASE_SHA = "8e5724b7ff3a92ce0aeb4271c2e36e8de1ab40cf"
HEAD_SHA = "bbf43b0cbc83ea7f5da11eb2d8d8ff1c7a2c52c5"

PAYMENT_CLIENT_PATH = "src/Orders.Api/Services/PaymentClient.cs"
PAYMENT_CLIENT = """\
namespace Orders.Api.Services;

public sealed class PaymentClient
{
    public Task ChargeAsync(int orderId, decimal amount, CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}
"""
ORDER_PATH = "src/Orders.Api/Models/Order.cs"
ORDER = """\
namespace Orders.Api.Models;

public record Order(int Id, string CustomerName, decimal Total, DateTime CreatedAtUtc);
"""
BASE_TREE = {
    PAYMENT_CLIENT_PATH: PAYMENT_CLIENT,
    ORDER_PATH: ORDER,
    "src/Orders.Api/bin/Debug/Generated.cs": "class Generated { void Skipped() { } }\n",
    "README.md": "# Orders\n",
}

_FILE_LIST = TypeAdapter(list[PrFile])


class ScriptedChatModel(BaseChatModel):
    replies: list[AIMessage]
    review: FileReview = Field(default_factory=FileReview)
    calls: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])
    structured_calls: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])
    bound_tools: list[list[str]] = Field(default_factory=list[list[str]])

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools.append([cast(BaseTool, item).name for item in tools])
        return cast(Runnable[LanguageModelInput, AIMessage], self.model_copy())

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        async def respond(messages: list[BaseMessage]) -> FileReview:
            self.calls.append(list(messages))
            self.structured_calls.append(list(messages))
            return self.review

        return cast(
            Runnable[LanguageModelInput, dict[str, Any] | BaseModel], RunnableLambda(respond)
        )


class FakeGitHub:
    def __init__(
        self,
        *,
        pr_files: list[PrFile] | None = None,
        base_files: dict[str, str] | None = None,
        head_files: dict[str, str] | None = None,
        failures: dict[str, GitHubError] | None = None,
    ) -> None:
        self.pull_request = PullRequest.model_validate_json(
            (FIXTURE / "pr.json").read_text(encoding="utf-8")
        )
        self.pr_files = (
            _FILE_LIST.validate_json((FIXTURE / "files.json").read_text(encoding="utf-8"))
            if pr_files is None
            else pr_files
        )
        self.base_files = base_files if base_files is not None else BASE_TREE
        self.head_files = head_files if head_files is not None else {}
        self.failures = failures if failures is not None else {}
        self.tree_shas: list[str] = []
        self.content_reads: list[tuple[str, str]] = []

    async def get_pull_request(self, ref: PrRef) -> PullRequest:
        return self.pull_request

    async def list_pull_request_files(self, ref: PrRef) -> list[PrFile]:
        return self.pr_files

    async def list_tree(self, ref: PrRef, sha: str) -> list[str]:
        self.tree_shas.append(sha)
        return list(self.base_files)

    async def get_files_content(self, ref: PrRef, paths: Iterable[str], sha: str) -> dict[str, str]:
        return {path: await self.get_file_content(ref, path, sha) for path in paths}

    async def get_file_content(self, ref: PrRef, path: str, sha: str) -> str:
        self.content_reads.append((path, sha))
        failure = self.failures.get(path)
        if failure is not None:
            raise failure
        tree = self.head_files if sha == HEAD_SHA else self.base_files
        if path not in tree:
            raise PrNotFoundError(f"{path} is not in {sha}")
        return tree[path]
