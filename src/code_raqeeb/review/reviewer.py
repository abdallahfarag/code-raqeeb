from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from code_raqeeb.github.diff import ChangedFile
from code_raqeeb.github.service import PullRequestBundle
from code_raqeeb.review.models import DraftComment, FileReview
from code_raqeeb.review.prompts import SYSTEM_PROMPT, build_file_prompt


class FileReviewer:
    def __init__(self, model: BaseChatModel, *, max_tool_rounds: int) -> None:
        self._model = model
        self._max_tool_rounds = max_tool_rounds

    async def review(
        self, bundle: PullRequestBundle, file: ChangedFile, tools: Sequence[BaseTool]
    ) -> list[DraftComment]:
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_file_prompt(bundle, file)),
        ]
        await self._run_tools(messages, tools)

        if isinstance(messages[-1], AIMessage):
            messages.pop()

        result = await self._model.with_structured_output(FileReview).ainvoke(messages)
        return FileReview.model_validate(result).comments

    async def _run_tools(self, messages: list[BaseMessage], tools: Sequence[BaseTool]) -> None:
        if self._max_tool_rounds == 0 or not tools:
            return
        model_with_tools = self._model.bind_tools(tools)
        tools_by_name = {available.name: available for available in tools}

        for _ in range(self._max_tool_rounds):
            reply = await model_with_tools.ainvoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                return
            for call in reply.tool_calls:
                selected = tools_by_name.get(call["name"])
                if selected is None:
                    output = f"Unknown tool: {call['name']}"
                else:
                    output = await selected.ainvoke(call["args"])
                messages.append(
                    ToolMessage(content=str(output), tool_call_id=call["id"] or call["name"])
                )
