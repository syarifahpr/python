"""LangChain `BaseChatModel` adapter around `FlowiseClient`.

uqlm's scorers (`BlackBoxUQ`, `WhiteBoxUQ`, `LLMPanel`, ...) all take an
`llm` and, deep in `uqlm.utils.response_generator.ResponseGenerator`,
require it to actually be an instance of
`langchain_core.language_models.chat_models.BaseChatModel`:

    assert isinstance(self.llm, BaseChatModel), "..."
    ...
    result = await self.llm.ainvoke(messages)

Flowise's REST API (`POST {base_url}/api/v1/prediction/{chatflow_id}` with
`{"question": ...}`) isn't a LangChain integration, so this subclasses
`BaseChatModel` and forwards each call to `FlowiseClient.predict()`. Only
`_generate` (sync) is implemented: `BaseChatModel`'s default `_agenerate`
runs `_generate` in a thread-pool executor, which is enough to satisfy
uqlm's `await llm.ainvoke(...)` calls since `FlowiseClient` is a plain
`requests`-based (blocking) client.
"""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from flowise_client import FlowiseClient


class FlowiseChatModel(BaseChatModel):
    """Adapts a `FlowiseClient` to LangChain's `BaseChatModel` interface.

    Flowise's prediction endpoint takes a single `question` string, not a
    message history, so `_generate` uses the last `HumanMessage`'s content
    as the question and ignores any earlier turns/system prompt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: FlowiseClient
    session_id: Optional[str] = None
    # uqlm samples multiple candidate responses per question to measure
    # consistency, and its ResponseGenerator asserts `count == 1` whenever
    # `llm.temperature == 0`. Flowise's REST API has no temperature knob to
    # forward this to, so this is a fixed, non-zero placeholder that exists
    # purely to satisfy that check; any actual response variability comes
    # from the chatflow's own (non-)determinism, not from this value.
    temperature: float = 1.0

    @property
    def _llm_type(self) -> str:
        return "flowise"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            messages[-1].content if messages else "",
        )
        answer = self.client.predict(question, session_id=self.session_id)
        message = AIMessage(content=answer)
        return ChatResult(generations=[ChatGeneration(message=message)])
