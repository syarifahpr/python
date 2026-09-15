import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowise_client import FlowiseClient
from flowise_llm import FlowiseChatModel


def _make_llm(**kwargs) -> FlowiseChatModel:
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc")
    return FlowiseChatModel(client=client, **kwargs)


def test_is_a_base_chat_model():
    # uqlm's ResponseGenerator asserts isinstance(llm, BaseChatModel) before use
    assert isinstance(_make_llm(), BaseChatModel)


def test_generate_uses_last_human_message_as_question():
    llm = _make_llm()
    with patch.object(FlowiseClient, "predict", return_value="the answer") as predict:
        result = llm._generate([SystemMessage("sys"), HumanMessage("what is sorghum?")])
    predict.assert_called_once_with("what is sorghum?", session_id=None)
    assert result.generations[0].message.content == "the answer"
    assert isinstance(result.generations[0].message, AIMessage)


def test_ainvoke_runs_generate_via_executor():
    # BaseChatModel's default _agenerate runs _generate in a thread executor
    # when only _generate is overridden -- this is what uqlm's
    # `await llm.ainvoke(messages)` relies on.
    llm = _make_llm()
    with patch.object(FlowiseClient, "predict", return_value="async answer"):
        message = asyncio.run(llm.ainvoke([HumanMessage("question?")]))
    assert message.content == "async answer"


def test_temperature_defaults_nonzero_and_is_settable():
    # uqlm's ResponseGenerator requires temperature > 0 whenever it samples
    # more than one response per prompt, and mutates llm.temperature
    # internally (saving/restoring it) around candidate generation.
    llm = _make_llm()
    assert llm.temperature > 0
    llm.temperature = 0.7
    assert llm.temperature == 0.7


def test_session_id_forwarded_to_client():
    llm = _make_llm(session_id="abc-123")
    with patch.object(FlowiseClient, "predict", return_value="ok") as predict:
        llm._generate([HumanMessage("hi")])
    predict.assert_called_once_with("hi", session_id="abc-123")
