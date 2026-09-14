"""A promptbench-compatible callable wrapping a Flowise RAG chatflow.

promptbench's evaluation pattern only requires a callable of the form
``model(input_text) -> str`` (see promptbench.models.LLMModel.__call__).
Rather than teaching promptbench's fixed MODEL_LIST about a custom REST
endpoint, we satisfy that same calling convention directly so the rest of
promptbench (Prompt, InputProcess, OutputProcess) can be reused as-is.
"""
from __future__ import annotations

from flowise_client import FlowiseClient


class FlowiseRAGModel:
    def __init__(self, client: FlowiseClient):
        self.client = client

    def __call__(self, input_text: str, **kwargs) -> str:
        return self.client.predict(input_text, session_id=kwargs.get("session_id"))
