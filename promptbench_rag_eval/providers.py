"""Builds the RAG model callable for whichever provider (Flowise or Dify)
the CLI is pointed at, so eval_rag.py / eval_robustness.py don't need to
know about each client's construction details.
"""
from __future__ import annotations

import sys

from dify_client import DifyClient
from flowise_client import FlowiseClient
from rag_model import DifyRAGModel, FlowiseRAGModel

PROVIDERS = ("flowise", "dify")


def build_model(provider: str, url: str | None, api_key: str | None, timeout: float):
    if provider == "flowise":
        if not url:
            sys.exit(
                "Missing Flowise prediction URL. Pass --url or set FLOWISE_API_URL "
                "(e.g. https://cloud.flowiseai.com/api/v1/prediction/<chatflow-id>)."
            )
        client = FlowiseClient(url=url, api_key=api_key, timeout=timeout)
        return FlowiseRAGModel(client)

    if provider == "dify":
        if not api_key:
            sys.exit(
                "Missing Dify API key. A Dify chat-app share link (udify.app/chat/...) is "
                "the human chat UI, not a callable API endpoint. Get an API key from the "
                "Dify console (open your app -> API Access -> API Key) and pass --api-key "
                "or set DIFY_API_KEY."
            )
        client = DifyClient(api_key=api_key, base_url=url or "https://api.dify.ai/v1", timeout=timeout)
        return DifyRAGModel(client)

    sys.exit(f"Unknown --provider {provider!r}. Choose from {PROVIDERS}.")
