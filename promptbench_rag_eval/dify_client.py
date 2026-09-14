"""Minimal client for the Dify "chat-messages" REST API.

Dify (https://dify.ai) exposes each published chat app as:
    POST {base_url}/chat-messages
    headers: Authorization: Bearer {api_key}
    body: {"inputs": {}, "query": "<question>", "response_mode": "blocking", "user": "<id>"}
    resp (blocking mode): {"answer": "<answer>", "conversation_id": "...", ...}

Important: a share link like https://udify.app/chat/<app-token> is Dify's
*human* chat UI (opened in a browser) — it is not itself a callable API
endpoint and does not accept POST requests. To call the app
programmatically you need an API key from the Dify console (open your app
-> "API Access" / "Akses API" -> API Key). The app is identified by that
key, not by the udify.app URL; for Dify Cloud the API base URL is always
https://api.dify.ai/v1 regardless of which app you're calling. For a
self-hosted Dify instance, pass its own base URL instead.

Note: Dify "Agent" type apps may require ``response_mode: "streaming"``
(Server-Sent Events) rather than "blocking" — if you get an error saying
blocking mode isn't supported, that's why; this client only implements
blocking mode since it's sufficient for regular chat/RAG apps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from errors import is_quota_error


class DifyError(RuntimeError):
    """Raised when the Dify API returns an error or an unexpected response."""


class DifyQuotaExceededError(DifyError):
    """Raised when the Dify app/workspace's usage quota or rate limit is hit.

    Retrying does not help here, so this is raised immediately without going
    through the retry loop.
    """


@dataclass
class DifyClient:
    api_key: str
    base_url: str = "https://api.dify.ai/v1"
    user: str = "promptbench-rag-eval"
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 2.0

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def predict(self, question: str, conversation_id: Optional[str] = None) -> str:
        payload: dict[str, Any] = {
            "inputs": {},
            "query": question,
            "response_mode": "blocking",
            "user": self.user,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        url = f"{self.base_url.rstrip('/')}/chat-messages"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
                if response.status_code == 429 or (
                    response.status_code >= 400 and is_quota_error(response.text)
                ):
                    raise DifyQuotaExceededError(
                        f"Dify quota/rate limit exceeded (HTTP {response.status_code}): {response.text[:500]}"
                    )
                if response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
                if response.status_code >= 400:
                    raise DifyError(f"Dify API returned HTTP {response.status_code}: {response.text[:500]}")
                data = response.json()
                answer = data.get("answer")
                return answer if answer is not None else str(data)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
        raise DifyError(f"Failed to reach Dify API after {self.max_retries} attempts: {last_exc}")
