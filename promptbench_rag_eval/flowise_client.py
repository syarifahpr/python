"""Minimal client for the Flowise "prediction" REST API.

Flowise (https://flowiseai.com) exposes each chatflow as:
    POST {base_url}/api/v1/prediction/{chatflow_id}
    body: {"question": "<user question>", ...}
    resp: {"text": "<answer>", ...}  # shape can vary slightly by chatflow config

This wraps that call with retries and defensive response parsing so it can be
used as the "model" under evaluation in eval_rag.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


class FlowiseError(RuntimeError):
    """Raised when the Flowise API returns an error or an unexpected response."""


class FlowiseQuotaExceededError(FlowiseError):
    """Raised when the Flowise account's prediction quota is exhausted.

    Retrying does not help here (the quota won't refill mid-run), so this is
    raised immediately without going through the retry loop, letting callers
    stop the whole evaluation early instead of burning through every
    remaining request only to have each one fail the same way.
    """


@dataclass
class FlowiseClient:
    url: str
    api_key: Optional[str] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    override_config: dict = field(default_factory=dict)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def predict(self, question: str, session_id: Optional[str] = None) -> str:
        """Send a question to the Flowise chatflow and return the answer text."""
        payload: dict[str, Any] = {"question": question}
        if self.override_config:
            payload["overrideConfig"] = self.override_config
        if session_id:
            payload["overrideConfig"] = {
                **payload.get("overrideConfig", {}),
                "sessionId": session_id,
            }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if response.status_code >= 400 and "limit exceeded" in response.text.lower():
                    raise FlowiseQuotaExceededError(
                        f"Flowise prediction quota exceeded (HTTP {response.status_code}): "
                        f"{response.text[:500]}"
                    )
                if response.status_code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
                if response.status_code >= 400:
                    raise FlowiseError(
                        f"Flowise API returned HTTP {response.status_code}: {response.text[:500]}"
                    )
                return self._extract_text(response.json())
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
                    continue
        raise FlowiseError(f"Failed to reach Flowise API after {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def _extract_text(data: Any) -> str:
        """Flowise responses are usually {"text": "..."} but some chatflows
        (e.g. ones returning structured output) use other keys."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("text", "answer", "result", "output"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # fall back: stringify the whole payload so nothing is silently lost
            return str(data)
        return str(data)
