"""Shared helper for detecting quota/rate-limit errors across providers."""
from __future__ import annotations


def is_quota_error(message: str) -> bool:
    m = message.lower()
    return "quota" in m or "limit exceeded" in m or "rate limit" in m
