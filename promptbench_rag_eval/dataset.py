"""Shared QA dataset loading for eval_rag.py and eval_robustness.py."""
from __future__ import annotations

import csv


def load_qa_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in dataset: {path}")
    for row in rows:
        if "question" not in row or "answer" not in row:
            raise ValueError("Dataset CSV must have 'question' and 'answer' columns")
    return rows
