"""SQuAD-style exact-match / F1 scoring for RAG answer evaluation.

promptbench ships an ``Eval.compute_squad_v2_f1`` helper, but it depends on a
``promptbench/metrics/squad_v2/squad_v2.py`` resource file that is not
actually bundled in the published PyPI wheel (0.0.4), so calling it raises
``ModuleNotFoundError``. This module reimplements the same standard
normalize-and-overlap formula used by the original SQuAD evaluation script
so answers can be scored without that missing dependency.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_pair(prediction: str, ground_truth: str) -> dict:
    return {
        "em": exact_match_score(prediction, ground_truth),
        "f1": f1_score(prediction, ground_truth),
    }


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    n = len(scores)
    return {
        "em": sum(s["em"] for s in scores) / n,
        "f1": sum(s["f1"] for s in scores) / n,
        "n": n,
    }
