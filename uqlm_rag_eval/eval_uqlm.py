"""Evaluate a Flowise RAG chatbot's hallucination risk with uqlm
(https://github.com/cvs-health/uqlm).

uqlm's `BlackBoxUQ` estimates hallucination likelihood by sampling several
responses to the same question and scoring how consistent they are with
each other (semantic entropy, NLI-based non-contradiction, exact match,
embedding cosine similarity, ...) -- no token log-probabilities needed,
which suits a REST endpoint like Flowise's `/api/v1/prediction/<id>` that
only ever returns response text. `flowise_llm.FlowiseChatModel` adapts
`FlowiseClient` to the `langchain_core.BaseChatModel` interface uqlm
requires; see that module for why.

Each question is also scored with SQuAD-style EM/F1 against the dataset's
`answer` column (metrics.py), so uqlm's confidence scores can be checked
against whether the chatbot was actually right, not just self-consistent.

Example:
    python eval_uqlm.py \\
        --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \\
        --dataset data/qa_dataset.example.csv \\
        --output results.csv

Credentials/URL can also be provided via a .env file (see .env.example):
    FLOWISE_API_URL=...
    FLOWISE_API_KEY=...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from uqlm import BlackBoxUQ
except ImportError as exc:  # pragma: no cover - environment guidance only
    sys.exit(
        "Could not import uqlm. Install it with:\n"
        "  pip install -r requirements.txt\n"
        f"(original error: {exc})"
    )

from dataset import load_qa_dataset
from flowise_client import FlowiseClient
from flowise_llm import FlowiseChatModel
from metrics import score_pair

DEFAULT_SCORERS = ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]
ALL_SCORERS = DEFAULT_SCORERS + ["bert_score", "entailment", "semantic_sets_confidence"]


def run_evaluation(
    url: str,
    api_key: str | None,
    dataset: list[dict],
    scorers: list[str],
    num_responses: int,
    timeout: float,
) -> tuple[list[dict], dict]:
    client = FlowiseClient(url=url, api_key=api_key, timeout=timeout)
    llm = FlowiseChatModel(client=client)
    bbuq = BlackBoxUQ(llm=llm, scorers=scorers)

    questions = [row["question"] for row in dataset]
    uq_result = bbuq.generate_and_score_sync(prompts=questions, num_responses=num_responses)

    data = uq_result.data
    results = []
    for i, row in enumerate(dataset):
        answer = data["responses"][i]
        em_f1 = score_pair(answer, row["answer"])
        result = {
            "question": row["question"],
            "expected_answer": row["answer"],
            "response": answer,
            "em": em_f1["em"],
            "f1": em_f1["f1"],
        }
        for scorer in scorers:
            result[scorer] = data[scorer][i]
        result["sampled_responses"] = " ||| ".join(data["sampled_responses"][i])
        results.append(result)

    return results, uq_result.metadata


def summarize(results: list[dict], scorers: list[str], metadata: dict) -> dict:
    n = len(results)
    summary = {
        "n": n,
        "avg_em": sum(r["em"] for r in results) / n,
        "avg_f1": sum(r["f1"] for r in results) / n,
        "metadata": metadata,
    }
    for scorer in scorers:
        summary[f"avg_{scorer}"] = sum(r[scorer] for r in results) / n
    return summary


def print_summary(summary: dict, scorers: list[str]) -> None:
    print(f"\n=== Summary (n={summary['n']}) ===")
    print(f"avg_em={summary['avg_em']:.3f}  avg_f1={summary['avg_f1']:.3f}  (vs. dataset's expected answers)")
    print("uqlm confidence scores (0=hallucination-prone, 1=consistent across samples):")
    for scorer in scorers:
        print(f"  avg_{scorer}={summary[f'avg_{scorer}']:.3f}")


def save_results(results: list[dict], summary: dict, output_path: str) -> None:
    out = Path(output_path)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_path = out.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved per-question results to {out}")
    print(f"Saved summary to {summary_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=None, help="Flowise prediction API URL. Falls back to FLOWISE_API_URL.")
    parser.add_argument("--api-key", default=None, help="Flowise API key, if the chatflow requires one. Falls back to FLOWISE_API_KEY.")
    parser.add_argument("--dataset", default="data/qa_dataset.example.csv", help="CSV with question,answer columns")
    parser.add_argument("--output", default="results.csv", help="Where to write per-question results")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows of the dataset")
    parser.add_argument("--num-responses", type=int, default=5, help="Sampled responses per question used to measure consistency (uqlm's num_responses)")
    parser.add_argument(
        "--scorers",
        default=",".join(DEFAULT_SCORERS),
        help=(
            f"Comma-separated uqlm BlackBoxUQ scorers, from {ALL_SCORERS}. "
            "semantic_negentropy/noncontradiction/entailment/semantic_sets_confidence "
            "download an NLI model (~1.5GB) on first use; cosine_sim downloads a small "
            "sentence-transformer (~90MB); bert_score downloads a BERT model. Use e.g. "
            "--scorers exact_match,cosine_sim for a lighter run."
        ),
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    args.url = args.url or os.getenv("FLOWISE_API_URL")
    args.api_key = args.api_key or os.getenv("FLOWISE_API_KEY")
    if not args.url:
        sys.exit(
            "Missing Flowise prediction URL. Pass --url or set FLOWISE_API_URL "
            "(e.g. https://cloud.flowiseai.com/api/v1/prediction/<chatflow-id>)."
        )

    dataset = load_qa_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]
    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    for scorer in scorers:
        if scorer not in ALL_SCORERS:
            sys.exit(f"Unknown scorer {scorer!r}. Choose from {ALL_SCORERS}.")

    print(
        f"Evaluating {len(dataset)} question(s) against Flowise ({args.url}) with uqlm "
        f"BlackBoxUQ (num_responses={args.num_responses}, scorers={scorers}) "
        f"-- {1 + args.num_responses} request(s) per question"
    )
    results, metadata = run_evaluation(
        url=args.url,
        api_key=args.api_key,
        dataset=dataset,
        scorers=scorers,
        num_responses=args.num_responses,
        timeout=args.timeout,
    )
    summary = summarize(results, scorers, metadata)
    print_summary(summary, scorers)
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
