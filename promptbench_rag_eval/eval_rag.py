"""Evaluate a RAG chatbot's answers (Flowise or Dify) against a labeled QA
set, across several prompt phrasings, using promptbench's Prompt/InputProcess
utilities for templating and this project's own SQuAD-style EM/F1 scoring.

Example (Flowise):
    python eval_rag.py --provider flowise \\
        --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \\
        --dataset data/qa_dataset.example.csv \\
        --output results.csv

Example (Dify — the udify.app/chat/<token> share link is a human chat UI,
not an API endpoint; you need an API key from the Dify console instead,
see providers.py/dify_client.py):
    python eval_rag.py --provider dify --api-key app-xxxxxxxx \\
        --dataset data/qa_dataset.example.csv \\
        --output results.csv

Credentials/URL can also be provided via a .env file (see .env.example):
    FLOWISE_API_URL=...
    FLOWISE_API_KEY=...
    DIFY_API_KEY=...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

try:
    from promptbench import InputProcess, OutputProcess, Prompt
except ImportError as exc:  # pragma: no cover - environment guidance only
    sys.exit(
        "Could not import promptbench. Install it with:\n"
        "  pip install -r requirements.txt\n"
        "If installation of the 'autocorrect' dependency fails on modern pip/setuptools, run:\n"
        "  pip install 'setuptools<60' && pip install autocorrect==2.6.1\n"
        "before installing the rest of requirements.txt.\n"
        f"(original error: {exc})"
    )

from dataset import load_qa_dataset
from errors import is_quota_error
from metrics import aggregate, score_pair
from prompts import load_templates
from providers import PROVIDERS, build_model, env_defaults


def run_evaluation(
    model,
    templates: list[str],
    dataset: list[dict],
    workers: int = 1,
    sleep: float = 0.0,
) -> list[dict]:
    prompt_bank = Prompt(templates)

    def evaluate_one(template_idx: int, row: dict) -> tuple[dict, bool]:
        template = prompt_bank[template_idx]
        input_text = InputProcess.basic_format(template, {"question": row["question"]})
        start = time.time()
        try:
            raw_pred = model(input_text)
            error = None
        except Exception as exc:  # noqa: BLE001 - surfaced in the results row
            raw_pred = ""
            error = str(exc)
        quota_exceeded = bool(error) and is_quota_error(error)
        latency = time.time() - start
        pred = OutputProcess.general(raw_pred) if raw_pred else ""
        scores = score_pair(pred, row["answer"]) if not error else {"em": 0.0, "f1": 0.0}
        result = {
            "template_idx": template_idx,
            "template": template,
            "question": row["question"],
            "expected_answer": row["answer"],
            "raw_answer": raw_pred,
            "cleaned_answer": pred,
            "em": scores["em"],
            "f1": scores["f1"],
            "latency_s": round(latency, 3),
            "error": error,
        }
        return result, quota_exceeded

    jobs = [(t_idx, row) for t_idx in range(len(prompt_bank)) for row in dataset]

    if workers <= 1:
        results: list[dict] = []
        for t_idx, row in jobs:
            result, quota_exceeded = evaluate_one(t_idx, row)
            results.append(result)
            if quota_exceeded:
                print(
                    f"\nProvider quota/rate limit exceeded after {len(results)} request(s) — "
                    "stopping early instead of sending the rest, which would all fail the "
                    "same way. Wait for the quota to reset and re-run, or increase your plan's limit.",
                    file=sys.stderr,
                )
                break
            if sleep:
                time.sleep(sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [r for r, _ in pool.map(lambda job: evaluate_one(*job), jobs)]

    return results


def summarize(results: list[dict]) -> dict:
    by_template: dict[int, list[dict]] = {}
    for r in results:
        by_template.setdefault(r["template_idx"], []).append(r)

    summary = {}
    for t_idx, rows in by_template.items():
        summary[t_idx] = {
            "template": rows[0]["template"],
            **aggregate([{"em": r["em"], "f1": r["f1"]} for r in rows]),
            "errors": sum(1 for r in rows if r["error"]),
            "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 3),
        }
    summary["overall"] = aggregate([{"em": r["em"], "f1": r["f1"]} for r in results])
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== Per-prompt results ===")
    for t_idx, stats in summary.items():
        if t_idx == "overall":
            continue
        print(
            f"[{t_idx}] EM={stats['em']:.3f} F1={stats['f1']:.3f} "
            f"n={stats['n']} errors={stats['errors']} avg_latency={stats['avg_latency_s']}s\n"
            f"    template: {stats['template']!r}"
        )
    overall = summary["overall"]
    print(f"\n=== Overall: EM={overall['em']:.3f} F1={overall['f1']:.3f} (n={overall['n']}) ===")


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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS, default="flowise", help="Which RAG backend to call")
    parser.add_argument(
        "--url",
        default=None,
        help="Flowise prediction API URL (--provider flowise), or Dify API base URL "
        "(--provider dify; default https://api.dify.ai/v1). Falls back to "
        "FLOWISE_API_URL/DIFY_BASE_URL for the selected --provider if omitted.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (optional for Flowise, required for Dify). Falls back to "
        "FLOWISE_API_KEY/DIFY_API_KEY for the selected --provider if omitted.",
    )
    parser.add_argument("--dataset", default="data/qa_dataset.example.csv", help="CSV with question,answer columns")
    parser.add_argument("--templates", default=None, help="Text file with one prompt template per line")
    parser.add_argument("--output", default="results.csv", help="Where to write per-question results")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows of the dataset")
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests to the provider (default: 1)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between sequential requests")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    args = parser.parse_args()

    env_url, env_api_key = env_defaults(args.provider)
    args.url = args.url or env_url
    args.api_key = args.api_key or env_api_key

    dataset = load_qa_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]
    templates = load_templates(args.templates)

    model = build_model(args.provider, args.url, args.api_key, args.timeout)

    print(f"Evaluating {len(dataset)} question(s) x {len(templates)} prompt template(s) "
          f"against {args.provider} ({args.url or 'default endpoint'})")
    results = run_evaluation(model, templates, dataset, workers=args.workers, sleep=args.sleep)
    summary = summarize(results)
    print_summary(summary)
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
