"""Measure robustness of a RAG chatbot (Flowise or Dify): for each question,
perturb it with small, meaning-preserving noise (typos, casing/punctuation
changes) and see how much the chatbot's answer degrades compared to the
clean question — the same idea behind promptbench's adversarial prompt
attacks (TextFooler/TextBugger/DeepWordBug/CheckList/...), reimplemented
without promptbench's TextAttack-based `prompt_attack` module since that
targets classification tasks with discrete labels, not free-text RAG
answers. See perturbations.py for details on each perturbation type.

Example (Flowise):
    python eval_robustness.py --provider flowise \\
        --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \\
        --dataset data/qa_dataset.example.csv \\
        --output robustness.csv

Example (Dify — needs an API key from the Dify console, not the
udify.app/chat/<token> share link; see providers.py/dify_client.py):
    python eval_robustness.py --provider dify --api-key app-xxxxxxxx \\
        --dataset data/qa_dataset.example.csv \\
        --output robustness.csv

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
from pathlib import Path

from dotenv import load_dotenv

try:
    from promptbench import OutputProcess
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
from metrics import f1_score, score_pair
from perturbations import PERTURBATIONS
from providers import PROVIDERS, build_model


def query(model, text: str) -> tuple[str, str | None]:
    """Call the model and return (cleaned_answer, error)."""
    try:
        raw = model(text)
    except Exception as exc:  # noqa: BLE001 - surfaced in the results row
        return "", str(exc)
    return (OutputProcess.general(raw) if raw else ""), None


def run_robustness_eval(
    model,
    dataset: list[dict],
    attack_names: list[str],
    sleep: float = 0.0,
    seed: int = 0,
) -> list[dict]:
    results: list[dict] = []

    for row_idx, row in enumerate(dataset):
        question = row["question"]
        expected = row["answer"]

        clean_answer, clean_error = query(model, question)
        clean_scores = score_pair(clean_answer, expected) if not clean_error else {"em": 0.0, "f1": 0.0}
        results.append({
            "question": question,
            "attack": "clean",
            "perturbed_question": question,
            "answer": clean_answer,
            "f1_vs_ground_truth": clean_scores["f1"],
            "answer_stability_vs_clean": 1.0,
            "error": clean_error,
        })
        if sleep:
            time.sleep(sleep)

        if clean_error and is_quota_error(clean_error):
            print(
                f"\nProvider quota/rate limit exceeded after {len(results)} request(s) — "
                "stopping early. Wait for the quota to reset and re-run, or increase your "
                "plan's limit.",
                file=sys.stderr,
            )
            return results

        for attack_name in attack_names:
            perturb_fn = PERTURBATIONS[attack_name]
            perturbed_question = perturb_fn(question, seed=seed + row_idx)
            answer, error = query(model, perturbed_question)
            scores = score_pair(answer, expected) if not error else {"em": 0.0, "f1": 0.0}
            stability = f1_score(answer, clean_answer) if (answer and clean_answer) else 0.0
            results.append({
                "question": question,
                "attack": attack_name,
                "perturbed_question": perturbed_question,
                "answer": answer,
                "f1_vs_ground_truth": scores["f1"],
                "answer_stability_vs_clean": stability,
                "error": error,
            })
            if sleep:
                time.sleep(sleep)

            if error and is_quota_error(error):
                print(
                    f"\nProvider quota/rate limit exceeded after {len(results)} request(s) — "
                    "stopping early. Wait for the quota to reset and re-run, or increase your "
                    "plan's limit.",
                    file=sys.stderr,
                )
                return results

    return results


def summarize(results: list[dict]) -> dict:
    by_attack: dict[str, list[dict]] = {}
    for r in results:
        by_attack.setdefault(r["attack"], []).append(r)

    clean_f1 = sum(r["f1_vs_ground_truth"] for r in by_attack.get("clean", [])) / max(len(by_attack.get("clean", [])), 1)

    summary = {}
    for attack, rows in by_attack.items():
        n = len(rows)
        avg_f1 = sum(r["f1_vs_ground_truth"] for r in rows) / n
        avg_stability = sum(r["answer_stability_vs_clean"] for r in rows) / n
        summary[attack] = {
            "n": n,
            "avg_f1_vs_ground_truth": round(avg_f1, 4),
            "f1_drop_vs_clean": round(clean_f1 - avg_f1, 4) if attack != "clean" else 0.0,
            "avg_answer_stability_vs_clean": round(avg_stability, 4),
            "errors": sum(1 for r in rows if r["error"]),
        }
    return summary


def print_summary(summary: dict) -> None:
    print("\n=== Robustness results ===")
    print(f"{'attack':<14} {'n':>4} {'avg_f1':>8} {'f1_drop':>9} {'stability':>10} {'errors':>7}")
    for attack, stats in summary.items():
        print(
            f"{attack:<14} {stats['n']:>4} {stats['avg_f1_vs_ground_truth']:>8.3f} "
            f"{stats['f1_drop_vs_clean']:>9.3f} {stats['avg_answer_stability_vs_clean']:>10.3f} "
            f"{stats['errors']:>7}"
        )
    print(
        "\n(f1_drop_vs_clean: how much answer quality falls when the question is perturbed — "
        "higher means less robust. stability: word-overlap between the perturbed answer and "
        "the clean answer — lower means the chatbot's answer changed more.)"
    )


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
        default=os.getenv("FLOWISE_API_URL") or os.getenv("DIFY_BASE_URL"),
        help="Flowise prediction API URL (--provider flowise), or Dify API base URL "
        "(--provider dify; default https://api.dify.ai/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("FLOWISE_API_KEY") or os.getenv("DIFY_API_KEY"),
        help="API key (optional for Flowise, required for Dify)",
    )
    parser.add_argument("--dataset", default="data/qa_dataset.example.csv", help="CSV with question,answer columns")
    parser.add_argument("--output", default="robustness.csv", help="Where to write per-question results")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows of the dataset")
    parser.add_argument(
        "--attacks", default=",".join(PERTURBATIONS), help=f"Comma-separated perturbations to run: {list(PERTURBATIONS)}"
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between requests (default: 1.0)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible perturbations")
    args = parser.parse_args()

    attack_names = [a.strip() for a in args.attacks.split(",") if a.strip()]
    unknown = [a for a in attack_names if a not in PERTURBATIONS]
    if unknown:
        sys.exit(f"Unknown attack(s) {unknown}. Available: {list(PERTURBATIONS)}")

    dataset = load_qa_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    model = build_model(args.provider, args.url, args.api_key, args.timeout)

    total_requests = len(dataset) * (1 + len(attack_names))
    print(
        f"Evaluating {len(dataset)} question(s) x (1 clean + {len(attack_names)} attack(s)) "
        f"= {total_requests} request(s) against {args.provider} ({args.url or 'default endpoint'})"
    )
    results = run_robustness_eval(model, dataset, attack_names, sleep=args.sleep, seed=args.seed)
    summary = summarize(results)
    print_summary(summary)
    save_results(results, summary, args.output)


if __name__ == "__main__":
    main()
