"""Evaluate a RAG chatbot's answers (Flowise or Dify) for text-quality and
integrity issues using deepchecks' NLP module
(https://github.com/deepchecks/deepchecks), complementing eval_rag.py's
correctness (EM/F1) scoring and eval_robustness.py's perturbation testing
with checks deepchecks is actually built for: duplicate/canned answers,
garbled special-character output, repeated boilerplate substrings,
statistical outliers in answer properties, and (with --compare-attack)
property drift between clean-question and perturbed-question answers.

Example (Flowise):
    python eval_deepchecks.py --provider flowise \\
        --url https://cloud.flowiseai.com/api/v1/prediction/d8c9773e-d2f0-4045-89a3-667f2ec75559 \\
        --dataset data/qa_dataset.example.csv \\
        --output deepchecks_report.html

Example (Dify — needs an API key from the Dify console, not the
udify.app/chat/<token> share link; see providers.py/dify_client.py):
    python eval_deepchecks.py --provider dify --api-key app-xxxxxxxx \\
        --dataset data/qa_dataset.example.csv \\
        --output deepchecks_report.html

Add a robustness comparison (clean vs. one perturbation type from
perturbations.py) as a train/test drift suite:
    python eval_deepchecks.py --provider flowise --compare-attack deepwordbug ...

Credentials/URL can also come from a .env file (see .env.example):
    FLOWISE_API_URL=...
    FLOWISE_API_KEY=...
    DIFY_API_KEY=...

Notes:
- Unlike eval_rag.py/eval_robustness.py, this script does not depend on
  promptbench — it feeds deepchecks the chatbot's raw, unaltered answers
  (promptbench's OutputProcess.general lowercases and strips punctuation,
  which is fine for EM/F1 matching but would corrupt the text deepchecks is
  meant to inspect). Install deepchecks' NLP extra separately from
  requirements.txt — see requirements-deepchecks.txt (it pins
  scikit-learn/category-encoders versions deepchecks 0.19.x actually works
  with, since it doesn't cap either itself, and keeps this fully decoupled
  from promptbench's own dependency pins):
      pip install -r requirements-deepchecks.txt
- Property-based checks (TextPropertyOutliers, PropertyDrift) call
  deepchecks' TextData.calculate_builtin_properties(), which unconditionally
  downloads a ~130MB fastText language-ID model from
  dl.fbaipublicfiles.com the first time it runs (cached under
  ~/.deepchecks afterwards) and requires the `fasttext` package. If you're
  offline, don't have that package, or don't want the download, pass
  --skip-properties: you still get the pure-text integrity checks
  (duplicates, special characters, frequent substrings[, train/test sample
  mix]).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from dataset import load_qa_dataset
from errors import is_quota_error
from metrics import score_pair
from perturbations import PERTURBATIONS
from providers import PROVIDERS, build_model, env_defaults


def query(model, text: str) -> tuple[str, str | None]:
    """Call the model and return (raw_answer, error). Unlike
    eval_rag.py/eval_robustness.py, the answer is left unmodified — deepchecks
    should see the chatbot's actual output, not a lowercased/punctuation-stripped
    version of it (see module docstring)."""
    try:
        raw = model(text)
    except Exception as exc:  # noqa: BLE001 - surfaced in the results row
        return "", str(exc)
    return raw or "", None


def collect_answers(model, dataset: list[dict], sleep: float = 0.0) -> list[dict]:
    """Query the model for every question and score it against the expected
    answer. Stops early (returning what was collected so far) on a
    quota/rate-limit error, same behavior as eval_rag.py/eval_robustness.py.
    """
    results: list[dict] = []
    for row in dataset:
        answer, error = query(model, row["question"])
        scores = score_pair(answer, row["answer"]) if not error else {"em": 0.0, "f1": 0.0}
        results.append({
            "question": row["question"],
            "expected_answer": row["answer"],
            "answer": answer,
            "em": scores["em"],
            "f1": scores["f1"],
            "error": error,
        })
        if sleep:
            time.sleep(sleep)
        if error and is_quota_error(error):
            print(
                f"\nProvider quota/rate limit exceeded after {len(results)} request(s) — "
                "stopping early instead of sending the rest, which would all fail the same way.",
                file=sys.stderr,
            )
            break
    return results


def build_text_data(results: list[dict], name: str, compute_properties: bool):
    """Wrap a list of collect_answers() rows in a deepchecks TextData,
    excluding rows where the provider errored (there's no chatbot answer to
    evaluate) and attaching numeric metadata (em/f1) so it's visible
    alongside each answer in the deepchecks report.

    Returns (text_data, properties_ok, n_excluded).
    """
    import pandas as pd
    from deepchecks.nlp import TextData

    usable = [r for r in results if not r["error"]]
    n_excluded = len(results) - len(usable)

    metadata = pd.DataFrame({
        "em": [r["em"] for r in usable],
        "f1": [r["f1"] for r in usable],
    })
    text_data = TextData(
        raw_text=[r["answer"] for r in usable],
        metadata=metadata,
        categorical_metadata=[],
        name=name,
    )

    properties_ok = False
    if compute_properties:
        try:
            text_data.calculate_builtin_properties(include_long_calculation_properties=False)
            properties_ok = True
        except Exception as exc:  # noqa: BLE001 - best-effort, see module docstring
            print(
                f"\nCould not compute deepchecks text properties for {name!r} ({exc}). "
                "Continuing with text-integrity checks only (duplicates/special-characters/"
                "frequent-substrings). Install `fasttext` and ensure network access to "
                "dl.fbaipublicfiles.com to enable property-based checks, or pass "
                "--skip-properties to silence this.",
                file=sys.stderr,
            )

    return text_data, properties_ok, n_excluded


def build_single_dataset_suite(properties_ok: bool):
    from deepchecks.nlp import Suite
    from deepchecks.nlp.checks import FrequentSubstrings, SpecialCharacters, TextDuplicates, TextPropertyOutliers

    checks = [
        TextDuplicates().add_condition_ratio_less_or_equal(),
        SpecialCharacters().add_condition_samples_ratio_w_special_characters_less_or_equal(),
        FrequentSubstrings().add_condition_zero_result(),
    ]
    if properties_ok:
        checks.append(TextPropertyOutliers().add_condition_outlier_ratio_less_or_equal())
    return Suite("Chatbot answer quality", *checks)


def build_train_test_suite(properties_ok: bool):
    from deepchecks.nlp import Suite
    from deepchecks.nlp.checks import PropertyDrift, TrainTestSamplesMix

    checks = [TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal()]
    if properties_ok:
        checks.append(PropertyDrift().add_condition_drift_score_less_than())
    return Suite("Chatbot answer drift: clean vs. perturbed question", *checks)


def summarize_suite_result(suite_result) -> list[dict]:
    """Flatten a deepchecks SuiteResult's conditions into plain dicts (one
    per condition) for a JSON summary and console printout."""
    summary = []
    for result in suite_result.results:
        if type(result).__name__ != "CheckResult":
            continue  # skip CheckFailure entries (e.g. checks irrelevant without a test dataset)
        for condition in result.conditions_results:
            summary.append({
                "check": result.check.name(),
                "condition": condition.name,
                "category": condition.category.value,
                "details": condition.details,
            })
    return summary


def print_condition_summary(title: str, conditions: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not conditions:
        print("(no conditions evaluated)")
        return
    for c in conditions:
        print(f"[{c['category'].upper():4}] {c['check']} — {c['condition']}\n       {c['details']}")


def save_answers_csv(results: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument("--output", default="deepchecks_report.html", help="Where to write the deepchecks HTML report")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows of the dataset")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between sequential requests")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--compare-attack",
        choices=list(PERTURBATIONS),
        default=None,
        help="Also collect answers to a perturbed version of each question (see perturbations.py) "
        "and run a train/test drift suite comparing clean vs. perturbed answers.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible perturbations")
    parser.add_argument(
        "--skip-properties",
        action="store_true",
        help="Skip deepchecks' text-property calculation (TextPropertyOutliers/PropertyDrift). Use this if "
        "you're offline or don't have the `fasttext` package installed — see module docstring.",
    )
    args = parser.parse_args()

    env_url, env_api_key = env_defaults(args.provider)
    args.url = args.url or env_url
    args.api_key = args.api_key or env_api_key

    dataset = load_qa_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    model = build_model(args.provider, args.url, args.api_key, args.timeout)

    print(f"Collecting {len(dataset)} answer(s) from {args.provider} ({args.url or 'default endpoint'})")
    clean_results = collect_answers(model, dataset, sleep=args.sleep)

    perturbed_results = None
    if args.compare_attack:
        perturb_fn = PERTURBATIONS[args.compare_attack]
        perturbed_dataset = [
            {"question": perturb_fn(row["question"], seed=args.seed + idx), "answer": row["answer"]}
            for idx, row in enumerate(dataset[: len(clean_results)])
        ]
        print(f"Collecting {len(perturbed_dataset)} answer(s) for the '{args.compare_attack}' perturbation")
        perturbed_results = collect_answers(model, perturbed_dataset, sleep=args.sleep)

    out = Path(args.output)
    all_conditions: list[dict] = []

    clean_td, properties_ok, n_excluded = build_text_data(
        clean_results, name="clean", compute_properties=not args.skip_properties
    )
    if n_excluded:
        print(f"Excluded {n_excluded} row(s) with provider errors from the deepchecks report.")

    suite = build_single_dataset_suite(properties_ok)
    suite_result = suite.run(clean_td)
    suite_result.save_as_html(str(out))
    print(f"\nSaved deepchecks report to {out}")
    conditions = summarize_suite_result(suite_result)
    print_condition_summary("Chatbot answer quality", conditions)
    all_conditions.extend({"suite": "answer_quality", **c} for c in conditions)

    save_answers_csv(clean_results, out.with_suffix(".csv"))

    if perturbed_results is not None:
        perturbed_td, perturbed_properties_ok, perturbed_excluded = build_text_data(
            perturbed_results, name=args.compare_attack, compute_properties=not args.skip_properties
        )
        if perturbed_excluded:
            print(f"Excluded {perturbed_excluded} perturbed row(s) with provider errors.")

        compare_suite = build_train_test_suite(properties_ok and perturbed_properties_ok)
        compare_result = compare_suite.run(train_dataset=clean_td, test_dataset=perturbed_td)
        compare_out = out.with_name(f"{out.stem}.compare-{args.compare_attack}{out.suffix}")
        compare_result.save_as_html(str(compare_out))
        print(f"Saved drift comparison report to {compare_out}")
        compare_conditions = summarize_suite_result(compare_result)
        print_condition_summary(f"Clean vs. '{args.compare_attack}' drift", compare_conditions)
        all_conditions.extend({"suite": "clean_vs_perturbed_drift", **c} for c in compare_conditions)

        save_answers_csv(perturbed_results, out.with_name(f"{out.stem}.compare-{args.compare_attack}.csv"))

    summary_path = out.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_conditions, f, indent=2, ensure_ascii=False)
    print(f"Saved condition summary to {summary_path}")


if __name__ == "__main__":
    main()
