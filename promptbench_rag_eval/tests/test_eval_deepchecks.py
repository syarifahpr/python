import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_deepchecks import collect_answers, query, summarize_suite_result


def test_query_returns_raw_answer_unmodified():
    # Unlike eval_rag.py/eval_robustness.py, eval_deepchecks.py must not run
    # promptbench's OutputProcess.general (lowercase + strip punctuation) on
    # the answer, since deepchecks needs the chatbot's real output.
    answer, error = query(lambda text: "Sorgum ADALAH Tanaman!!", "question")
    assert answer == "Sorgum ADALAH Tanaman!!"
    assert error is None


def test_query_returns_error_on_exception():
    def failing_model(text):
        raise RuntimeError("boom")

    answer, error = query(failing_model, "question")
    assert answer == ""
    assert error == "boom"


def test_collect_answers_scores_each_row():
    model = {"Q1": "Paris", "Q2": "Lyon"}.get
    dataset = [{"question": "Q1", "answer": "Paris"}, {"question": "Q2", "answer": "Marseille"}]
    results = collect_answers(model, dataset)
    assert results[0]["answer"] == "Paris"
    assert results[0]["em"] == 1.0
    assert results[1]["answer"] == "Lyon"
    assert results[1]["em"] == 0.0
    assert all(r["error"] is None for r in results)


def test_collect_answers_stops_on_quota_error():
    def quota_model(text):
        raise RuntimeError("Predictions limit exceeded")

    dataset = [{"question": f"Q{i}", "answer": "x"} for i in range(5)]
    results = collect_answers(quota_model, dataset)
    assert len(results) == 1
    assert results[0]["error"] == "Predictions limit exceeded"


def test_collect_answers_keeps_going_on_non_quota_error():
    def flaky_model(text):
        if text == "Q1":
            raise RuntimeError("temporary glitch")
        return "ok"

    dataset = [{"question": "Q1", "answer": "x"}, {"question": "Q2", "answer": "ok"}]
    results = collect_answers(flaky_model, dataset)
    assert len(results) == 2
    assert results[0]["error"] == "temporary glitch"
    assert results[1]["answer"] == "ok"


class _FakeCategory:
    def __init__(self, value):
        self.value = value


class _FakeCondition:
    def __init__(self, name, category, details):
        self.name = name
        self.category = _FakeCategory(category)
        self.details = details


class _FakeCheck:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class CheckResult:  # name matters: summarize_suite_result() dispatches on type(...).__name__
    def __init__(self, check_name, conditions):
        self.check = _FakeCheck(check_name)
        self.conditions_results = conditions


class CheckFailure:  # name matters, see above
    pass


class _FakeSuiteResult:
    def __init__(self, results):
        self.results = results


def test_summarize_suite_result_flattens_conditions_and_skips_failures():
    good = CheckResult("Text Duplicates", [_FakeCondition("ratio <= 5%", "WARN", "Found 25%")])
    failure = CheckFailure()
    suite_result = _FakeSuiteResult([good, failure])

    summary = summarize_suite_result(suite_result)

    assert summary == [
        {"check": "Text Duplicates", "condition": "ratio <= 5%", "category": "WARN", "details": "Found 25%"}
    ]
