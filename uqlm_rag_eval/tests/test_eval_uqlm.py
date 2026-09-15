import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_uqlm import run_evaluation, summarize


def _mock_response(text):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"text": text}
    resp.text = ""
    return resp


def test_run_evaluation_end_to_end_with_exact_match_scorer():
    """Full pipeline against a mocked Flowise API: FlowiseChatModel -> uqlm's
    BlackBoxUQ -> per-question results. Uses only the exact_match scorer,
    which needs no model downloads, so this runs fast/offline.
    """
    dataset = [
        {"question": "What is sorghum?", "answer": "A cereal crop."},
        {"question": "What climate does it need?", "answer": "Warm, dry climate."},
    ]
    # Every call returns the exact same text per question's "identity",
    # so exact_match consistency is 1.0 and EM/F1 against a mismatched
    # ground truth is 0.0 -- easy to assert on deterministically.
    with patch(
        "flowise_client.requests.post",
        return_value=_mock_response("A cereal crop."),
    ) as post:
        results, metadata = run_evaluation(
            url="https://example.com/api/v1/prediction/abc",
            api_key=None,
            dataset=dataset,
            scorers=["exact_match"],
            num_responses=2,
            timeout=30.0,
        )

    # 1 original + 2 sampled responses per question, for 2 questions
    assert post.call_count == 2 * (1 + 2)
    assert len(results) == 2
    for row in results:
        assert row["response"] == "A cereal crop."
        assert row["exact_match"] == 1.0
    assert results[0]["em"] == 1.0
    assert results[1]["em"] == 0.0

    summary = summarize(results, ["exact_match"], metadata)
    assert summary["n"] == 2
    assert summary["avg_exact_match"] == 1.0
