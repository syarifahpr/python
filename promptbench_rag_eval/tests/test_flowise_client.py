import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowise_client import FlowiseClient, FlowiseError, FlowiseQuotaExceededError


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_predict_extracts_text_field():
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc")
    with patch("flowise_client.requests.post", return_value=_mock_response(json_data={"text": "hello"})) as post:
        answer = client.predict("hi")
    assert answer == "hello"
    _, kwargs = post.call_args
    assert kwargs["json"] == {"question": "hi"}
    assert "Authorization" not in kwargs["headers"]


def test_predict_sends_api_key_header():
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc", api_key="secret")
    with patch("flowise_client.requests.post", return_value=_mock_response(json_data={"text": "ok"})) as post:
        client.predict("hi")
    _, kwargs = post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


def test_predict_falls_back_to_answer_field():
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc")
    with patch("flowise_client.requests.post", return_value=_mock_response(json_data={"answer": "fallback"})):
        assert client.predict("hi") == "fallback"


def test_predict_raises_on_4xx():
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc", max_retries=1)
    with patch("flowise_client.requests.post", return_value=_mock_response(status_code=404, text="not found")):
        try:
            client.predict("hi")
            assert False, "expected FlowiseError"
        except FlowiseError as exc:
            assert "404" in str(exc)


def test_predict_retries_on_5xx_then_succeeds():
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc", max_retries=2, retry_backoff=0)
    responses = [_mock_response(status_code=500), _mock_response(json_data={"text": "recovered"})]
    with patch("flowise_client.requests.post", side_effect=responses):
        assert client.predict("hi") == "recovered"


def test_predict_raises_quota_exceeded_without_retrying():
    client = FlowiseClient(url="https://example.com/api/v1/prediction/abc", max_retries=3, retry_backoff=0)
    quota_body = '{"statusCode":500,"success":false,"message":"Error: predictionsServices.buildChatflow - Predictions limit exceeded"}'
    with patch(
        "flowise_client.requests.post",
        return_value=_mock_response(status_code=500, text=quota_body),
    ) as post:
        try:
            client.predict("hi")
            assert False, "expected FlowiseQuotaExceededError"
        except FlowiseQuotaExceededError as exc:
            assert "limit exceeded" in str(exc).lower()
    # no retries: quota exhaustion won't resolve itself mid-run
    assert post.call_count == 1
