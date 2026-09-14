import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dify_client import DifyClient, DifyError, DifyQuotaExceededError


def _mock_response(status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


def test_predict_extracts_answer_field():
    client = DifyClient(api_key="app-secret")
    with patch("dify_client.requests.post", return_value=_mock_response(json_data={"answer": "hello"})) as post:
        answer = client.predict("hi")
    assert answer == "hello"
    args, kwargs = post.call_args
    assert args[0] == "https://api.dify.ai/v1/chat-messages"
    assert kwargs["json"]["query"] == "hi"
    assert kwargs["json"]["response_mode"] == "blocking"
    assert kwargs["headers"]["Authorization"] == "Bearer app-secret"


def test_predict_uses_custom_base_url_for_self_hosted():
    client = DifyClient(api_key="app-secret", base_url="https://my-dify.example.com/v1")
    with patch("dify_client.requests.post", return_value=_mock_response(json_data={"answer": "ok"})) as post:
        client.predict("hi")
    args, _ = post.call_args
    assert args[0] == "https://my-dify.example.com/v1/chat-messages"


def test_predict_raises_on_4xx():
    client = DifyClient(api_key="app-secret", max_retries=1)
    with patch("dify_client.requests.post", return_value=_mock_response(status_code=404, text="not found")):
        try:
            client.predict("hi")
            assert False, "expected DifyError"
        except DifyError as exc:
            assert "404" in str(exc)


def test_predict_raises_quota_exceeded_on_429_without_retry():
    client = DifyClient(api_key="app-secret", max_retries=3, retry_backoff=0)
    with patch(
        "dify_client.requests.post",
        return_value=_mock_response(status_code=429, text='{"message":"rate limit exceeded"}'),
    ) as post:
        try:
            client.predict("hi")
            assert False, "expected DifyQuotaExceededError"
        except DifyQuotaExceededError:
            pass
    assert post.call_count == 1


def test_predict_retries_on_5xx_then_succeeds():
    client = DifyClient(api_key="app-secret", max_retries=2, retry_backoff=0)
    responses = [_mock_response(status_code=500), _mock_response(json_data={"answer": "recovered"})]
    with patch("dify_client.requests.post", side_effect=responses):
        assert client.predict("hi") == "recovered"
