"""Strict Groq HTTP boundary regression tests. No sockets or real credentials."""

import asyncio
import json
from io import BytesIO
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from llm_routing_eval.adapters.providers import (
    HttpResult,
    UrllibGroqTransport,
)
from llm_routing_eval.adapters.providers import transport as transport_module
from llm_routing_eval.adapters.providers.response_mapping import GroqResponseMapper
from llm_routing_eval.core import CallStatus, ErrorCategory, ModelRequest


class FakeResponse:
    def __init__(self, raw: bytes, *, status: int = 200) -> None:
        self.status = status
        self.raw = raw
        self.read_limit: int | None = None
        self.closed = False

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def read(self, limit: int) -> bytes:
        self.read_limit = limit
        return self.raw[:limit]


class FakeOpener:
    def __init__(self, response: FakeResponse | HTTPError) -> None:
        self.response = response
        self.request: Request | None = None
        self.timeout: float | None = None

    def open(self, request: Request, *, timeout: float) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        if isinstance(self.response, HTTPError):
            raise self.response
        return self.response


def stub_http(monkeypatch: pytest.MonkeyPatch, reply: FakeResponse | HTTPError) -> FakeOpener:
    opener = FakeOpener(reply)

    def replace_opener(*handlers: object) -> FakeOpener:
        assert any(isinstance(h, transport_module._RejectRedirects) for h in handlers)
        return opener

    monkeypatch.setattr(transport_module, "build_opener", replace_opener)
    return opener


def send() -> HttpResult:
    return asyncio.run(
        UrllibGroqTransport().post_chat(
            {"model": "synthetic-small", "messages": [{"role": "user", "content": "fixture"}]},
            api_key="NOT_A_REAL_KEY",
            timeout_seconds=3.5,
        )
    )


def test_successfully_parses_one_bounded_200_response(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"choices": [{"message": {"content": "synthetic"}}]}
    response = FakeResponse(json.dumps(payload).encode("utf-8"))
    opener = stub_http(monkeypatch, response)
    result = send()
    assert result == HttpResult(status_code=200, body=payload)
    assert response.read_limit == transport_module._MAX_RESPONSE_BYTES + 1
    assert response.closed
    assert opener.request is not None
    assert opener.request.full_url == "https://api.groq.com/openai/v1/chat/completions"
    assert opener.request.get_method() == "POST"
    assert opener.request.get_header("Authorization") == "Bearer NOT_A_REAL_KEY"
    assert opener.timeout == 3.5


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{ invalid provider json",
        b"\xff\xfe\xff",
        b"[]",
        b"1",
        b"null",
        b'"plain"',
        b"{" + b"a" * (transport_module._MAX_RESPONSE_BYTES + 1),
    ],
)
def test_non_object_or_invalid_or_oversized_200_is_nonretryable(
    monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    reply = FakeResponse(raw)
    stub_http(monkeypatch, reply)
    result = send()
    assert result.status_code == 200
    assert result.body is None
    assert reply.closed
    request = ModelRequest(
        run_id="run",
        case_id="case",
        trace_id="trace",
        attempt_id="attempt",
        provider="groq",
        model="synthetic-small",
        prompt="fixture",
        timeout_seconds=3.5,
    )
    mapped = GroqResponseMapper().map(request, result, elapsed_ms=1.0)
    assert mapped.status is CallStatus.FAILED
    assert mapped.error is not None
    assert mapped.error.code == "GROQ_INVALID_RESPONSE"
    assert not mapped.error.retryable
    assert "invalid provider json" not in mapped.model_dump_json()


def test_non_200_success_code_does_not_read_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnreadableResponse(FakeResponse):
        def read(self, limit: int) -> bytes:
            raise AssertionError(f"Unexpected read of non-200 body: {limit}")

    reply = UnreadableResponse(b"DO_NOT_LOG_THIS", status=202)
    stub_http(monkeypatch, reply)
    result = send()
    assert result == HttpResult(status_code=202)
    assert reply.closed


def test_http_errors_never_read_or_forward_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnreadableBody(BytesIO):
        def read(self, *_args: object) -> bytes:
            raise AssertionError("Provider error body should not be read")

    body = UnreadableBody(b"SECRET_API_ERROR_BODY")
    exc = HTTPError(
        "https://api.groq.com/openai/v1/chat/completions",
        429,
        "SECRET_HTTP_MESSAGE",
        {},
        body,
    )
    stub_http(monkeypatch, exc)
    result = send()
    assert result == HttpResult(status_code=429)
    assert exc.closed
    assert body.closed
    assert "SECRET_API_ERROR_BODY" not in str(result)


def test_redirect_handler_rejects_sensitive_forwarding() -> None:
    handler = transport_module._RejectRedirects()
    original = Request("https://api.groq.com/openai/v1/chat/completions")
    redirected = handler.redirect_request(
        original,
        None,
        302,
        "Location: other host",
        {},
        "https://attacker.invalid/collect",
    )
    assert redirected is None


def test_transient_409_is_bounded_retryable_and_redacted() -> None:
    request = ModelRequest(
        run_id="run",
        case_id="case",
        trace_id="trace",
        attempt_id="attempt",
        provider="groq",
        model="synthetic-small",
        prompt="fixture",
        timeout_seconds=3.5,
    )
    mapped = GroqResponseMapper().map(
        request,
        HttpResult(status_code=409, body={"error": "PRIVATE_REMOTE_DETAILS"}),
        elapsed_ms=1.0,
    )
    assert mapped.status is CallStatus.FAILED
    assert mapped.error is not None
    assert mapped.error.code == "GROQ_CONFLICT"
    assert mapped.error.category is ErrorCategory.PROVIDER
    assert mapped.error.retryable
    assert "PRIVATE_REMOTE_DETAILS" not in mapped.model_dump_json()
