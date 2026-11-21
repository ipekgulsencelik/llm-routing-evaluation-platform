"""Retry-After parsing, bounds, and outbound cancel propagation. No networking."""

import asyncio
from collections.abc import Mapping
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest
from pydantic import ValidationError

from llm_routing_eval.adapters.providers import GroqChatProvider, GroqProviderOptions, HttpResult
from llm_routing_eval.adapters.providers import transport as transport_module
from llm_routing_eval.adapters.providers.response_mapping import GroqResponseMapper
from llm_routing_eval.core import (
    CallStatus,
    ErrorCategory,
    ErrorInfo,
    ModelRequest,
)

MODEL = "openai/gpt-oss-20b"


def request() -> ModelRequest:
    return ModelRequest(
        run_id="run",
        case_id="case",
        trace_id="trace",
        attempt_id="attempt",
        provider="groq",
        model=MODEL,
        prompt="A deliberately synthetic prompt",
        timeout_seconds=2,
    )


class PermitSynthetic:
    def authorize(self, outgoing: ModelRequest) -> bool:
        return outgoing.prompt == "A deliberately synthetic prompt"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("0", 0.0),
        ("0.125", 0.125),
        ("2", 2.0),
        ("9.5", 9.5),
        ("86400", 86400.0),
        ("86401", None),
        ("999999999999999999999", None),
        ("-1", None),
        ("NaN", None),
        ("inf", None),
        ("1e20", None),
        ("15, 20", None),
        ("Sat, 21 Nov 2026 12:00:00 GMT", None),
        ("3\nX-Secret: leaked", None),
        ("", None),
    ],
)
def test_retry_after_is_only_a_valid_bounded_numeric_seconds_hint(
    header: str, expected: float | None
) -> None:
    headers = Message()
    headers["Retry-After"] = header
    assert transport_module._retry_after_seconds(headers) == expected


@pytest.mark.parametrize("status", [200, 400, 409, 500])
def test_mapper_never_propagates_retry_hint_for_non_rate_limit(status: int) -> None:
    response = GroqResponseMapper().map(
        request(),
        HttpResult(status_code=status, retry_after_seconds=3.0),
        elapsed_ms=1.0,
    )
    assert response.error is not None
    assert response.error.retry_after_seconds is None


def test_mapper_preserves_only_safe_429_hint() -> None:
    valid = GroqResponseMapper().map(
        request(), HttpResult(429, retry_after_seconds=4.5), elapsed_ms=1.0
    )
    assert valid.status is CallStatus.RATE_LIMITED
    assert valid.error is not None and valid.error.retry_after_seconds == 4.5
    for invalid in (float("inf"), float("nan"), -1.0, 1e12):
        result = GroqResponseMapper().map(
            request(), HttpResult(429, retry_after_seconds=invalid), elapsed_ms=1.0
        )
        assert result.error is not None and result.error.retry_after_seconds is None


def test_core_refuses_retry_hint_for_non_retryable_errors() -> None:
    with pytest.raises(ValidationError, match="RETRY_HINT_REQUIRES_RETRYABLE_RATE_LIMIT"):
        ErrorInfo(category=ErrorCategory.PROVIDER, code="SYNTHETIC", retry_after_seconds=1.0)
    with pytest.raises(ValidationError):
        ErrorInfo(
            category=ErrorCategory.RATE_LIMIT,
            code="SYNTHETIC",
            retryable=True,
            retry_after_seconds=float("nan"),
        )


def test_429_http_error_reads_header_but_never_reads_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SecretBody(BytesIO):
        def read(self, *_args: object) -> bytes:
            raise AssertionError("Provider error body must remain unread")

    headers = Message()
    headers["Retry-After"] = "4.5"
    body = SecretBody(b"PRIVATE HTTP ERROR TEXT")
    error = HTTPError("https://api.groq.com", 429, "secret reason", headers, body)

    class ErrorOpener:
        def open(self, *_args: object, **_kwargs: object) -> None:
            raise error

    monkeypatch.setattr(transport_module, "build_opener", lambda *_: ErrorOpener())
    result = transport_module.UrllibGroqTransport._post({"model": MODEL}, "NOT_REAL", 2.0)
    assert result == HttpResult(status_code=429, retry_after_seconds=4.5)
    assert body.closed
    assert "PRIVATE" not in str(result)


def test_default_transport_cannot_cancel_os_thread_but_async_protocol_propagates_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_KEY")

    class BlockingTransport:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def post_chat(
            self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
        ) -> HttpResult:
            _ = body, api_key, timeout_seconds
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            return HttpResult(status_code=200)

    async def run() -> bool:
        transport = BlockingTransport()
        provider = GroqChatProvider(
            GroqProviderOptions(allowed_models=frozenset({MODEL}), live_enabled=True),
            transport=transport,
            outbound_authorization=PermitSynthetic(),
        )
        task = asyncio.create_task(provider.generate(request()))
        await transport.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return transport.cancelled.is_set()

    assert asyncio.run(run())
