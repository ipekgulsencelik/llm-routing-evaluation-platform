"""In-memory contract tests for a cancellation-cooperative async HTTP transport.

These tests do not import HTTPX, use sockets, or require API credentials.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest

from llm_routing_eval.adapters.providers import (
    AsyncClientGroqTransport,
    DenyAllOutbound,
    GroqChatProvider,
    GroqProviderOptions,
    HttpResult,
)
from llm_routing_eval.adapters.providers import transport as transport_module
from llm_routing_eval.adapters.providers.response_mapping import GroqResponseMapper
from llm_routing_eval.core import CallStatus, ErrorCategory, ModelRequest


class FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        chunks: tuple[bytes, ...] = (),
        headers: Mapping[str, str] | None = None,
        fail_on_body: bool = False,
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks
        self.closed = False
        self.read_chunks = 0
        self.requested_chunk_size: int | None = None
        self.fail_on_body = fail_on_body

    async def aiter_bytes(self, chunk_size: int = 65_536) -> AsyncIterator[bytes]:
        self.requested_chunk_size = chunk_size
        if self.fail_on_body:
            raise AssertionError("Unsafe read of non-200 response body")
        for chunk in self.chunks:
            self.read_chunks += 1
            yield chunk


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, Mapping[str, object]]] = []

    def stream(
        self,
        method: str,
        url: str,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> AbstractAsyncContextManager[FakeResponse]:
        assert not follow_redirects
        assert timeout == 2.0
        assert headers["Authorization"] == "Bearer SYNTHETIC_CREDENTIAL"
        assert headers["Accept-Encoding"] == "identity"
        self.calls.append((method, url, json))

        @asynccontextmanager
        async def context() -> AsyncIterator[FakeResponse]:
            try:
                yield self.response
            finally:
                self.response.closed = True

        return context()


def send(response: FakeResponse) -> tuple[HttpResult, FakeClient]:
    client = FakeClient(response)
    result = asyncio.run(
        AsyncClientGroqTransport(client).post_chat(
            {"model": "synthetic-small", "messages": [{"role": "user", "content": "test"}]},
            api_key="SYNTHETIC_CREDENTIAL",
            timeout_seconds=2.0,
        )
    )
    return result, client


def test_async_transport_parses_streamed_success_and_closes_context() -> None:
    payload = {"choices": [{"message": {"content": "synthetic answer"}}]}
    data = json.dumps(payload).encode()
    response = FakeResponse(200, chunks=(data[:11], data[11:]))
    result, client = send(response)
    assert result == HttpResult(200, body=payload)
    assert client.calls == [
        (
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": "synthetic-small", "messages": [{"role": "user", "content": "test"}]},
        )
    ]
    assert response.requested_chunk_size == 65_536
    assert response.read_chunks == 2
    assert response.closed


@pytest.mark.parametrize(
    "raw",
    [b"", b"not JSON", b"null", b"[]", b'"string"', b"\xff\xff"],
)
def test_invalid_200_body_is_a_nonretryable_completion_failure(raw: bytes) -> None:
    response = FakeResponse(200, chunks=(raw,))
    result, _ = send(response)
    assert result == HttpResult(200)
    assert response.closed
    outgoing = ModelRequest(
        run_id="run",
        trace_id="trace",
        case_id="case",
        attempt_id="attempt",
        provider="groq",
        model="synthetic-small",
        prompt="test",
        timeout_seconds=2,
    )
    mapped = GroqResponseMapper().map(outgoing, result, elapsed_ms=2)
    assert mapped.status is CallStatus.FAILED
    assert mapped.error is not None and mapped.error.code == "GROQ_INVALID_RESPONSE"
    assert not mapped.error.retryable


def test_oversize_response_is_rejected_early_and_context_closed() -> None:
    response = FakeResponse(
        200,
        chunks=(b"x" * 10, b"a" * transport_module._MAX_RESPONSE_BYTES, b"SECRET_UNREAD"),
    )
    result, _ = send(response)
    assert result == HttpResult(200)
    assert response.read_chunks == 2
    assert response.closed


@pytest.mark.parametrize(
    ("status", "header", "expected"),
    [
        (429, "4.5", 4.5),
        (429, "86401", 86_400.0),
        (429, "invalid-header", None),
        (400, "4.5", None),
        (503, "4.5", None),
        (302, "4.5", None),
    ],
)
def test_non_200_never_reads_provider_body_and_only_429_extracts_hint(
    status: int, header: str, expected: float | None
) -> None:
    response = FakeResponse(
        status,
        headers={"retry-after": header},
        chunks=(b"DO_NOT_READ_PROVIDER_ERROR_TEXT",),
        fail_on_body=True,
    )
    result, _ = send(response)
    assert result == HttpResult(status, retry_after_seconds=expected)
    assert response.read_chunks == 0
    assert response.closed


def test_transport_cancel_closes_in_flight_stream_without_swallowing_cancel() -> None:
    class WaitingResponse(FakeResponse):
        def __init__(self) -> None:
            super().__init__(200)
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def aiter_bytes(self, chunk_size: int = 65_536) -> AsyncIterator[bytes]:
            _ = chunk_size
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            yield b"never reached"

    async def run() -> bool:
        response = WaitingResponse()
        client = FakeClient(response)
        task = asyncio.create_task(
            AsyncClientGroqTransport(client).post_chat(
                {"model": "synthetic"}, api_key="SYNTHETIC_CREDENTIAL", timeout_seconds=2.0
            )
        )
        await asyncio.wait_for(response.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return response.closed and response.cancelled.is_set()

    assert asyncio.run(run())


def test_provider_can_consume_streaming_transport_with_safe_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixtureApproval:
        def authorize(self, request: ModelRequest) -> bool:
            return request.prompt == "synthetic fixture"

    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_CREDENTIAL")
    payload = {
        "model": "synthetic-small",
        "choices": [{"message": {"content": "a synthetic response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 5},
    }
    client = FakeClient(FakeResponse(200, chunks=(json.dumps(payload).encode(),)))
    provider = GroqChatProvider(
        GroqProviderOptions(allowed_models=frozenset({"synthetic-small"}), live_enabled=True),
        transport=AsyncClientGroqTransport(client),
        outbound_authorization=FixtureApproval(),
    )
    outgoing = ModelRequest(
        run_id="run",
        trace_id="trace",
        case_id="case",
        attempt_id="attempt",
        provider="groq",
        model="synthetic-small",
        prompt="synthetic fixture",
        timeout_seconds=2,
    )
    result = asyncio.run(provider.generate(outgoing))
    assert result.status is CallStatus.SUCCESS
    assert result.usage is not None and result.usage.input_tokens == 7
    assert client.response.closed

    blocked_client = FakeClient(FakeResponse(200, chunks=(json.dumps(payload).encode(),)))
    denied = GroqChatProvider(
        GroqProviderOptions(allowed_models=frozenset({"synthetic-small"}), live_enabled=True),
        transport=AsyncClientGroqTransport(blocked_client),
        outbound_authorization=DenyAllOutbound(),
    )
    refused = asyncio.run(denied.generate(outgoing))
    assert refused.status is CallStatus.FAILED
    assert refused.error is not None and refused.error.category is ErrorCategory.PRIVACY
    assert not blocked_client.calls


def test_transport_exception_is_redacted_at_provider_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixtureApproval:
        def authorize(self, request: ModelRequest) -> bool:
            return True

    class ExplodingClient(FakeClient):
        def stream(
            self,
            method: str,
            url: str,
            *,
            json: Mapping[str, object],
            headers: Mapping[str, str],
            timeout: float,
            follow_redirects: bool,
        ) -> AbstractAsyncContextManager[FakeResponse]:
            raise RuntimeError("SECRET_EXCEPTION_WITH_PRIVATE_HEADERS")

    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "SYNTHETIC_CREDENTIAL")
    provider = GroqChatProvider(
        GroqProviderOptions(allowed_models=frozenset({"synthetic-small"}), live_enabled=True),
        transport=AsyncClientGroqTransport(ExplodingClient(FakeResponse(200))),
        outbound_authorization=FixtureApproval(),
    )
    result = asyncio.run(
        provider.generate(
            ModelRequest(
                run_id="run",
                trace_id="trace",
                case_id="case",
                attempt_id="attempt",
                provider="groq",
                model="synthetic-small",
                prompt="fixture",
                timeout_seconds=2,
            )
        )
    )
    assert result.error is not None and result.error.code == "GROQ_TRANSPORT_ERROR"
    assert result.error.retryable
    assert "SECRET_EXCEPTION" not in result.model_dump_json()
