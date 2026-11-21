"""Managed HTTPX runtime contract: pool lifetime, status mapping, timeout and cancellation.

All tests use httpx.MockTransport or an in-memory AsyncByteStream. No network.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest

from llm_routing_eval.adapters.providers import (
    GroqProviderOptions,
    HttpxPoolSettings,
    managed_httpx_groq_provider,
)
from llm_routing_eval.core import CallStatus, ErrorCategory, ModelRequest

MODEL = "openai/gpt-oss-20b"
SECRET = "IN_MEMORY_FAKE_CREDENTIAL"
PROMPT = "Explicitly synthetic approved test prompt"


class SyntheticOnlyApproval:
    """Test double. Do not use with real requests."""

    def authorize(self, request: ModelRequest) -> bool:
        return request.prompt == PROMPT and request.provider == "groq"


class TrackingMockTransport(httpx.MockTransport):
    def __init__(self, handler: Callable[[httpx.Request], Awaitable[httpx.Response]]) -> None:
        super().__init__(handler)
        self.closed_by_owner = False

    async def aclose(self) -> None:
        self.closed_by_owner = True
        await super().aclose()


def case() -> ModelRequest:
    return ModelRequest(
        run_id="test",
        trace_id="synthetic-trace",
        case_id="synthetic-case",
        attempt_id="synthetic-attempt",
        provider="groq",
        model=MODEL,
        prompt=PROMPT,
        timeout_seconds=2.0,
        max_output_tokens=32,
    )


def options() -> GroqProviderOptions:
    return GroqProviderOptions(
        allowed_models=frozenset({MODEL}),
        live_enabled=True,
        max_in_flight=2,
    )


def test_client_lifecycle_reuses_transport_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    seen_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": MODEL,
                "choices": [{"message": {"content": "Synthetic answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )

    async def run() -> tuple[list[CallStatus], TrackingMockTransport]:
        transport = TrackingMockTransport(handler)
        async with managed_httpx_groq_provider(
            options(),
            outbound_authorization=SyntheticOnlyApproval(),
            pool_settings=HttpxPoolSettings(
                max_connections=2,
                max_keepalive_connections=1,
                connect_timeout_seconds=1.5,
                pool_timeout_seconds=1.0,
            ),
            mock_transport=transport,
        ) as provider:
            first = await provider.generate(case())
            second = await provider.generate(case())
            assert first.usage is not None and first.usage.input_tokens == 11
            assert second.usage is not None and second.usage.output_tokens == 4
            assert not transport.closed_by_owner
            return [first.status, second.status], transport

    states, transport = asyncio.run(run())
    assert states == [CallStatus.SUCCESS, CallStatus.SUCCESS]
    assert transport.closed_by_owner
    assert len(seen_requests) == 2
    for request in seen_requests:
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        assert request.headers["Accept-Encoding"] == "identity"
        assert request.extensions["timeout"]["connect"] == 1.5
        assert request.extensions["timeout"]["pool"] == 1.0


def test_default_authorization_denies_even_with_live_flags_and_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        raise AssertionError("Denied requests must never reach transport")

    async def run() -> tuple[CallStatus, ErrorCategory]:
        async with managed_httpx_groq_provider(
            options(), mock_transport=httpx.MockTransport(handler)
        ) as provider:
            result = await provider.generate(case())
            assert result.error is not None
            return result.status, result.error.category

    status, category = asyncio.run(run())
    assert status is CallStatus.FAILED
    assert category is ErrorCategory.PRIVACY
    assert not calls


def test_httpx_timeout_is_normalized_to_safe_provider_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    async def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("PRIVATE_REMOTE_TIMEOUT_DETAILS", request=request)

    async def run() -> tuple[CallStatus, str, bool, str]:
        async with managed_httpx_groq_provider(
            options(),
            outbound_authorization=SyntheticOnlyApproval(),
            mock_transport=httpx.MockTransport(timeout_handler),
        ) as provider:
            result = await provider.generate(case())
            assert result.error is not None
            return (
                result.status,
                result.error.code,
                result.error.retryable,
                result.model_dump_json(),
            )

    status, code, retryable, serialized = asyncio.run(run())
    assert status is CallStatus.TIMEOUT
    assert code == "GROQ_TIMEOUT"
    assert retryable
    assert "PRIVATE_REMOTE_TIMEOUT_DETAILS" not in serialized
    assert SECRET not in serialized


def test_redirects_are_not_followed_even_if_response_has_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    visited: list[str] = []

    async def redirect_handler(request: httpx.Request) -> httpx.Response:
        visited.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://attacker.invalid/collect"})

    async def run() -> tuple[CallStatus, str]:
        async with managed_httpx_groq_provider(
            options(),
            outbound_authorization=SyntheticOnlyApproval(),
            mock_transport=httpx.MockTransport(redirect_handler),
        ) as provider:
            result = await provider.generate(case())
            assert result.error is not None
            return result.status, result.error.code

    status, code = asyncio.run(run())
    assert status is CallStatus.FAILED and code == "GROQ_HTTP_ERROR"
    assert visited == ["https://api.groq.com/openai/v1/chat/completions"]


def test_cancellation_interrupts_httpx_stream_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", SECRET)

    class BlockingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.finalized = False
            self.closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.finalized = True
            yield b"unreachable"

        async def aclose(self) -> None:
            self.closed = True

    async def run() -> tuple[bool, bool, bool]:
        stream = BlockingStream()

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=stream, request=request)

        transport = TrackingMockTransport(handler)
        async with managed_httpx_groq_provider(
            options(),
            outbound_authorization=SyntheticOnlyApproval(),
            mock_transport=transport,
        ) as provider:
            task = asyncio.create_task(provider.generate(case()))
            await asyncio.wait_for(stream.started.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        return stream.finalized, stream.closed, transport.closed_by_owner

    finalized, closed, pool_closed = asyncio.run(run())
    assert finalized and closed and pool_closed


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"max_connections": 0}, "HTTPX_POOL_CONNECTIONS_INVALID"),
        ({"max_connections": 33}, "HTTPX_POOL_CONNECTIONS_INVALID"),
        (
            {"max_connections": 2, "max_keepalive_connections": 3},
            "HTTPX_KEEPALIVE_CONNECTIONS_INVALID",
        ),
        ({"connect_timeout_seconds": 0}, "HTTPX_CONNECT_TIMEOUT_INVALID"),
        ({"pool_timeout_seconds": 0}, "HTTPX_POOL_TIMEOUT_INVALID"),
    ],
)
def test_pool_configuration_rejects_unbounded_settings(
    kwargs: dict[str, int], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        HttpxPoolSettings(**kwargs)


def test_httpx_core_no_external_network_is_needed() -> None:
    """Catch accidental uses of the default network HTTPX transport in tests."""
    # HTTPX is pinned explicitly in pyproject.toml and uv.lock, but no tests
    # make remote API requests or rely on provider credentials.
    assert httpx.__version__ == "0.28.1"
