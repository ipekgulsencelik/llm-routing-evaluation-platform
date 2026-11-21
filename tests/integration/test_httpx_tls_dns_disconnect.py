"""Native socket TLS, DNS and connection-abort tests; offline and loopback only.

Ephemeral fake certificates are created by trustme at test time. No TLS keys,
tokens, customer prompts or provider endpoints are committed to this project.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
import trustme

from llm_routing_eval.adapters.providers import GroqChatProvider, GroqProviderOptions
from llm_routing_eval.adapters.providers.httpx_runtime import (
    HttpxPoolSettings,
    HttpxStreamingClient,
)
from llm_routing_eval.adapters.providers.transport import (
    GroqNetworkError,
    GroqTLSVerificationError,
)
from llm_routing_eval.core import CallStatus, ModelRequest

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
_BODY = {"model": "local-synthetic", "messages": [{"role": "user", "content": "fixture"}]}
_HEADERS = {"Content-Type": "application/json"}


async def _drain_request(reader: asyncio.StreamReader) -> None:
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2)
    length = 0
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    assert length <= 4096
    if length:
        await asyncio.wait_for(reader.readexactly(length), timeout=2)


@asynccontextmanager
async def _server(handler: Handler, *, context: ssl.SSLContext | None = None) -> AsyncIterator[str]:
    """All socket activity stays on ephemeral 127.0.0.1 ports."""
    tasks: set[asyncio.Task[None]] = set()

    async def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        current = asyncio.current_task()
        if current is not None:
            tasks.add(current)
        try:
            await handler(reader, writer)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass
            if current is not None:
                tasks.discard(current)

    server = await asyncio.start_server(accepted, host="127.0.0.1", port=0, ssl=context)
    port = server.sockets[0].getsockname()[1]
    scheme = "https" if context is not None else "http"
    try:
        yield f"{scheme}://127.0.0.1:{port}/synthetic"
    finally:
        server.close()
        await server.wait_closed()
        for task in tuple(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tuple(tasks), return_exceptions=True)


def _client(native: httpx.AsyncClient) -> HttpxStreamingClient:
    return HttpxStreamingClient(
        native,
        HttpxPoolSettings(
            max_connections=2,
            max_keepalive_connections=0,
            connect_timeout_seconds=0.8,
            pool_timeout_seconds=0.5,
        ),
    )


def _stream(client: HttpxStreamingClient, url: str):
    return client.stream(
        "POST",
        url,
        json=_BODY,
        headers=_HEADERS,
        timeout=1.0,
        follow_redirects=False,
    )


def _tls_server(cert: trustme.LeafCert) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert.configure_cert(ctx)
    return ctx


def test_verified_localhost_tls_connects_with_explicit_test_ca() -> None:
    async def run() -> None:
        ca = trustme.CA()
        cert = ca.issue_cert("127.0.0.1")
        client_ctx = ssl.create_default_context()
        ca.configure_trust(client_ctx)
        assert client_ctx.check_hostname
        assert client_ctx.verify_mode is ssl.CERT_REQUIRED
        accepted = 0

        async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal accepted
            await _drain_request(reader)
            accepted += 1
            response = b'{"status":"synthetic"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: "
                + str(len(response)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + response
            )
            await writer.drain()

        async with _server(respond, context=_tls_server(cert)) as url:
            async with httpx.AsyncClient(
                verify=client_ctx, trust_env=False, follow_redirects=False
            ) as native:
                async with _stream(_client(native), url) as response:
                    assert response.status_code == 200
                    data = b"".join([chunk async for chunk in response.aiter_bytes()])
                    assert data == b'{"status":"synthetic"}'
        assert accepted == 1

    asyncio.run(run())


@pytest.mark.parametrize("cause", ["untrusted_ca", "hostname_mismatch"])
def test_local_tls_verification_failure_is_redacted_and_non_retryable(cause: str) -> None:
    async def run() -> None:
        ca = trustme.CA()
        cert = ca.issue_cert("localhost" if cause == "hostname_mismatch" else "127.0.0.1")
        client_ctx = ssl.create_default_context()
        if cause == "hostname_mismatch":
            ca.configure_trust(client_ctx)
        accepted = 0

        async def should_not_receive(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            nonlocal accepted
            accepted += 1
            raise AssertionError("TLS-verification failures must prevent HTTP data transfer")

        async with _server(should_not_receive, context=_tls_server(cert)) as url:
            async with httpx.AsyncClient(verify=client_ctx, trust_env=False) as native:
                with pytest.raises(
                    GroqTLSVerificationError, match="GROQ_TLS_VERIFICATION_FAILED"
                ) as failure:
                    async with _stream(_client(native), url):
                        raise AssertionError("Unverified TLS unexpectedly connected")
                assert "127.0.0.1" not in str(failure.value)
        assert accepted == 0

    asyncio.run(run())


def test_dns_failure_is_local_synthetic_resolution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_host = "member02-fail-closed.test.invalid"
    calls: list[str] = []
    original = socket.getaddrinfo

    def isolated_dns(host: str | bytes, port: object, *args: object, **kwargs: object):
        if host == invalid_host or host == invalid_host.encode():
            calls.append(invalid_host)
            raise socket.gaierror(socket.EAI_NONAME, "synthetic offline DNS failure")
        return original(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", isolated_dns)

    async def run() -> None:
        async with httpx.AsyncClient(trust_env=False) as native:
            url = f"https://{invalid_host}:443/synthetic"
            with pytest.raises(GroqNetworkError, match="GROQ_NETWORK_ERROR") as failure:
                async with _stream(_client(native), url):
                    raise AssertionError("DNS was not denied")
            assert invalid_host not in str(failure.value)

    asyncio.run(run())
    assert calls == [invalid_host]


@pytest.mark.parametrize("phase", ["before_headers", "partial_body"])
def test_loopback_remote_disconnect_is_redacted_network_failure(phase: str) -> None:
    async def run() -> None:
        accepted = 0

        async def disconnect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal accepted
            await _drain_request(reader)
            accepted += 1
            if phase == "partial_body":
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 200\r\nConnection: close\r\n\r\npartial"
                )
                await writer.drain()
            # Close the socket with an incomplete response, not a valid 200.

        async with _server(disconnect) as url:
            async with httpx.AsyncClient(trust_env=False) as native:
                with pytest.raises(GroqNetworkError, match="GROQ_NETWORK_ERROR") as failure:
                    async with _stream(_client(native), url) as response:
                        async for _ in response.aiter_bytes():
                            pass
                assert url not in str(failure.value)
        assert accepted == 1

    asyncio.run(run())


def test_provider_maps_tls_and_network_errors_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "DO_NOT_PRINT_SYNTHETIC_TEST_KEY")

    class SyntheticApproval:
        def authorize(self, request: ModelRequest) -> bool:
            return request.prompt == "synthetic test prompt"

    class BrokenTransport:
        def __init__(self, failure: Exception) -> None:
            self.failure = failure

        async def post_chat(
            self, body: dict[str, object], *, api_key: str, timeout_seconds: float
        ) -> None:
            raise self.failure

    def model_request() -> ModelRequest:
        return ModelRequest(
            run_id="run",
            trace_id="trace",
            case_id="case",
            attempt_id="attempt",
            provider="groq",
            model="fake/model",
            prompt="synthetic test prompt",
            timeout_seconds=1,
        )

    for error, expected, retry in [
        (GroqTLSVerificationError("private cert chain"), "GROQ_TLS_VERIFICATION_FAILED", False),
        (GroqNetworkError("private remote address"), "GROQ_NETWORK_ERROR", True),
    ]:
        provider = GroqChatProvider(
            GroqProviderOptions(allowed_models=frozenset({"fake/model"}), live_enabled=True),
            transport=BrokenTransport(error),
            outbound_authorization=SyntheticApproval(),
        )
        result = asyncio.run(provider.generate(model_request()))
        assert result.status is CallStatus.FAILED
        assert result.error is not None
        assert result.error.code == expected
        assert result.error.retryable is retry
        assert "private" not in result.model_dump_json()
        assert "DO_NOT_PRINT_SYNTHETIC_TEST_KEY" not in result.model_dump_json()
