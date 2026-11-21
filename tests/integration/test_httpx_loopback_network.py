"""Loopback TCP integration checks for HTTPX lifecycle and cancellation.

Unlike MockTransport tests these use real localhost sockets. They do not
connect to Groq, transmit banking data, or bypass Groq's privacy controls.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
import pytest

from llm_routing_eval.adapters.providers.httpx_runtime import (
    HttpxPoolSettings,
    HttpxStreamingClient,
)

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
_TEST_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
_TEST_BODY = {"model": "synthetic-fixture", "prompt": "not customer data"}


@asynccontextmanager
async def _loopback_server(handler: Handler) -> AsyncIterator[str]:
    """Own a fresh localhost-only server and always clean its active tasks."""
    tasks: set[asyncio.Task[None]] = set()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        active = asyncio.current_task()
        if active is not None:
            tasks.add(active)
        try:
            await handler(reader, writer)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            if active is not None:
                tasks.discard(active)

    server = await asyncio.start_server(accept, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/synthetic"
    finally:
        server.close()
        await server.wait_closed()
        for task in tuple(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tuple(tasks), return_exceptions=True)


async def _consume_request(reader: asyncio.StreamReader) -> None:
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2.0)
    length = 0
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    assert length <= 4096
    if length:
        await asyncio.wait_for(reader.readexactly(length), timeout=2.0)


def _write_headers(writer: asyncio.StreamWriter, *, body_size: int) -> None:
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        + b"Content-Type: application/json\r\n"
        + b"Content-Length: "
        + str(body_size).encode("ascii")
        + b"\r\nConnection: keep-alive\r\n\r\n"
    )


def _pool(*, connections: int = 2, pool_timeout: float = 0.25) -> HttpxPoolSettings:
    return HttpxPoolSettings(
        max_connections=connections,
        max_keepalive_connections=0,
        connect_timeout_seconds=1.0,
        pool_timeout_seconds=pool_timeout,
    )


def _stream(client: HttpxStreamingClient, url: str, *, timeout: float = 1.0):
    return client.stream(
        "POST",
        url,
        json=_TEST_BODY,
        headers=_TEST_HEADERS,
        timeout=timeout,
        follow_redirects=False,
    )


def test_real_loopback_tcp_json_response_and_shared_client() -> None:
    async def run() -> None:
        request_count = 0
        content = json.dumps({"kind": "synthetic", "count": 1}).encode("utf-8")

        async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal request_count
            await _consume_request(reader)
            request_count += 1
            _write_headers(writer, body_size=len(content))
            writer.write(content)
            await writer.drain()

        async with _loopback_server(respond) as url:
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=0),
            ) as native:
                client = HttpxStreamingClient(native, _pool())
                for _ in range(2):
                    async with _stream(client, url) as response:
                        assert response.status_code == 200
                        raw = b"".join([chunk async for chunk in response.aiter_bytes()])
                        assert json.loads(raw)["kind"] == "synthetic"
            assert native.is_closed
        assert request_count == 2

    asyncio.run(run())


def test_real_loopback_read_timeout_closes_socket() -> None:
    async def run() -> None:
        disconnected = asyncio.Event()

        async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _consume_request(reader)
            _write_headers(writer, body_size=10)
            writer.write(b"{")
            await writer.drain()
            try:
                assert await asyncio.wait_for(reader.read(1), timeout=2.0) == b""
            finally:
                disconnected.set()

        async with _loopback_server(stall) as url:
            async with httpx.AsyncClient(trust_env=False) as native:
                client = HttpxStreamingClient(native, _pool())
                with pytest.raises(TimeoutError, match="GROQ_HTTP_TIMEOUT"):
                    async with _stream(client, url, timeout=0.2) as response:
                        async for _ in response.aiter_bytes(chunk_size=1):
                            pass
                await asyncio.wait_for(disconnected.wait(), timeout=2.0)

    asyncio.run(run())


def test_real_loopback_cancel_during_body_closes_socket() -> None:
    async def run() -> None:
        disconnected = asyncio.Event()
        first_byte_sent = asyncio.Event()
        first_byte_received = asyncio.Event()

        async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _consume_request(reader)
            _write_headers(writer, body_size=10)
            writer.write(b"x")
            await writer.drain()
            first_byte_sent.set()
            try:
                assert await asyncio.wait_for(reader.read(1), timeout=2.0) == b""
            finally:
                disconnected.set()

        async with _loopback_server(stall) as url:
            async with httpx.AsyncClient(trust_env=False) as native:
                client = HttpxStreamingClient(native, _pool())

                async def consume() -> None:
                    async with _stream(client, url) as response:
                        async for _ in response.aiter_bytes(chunk_size=1):
                            first_byte_received.set()

                task = asyncio.create_task(consume())
                try:
                    await asyncio.wait_for(first_byte_sent.wait(), timeout=2.0)
                    await asyncio.wait_for(first_byte_received.wait(), timeout=2.0)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    await asyncio.wait_for(disconnected.wait(), timeout=2.0)
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_real_loopback_pool_saturation_maps_to_bounded_safe_timeout() -> None:
    async def run() -> None:
        first_headers = asyncio.Event()
        release_first = asyncio.Event()
        accepted = 0

        async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal accepted
            await _consume_request(reader)
            accepted += 1
            _write_headers(writer, body_size=10)
            await writer.drain()
            first_headers.set()
            await release_first.wait()

        async with _loopback_server(hold) as url:
            async with httpx.AsyncClient(
                trust_env=False,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            ) as native:
                client = HttpxStreamingClient(native, _pool(connections=1, pool_timeout=0.15))
                entered = asyncio.Event()

                async def occupy() -> None:
                    async with _stream(client, url):
                        entered.set()
                        await release_first.wait()

                first_task = asyncio.create_task(occupy())
                try:
                    await asyncio.wait_for(first_headers.wait(), timeout=2.0)
                    await asyncio.wait_for(entered.wait(), timeout=2.0)
                    with pytest.raises(TimeoutError, match="GROQ_HTTP_TIMEOUT"):
                        async with _stream(client, url):
                            raise AssertionError("Pool must not allow another socket")
                    assert accepted == 1
                finally:
                    release_first.set()
                    await asyncio.wait_for(first_task, timeout=2.0)

    asyncio.run(run())
