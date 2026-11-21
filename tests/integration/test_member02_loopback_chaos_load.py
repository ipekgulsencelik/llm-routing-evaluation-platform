"""Bounded localhost TCP stress regressions for HTTPX; never contact Groq.

This is a resource-safety / correctness gate, not throughput benchmarking.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx

from llm_routing_eval.adapters.providers.httpx_runtime import (
    HttpxPoolSettings,
    HttpxStreamingClient,
)

LocalHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]
_PAYLOAD = {"model": "synthetic-model", "prompt": "synthetic load-test only"}
_HEADERS = {"Content-Type": "application/json"}


@asynccontextmanager
async def _loopback(handler: LocalHandler) -> AsyncIterator[str]:
    """Isolate the test server to an ephemeral IPv4 loopback socket."""
    active: set[asyncio.Task[None]] = set()

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            active.add(task)
        try:
            await handler(reader, writer)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            if task is not None:
                active.discard(task)

    server = await asyncio.start_server(accept, host="127.0.0.1", port=0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/synthetic"
    finally:
        server.close()
        await server.wait_closed()
        for task in tuple(active):
            task.cancel()
        if active:
            await asyncio.gather(*tuple(active), return_exceptions=True)


async def _read_request(reader: asyncio.StreamReader) -> None:
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2.0)
    length = 0
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    assert length <= 4096
    if length:
        await asyncio.wait_for(reader.readexactly(length), timeout=2.0)


def _headers(size: int) -> bytes:
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(size).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
    )


def _request(client: HttpxStreamingClient, url: str):
    return client.stream(
        "POST",
        url,
        json=_PAYLOAD,
        headers=_HEADERS,
        timeout=3.0,
        follow_redirects=False,
    )


def _settings(max_connections: int) -> HttpxPoolSettings:
    return HttpxPoolSettings(
        max_connections=max_connections,
        max_keepalive_connections=0,
        connect_timeout_seconds=1.0,
        pool_timeout_seconds=2.0,
    )


def test_burst_of_16_real_loopback_requests_obeys_four_connection_limit() -> None:
    """No connection beyond four should be admitted while four are held."""

    async def run() -> tuple[int, int, int, list[int]]:
        entered = asyncio.Event()
        release = asyncio.Event()
        active_handlers = 0
        peak = 0
        accepted = 0
        body = json.dumps({"status": "synthetic-ok"}).encode("utf-8")

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal active_handlers, peak, accepted
            await _read_request(reader)
            accepted += 1
            active_handlers += 1
            peak = max(peak, active_handlers)
            if active_handlers == 4:
                entered.set()
            try:
                await release.wait()
            finally:
                active_handlers -= 1
            writer.write(_headers(len(body)) + body)
            await writer.drain()

        async with _loopback(handler) as url:
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=0),
            ) as native:
                client = HttpxStreamingClient(native, _settings(4))

                async def one() -> int:
                    async with _request(client, url) as response:
                        data = b"".join([chunk async for chunk in response.aiter_bytes()])
                        assert json.loads(data)["status"] == "synthetic-ok"
                        return response.status_code

                tasks = [asyncio.create_task(one()) for _ in range(16)]
                try:
                    await asyncio.wait_for(entered.wait(), timeout=3.0)
                    assert peak == 4
                    release.set()
                    statuses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
                finally:
                    release.set()
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        return accepted, peak, active_handlers, statuses

    accepted, peak, remaining, status_codes = asyncio.run(run())
    assert accepted == len(status_codes) == 16
    assert peak <= 4
    assert remaining == 0
    assert status_codes == [200] * 16


def test_cancellation_storm_closes_all_tcp_streams_then_pool_recovers() -> None:
    """Cancel three blocked readers and prove a fourth request can still finish."""

    async def run() -> tuple[int, int, bool]:
        started = asyncio.Event()
        all_disconnected = asyncio.Event()
        entered = 0
        disconnected = 0
        accepted = 0
        total_blocked = 3

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            nonlocal entered, disconnected, accepted
            await _read_request(reader)
            accepted += 1
            if accepted <= total_blocked:
                writer.write(_headers(100) + b"x")
                await writer.drain()
                try:
                    assert await asyncio.wait_for(reader.read(1), timeout=3.0) == b""
                finally:
                    disconnected += 1
                    if disconnected == total_blocked:
                        all_disconnected.set()
            else:
                body = b'{"status":"recovered"}'
                writer.write(_headers(len(body)) + body)
                await writer.drain()

        async with _loopback(handler) as url:
            async with httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                limits=httpx.Limits(max_connections=3, max_keepalive_connections=0),
            ) as native:
                client = HttpxStreamingClient(native, _settings(3))

                async def stalled_reader() -> None:
                    nonlocal entered
                    async with _request(client, url) as response:
                        async for _ in response.aiter_bytes(chunk_size=1):
                            entered += 1
                            if entered == total_blocked:
                                started.set()

                tasks = [asyncio.create_task(stalled_reader()) for _ in range(total_blocked)]
                try:
                    await asyncio.wait_for(started.wait(), timeout=3.0)
                    for task in tasks:
                        task.cancel()
                    stopped = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=3.0
                    )
                    assert all(isinstance(result, asyncio.CancelledError) for result in stopped)
                    await asyncio.wait_for(all_disconnected.wait(), timeout=3.0)
                    async with _request(client, url) as response:
                        body = b"".join([chunk async for chunk in response.aiter_bytes()])
                        recovered = (
                            response.status_code == 200
                            and json.loads(body)["status"] == "recovered"
                        )
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        return accepted, disconnected, recovered

    accepted, disconnected, recovered = asyncio.run(run())
    assert accepted == 4
    assert disconnected == 3
    assert recovered
