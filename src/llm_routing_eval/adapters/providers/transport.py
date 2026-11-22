"""Replaceable Groq HTTP boundary with bounded bodies and no credential redirects."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_BUILD_CHAT_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
# Only reviewed, fixed HTTPS endpoints may receive credentials or prompts.
ALLOWED_CHAT_ENDPOINTS = frozenset({_ENDPOINT, NVIDIA_BUILD_CHAT_ENDPOINT})


def _checked_endpoint(endpoint: str) -> str:
    if endpoint not in ALLOWED_CHAT_ENDPOINTS:
        raise ValueError("CHAT_ENDPOINT_NOT_ALLOWLISTED")
    return endpoint


_MAX_RESPONSE_BYTES = 1_048_576
_NUMERIC_SECONDS = re.compile(r"^(?:0|[1-9][0-9]{0,31})(?:\.[0-9]{1,3})?$")


def _retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """Parse Groq's numeric Retry-After seconds without retaining raw headers.

    The app won't sleep longer than a separately configured retry budget.
    Unrecognized formats or excessive values intentionally do not become hints.
    """
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        value = headers.get("retry-after")
    if not isinstance(value, str) or _NUMERIC_SECONDS.fullmatch(value.strip()) is None:
        return None
    seconds = float(value)
    # A very long numeric hint must never cause an earlier fallback retry.
    # A saturated hint is greater than all allowed per-run wait budgets.
    return min(seconds, 86_400.0)


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: Mapping[str, object] | None = None
    retry_after_seconds: float | None = None


def _decode_success_body(raw: bytes) -> HttpResult:
    """Map an HTTP 200 body to a bounded, strictly parsed response.

    Invalid and oversized completions are NOT transient upstream HTTP errors;
    the response mapper returns a non-retryable GROQ_INVALID_RESPONSE.
    """
    if len(raw) > _MAX_RESPONSE_BYTES:
        return HttpResult(status_code=200)
    try:
        decoded = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        return HttpResult(status_code=200)
    return (
        HttpResult(status_code=200, body=decoded)
        if isinstance(decoded, dict)
        else HttpResult(status_code=200)
    )


class GroqTLSVerificationError(Exception):
    """Non-retryable certificate/hostname verification failure (redacted)."""


class GroqNetworkError(Exception):
    """Retryable TCP/connectivity/protocol failure (redacted)."""


class GroqTransport(Protocol):
    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult: ...


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        # Neither credential nor customer text may be forwarded to new hosts.
        return None


class UrllibGroqTransport:
    """Standard-library HTTPS adapter; injectable for fully offline unit tests.

    The endpoint is restricted to a fixed allowlist (Groq by default, NVIDIA
    Build for the compatibility provider); arbitrary URLs are rejected.
    """

    def __init__(self, *, endpoint: str = _ENDPOINT) -> None:
        self._endpoint = _checked_endpoint(endpoint)

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        return await asyncio.to_thread(
            self._post, body, api_key, timeout_seconds, endpoint=self._endpoint
        )

    @staticmethod
    def _post(
        body: Mapping[str, object],
        api_key: str,
        timeout_seconds: float,
        *,
        endpoint: str = _ENDPOINT,
    ) -> HttpResult:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            _checked_endpoint(endpoint),
            data=payload,
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            opener = build_opener(_RejectRedirects())
            with opener.open(request, timeout=timeout_seconds) as response:
                # A non-200 response must never be parsed as a successful completion.
                # Do not read untrusted bodies for other status codes.
                if response.status != 200:
                    hint = (
                        _retry_after_seconds(getattr(response, "headers", None))
                        if response.status == 429
                        else None
                    )
                    return HttpResult(status_code=response.status, retry_after_seconds=hint)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                return _decode_success_body(raw)
        except HTTPError as exc:
            # Do not load/log error bodies or redirects; close their file handles.
            try:
                hint = (
                    _retry_after_seconds(getattr(exc, "headers", None)) if exc.code == 429 else None
                )
                return HttpResult(status_code=exc.code, retry_after_seconds=hint)
            finally:
                exc.close()
