"""Replaceable Groq HTTP boundary with bounded bodies and no credential redirects."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: Mapping[str, object] | None = None


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
    """Standard-library HTTPS adapter; injectable for fully offline unit tests."""

    async def post_chat(
        self, body: Mapping[str, object], *, api_key: str, timeout_seconds: float
    ) -> HttpResult:
        return await asyncio.to_thread(self._post, body, api_key, timeout_seconds)

    @staticmethod
    def _post(body: Mapping[str, object], api_key: str, timeout_seconds: float) -> HttpResult:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            _ENDPOINT,
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
                    return HttpResult(status_code=response.status)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    # Treat an oversized 200 response as a malformed completion,
                    # not as an HTTP 502 (which the orchestrator would retry).
                    return HttpResult(status_code=200)
                try:
                    decoded = json.loads(raw)
                except (ValueError, UnicodeError, RecursionError):
                    # Malformed provider JSON is not a transient transport failure.
                    # The mapper turns HTTP 200 + no body into a non-retryable code.
                    return HttpResult(status_code=200)
                if not isinstance(decoded, dict):
                    return HttpResult(status_code=200)
                return HttpResult(status_code=200, body=decoded)
        except HTTPError as exc:
            # Do not load/log error bodies or redirects; close their file handles.
            try:
                return HttpResult(status_code=exc.code)
            finally:
                exc.close()
