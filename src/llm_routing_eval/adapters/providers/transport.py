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
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    return HttpResult(status_code=502)
                decoded = json.loads(raw)
                if not isinstance(decoded, dict):
                    return HttpResult(status_code=502)
                return HttpResult(status_code=response.status, body=decoded)
        except HTTPError as exc:
            # Never load or log provider error response bodies, even 4xx/5xx.
            return HttpResult(status_code=exc.code)
