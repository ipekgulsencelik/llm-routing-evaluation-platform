"""Outbound privacy authorization boundary for live provider requests.

Only an independently reviewed application/privacy adapter can authorize a
specific outbound request. The default is deny-all, including when keys exist.
"""

from __future__ import annotations

from typing import Protocol

from llm_routing_eval.core import ModelRequest


class OutboundAuthorization(Protocol):
    """A narrowly scoped capability check; no persistence or logging."""

    def authorize(self, request: ModelRequest) -> bool: ...


class DenyAllOutbound:
    """Safe default while production-grade privacy approval is not integrated."""

    def authorize(self, request: ModelRequest) -> bool:
        _ = request
        return False
