"""Embedding adapters behind Core ``EmbeddingPort``; routers never import this package."""

from .remote import (
    NVIDIA_BUILD_EMBEDDINGS_ENDPOINT,
    DenyAllEmbeddingOutbound,
    EmbeddingOutboundAuthorization,
    EmbeddingUnavailableError,
    RemoteEmbeddingOptions,
    RemoteSemanticEmbedding,
    managed_nvidia_embedding,
)

__all__ = [
    "NVIDIA_BUILD_EMBEDDINGS_ENDPOINT",
    "DenyAllEmbeddingOutbound",
    "EmbeddingOutboundAuthorization",
    "EmbeddingUnavailableError",
    "RemoteEmbeddingOptions",
    "RemoteSemanticEmbedding",
    "managed_nvidia_embedding",
]
