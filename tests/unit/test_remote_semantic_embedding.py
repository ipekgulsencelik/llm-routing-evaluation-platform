"""Semantic EmbeddingPort adapter: deny-by-default gates and strict parsing (no network)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from llm_routing_eval.adapters.embeddings import (
    EmbeddingUnavailableError,
    RemoteEmbeddingOptions,
    RemoteSemanticEmbedding,
    managed_nvidia_embedding,
)
from llm_routing_eval.core import BenchmarkCase, RoutingContext
from llm_routing_eval.routing import DevelopmentExample, KnnRouter, ModelPair

MODEL = "nvidia/nv-embedqa-e5-v5"
SECRET = "IN_MEMORY_FAKE_NVIDIA_CREDENTIAL"
TEXTS = ("synthetic account question", "synthetic loan question")

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class SyntheticOnlyEmbeddingApproval:
    def __init__(self) -> None:
        self.calls = 0

    def authorize_embedding(self, *, model: str, texts: tuple[str, ...]) -> bool:
        self.calls += 1
        return model == MODEL and all(text.startswith("synthetic") for text in texts)


@pytest.fixture
def live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ROUTING_ALLOW_LIVE", "1")
    monkeypatch.setenv("NVIDIA_API_KEY", SECRET)


def vectors_for(inputs: list[str]) -> list[list[float]]:
    # Deterministic synthetic semantics: "account" vs "loan" axes.
    return [
        [1.0, 0.1, 0.0] if "account" in text else [0.0, 0.2, 1.0] if "loan" in text else [0.5] * 3
        for text in inputs
    ]


def ok_handler(seen: list[httpx.Request]) -> Handler:
    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        inputs = json.loads(request.content)["input"]
        data = [
            {"index": i, "embedding": v} for i, v in reversed(list(enumerate(vectors_for(inputs))))
        ]
        return httpx.Response(200, json={"data": data, "model": MODEL})

    return handler


def embed(
    handler: Handler,
    texts: tuple[str, ...] = TEXTS,
    *,
    approval: object | None = None,
    **opts: object,
) -> tuple[tuple[float, ...], ...]:
    values: dict[str, object] = {"model": MODEL, "live_enabled": True}
    values.update(opts)

    async def run() -> tuple[tuple[float, ...], ...]:
        async with managed_nvidia_embedding(
            RemoteEmbeddingOptions(**values),  # type: ignore[arg-type]
            outbound_authorization=approval,  # type: ignore[arg-type]
            mock_transport=httpx.MockTransport(handler),
        ) as embedder:
            return await embedder.embed(texts)

    return asyncio.run(run())


def test_default_denies_without_network(live_env: None) -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_PRIVACY_NOT_AUTHORIZED"):
        embed(ok_handler(seen))
    assert seen == []


@pytest.mark.parametrize(
    ("env", "opts", "code"),
    [
        ({}, {}, "EMBEDDING_LIVE_DISABLED"),
        ({"LLM_ROUTING_ALLOW_LIVE": "1"}, {"live_enabled": False}, "EMBEDDING_LIVE_DISABLED"),
        ({"LLM_ROUTING_ALLOW_LIVE": "1"}, {}, "EMBEDDING_API_KEY_MISSING"),
        (
            {"LLM_ROUTING_ALLOW_LIVE": "1", "NVIDIA_API_KEY": SECRET},
            {"max_text_chars": 5},
            "EMBEDDING_TEXT_BUDGET_EXCEEDED",
        ),
    ],
)
def test_gates_precede_network(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], opts: dict[str, object], code: str
) -> None:
    monkeypatch.delenv("LLM_ROUTING_ALLOW_LIVE", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    seen: list[httpx.Request] = []
    with pytest.raises(EmbeddingUnavailableError, match=code):
        embed(ok_handler(seen), approval=SyntheticOnlyEmbeddingApproval(), **opts)
    assert seen == []


def test_non_synthetic_text_is_denied_by_authorizer(live_env: None) -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_PRIVACY_NOT_AUTHORIZED"):
        embed(ok_handler(seen), ("customer text",), approval=SyntheticOnlyEmbeddingApproval())
    assert seen == []


def test_success_orders_by_index_batches_and_sends_fixed_request(live_env: None) -> None:
    seen: list[httpx.Request] = []
    texts = ("synthetic account a", "synthetic loan b", "synthetic account c")
    vectors = embed(
        ok_handler(seen), texts, approval=SyntheticOnlyEmbeddingApproval(), max_batch_size=2
    )
    assert vectors == (
        (1.0, 0.1, 0.0),
        (0.0, 0.2, 1.0),
        (1.0, 0.1, 0.0),
    )
    assert len(seen) == 2
    first = seen[0]
    assert str(first.url) == "https://integrate.api.nvidia.com/v1/embeddings"
    assert first.headers["Authorization"] == "Bearer " + SECRET
    body = json.loads(first.content)
    assert body == {
        "model": MODEL,
        "input": ["synthetic account a", "synthetic loan b"],
        "encoding_format": "float",
        "input_type": "query",
    }


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"data": []}, "EMBEDDING_COUNT_MISMATCH"),
        ({"data": "x"}, "EMBEDDING_COUNT_MISMATCH"),
        ([1, 2], "EMBEDDING_INVALID_RESPONSE"),
        (
            {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]},
            "EMBEDDING_INDEX_INVALID",
        ),
        (
            {"data": [{"index": 0, "embedding": [1.0, 0]}, {"index": 1, "embedding": [1.0]}]},
            "EMBEDDING_DIMENSIONS_INVALID",
        ),
        (
            {"data": [{"index": 0, "embedding": [0.0]}, {"index": 1, "embedding": [1.0]}]},
            "EMBEDDING_ZERO_VECTOR",
        ),
        (
            {"data": [{"index": 0, "embedding": ["1"]}, {"index": 1, "embedding": [1.0]}]},
            "EMBEDDING_NON_FINITE",
        ),
    ],
)
def test_malformed_responses_fail_closed(live_env: None, payload: object, code: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(EmbeddingUnavailableError, match=code):
        embed(handler, approval=SyntheticOnlyEmbeddingApproval())


def test_expected_dimensions_are_enforced(live_env: None) -> None:
    with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_DIMENSIONS_INVALID"):
        embed(ok_handler([]), approval=SyntheticOnlyEmbeddingApproval(), expected_dimensions=4)


@pytest.mark.parametrize(
    ("status", "code"),
    [(429, "EMBEDDING_RATE_LIMIT"), (500, "EMBEDDING_HTTP_ERROR"), (307, "EMBEDDING_HTTP_ERROR")],
)
def test_http_errors_never_expose_body(live_env: None, status: int, code: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, text="PRIVATE REMOTE TEXT", headers={"Location": "https://x.invalid"}
        )

    with pytest.raises(EmbeddingUnavailableError) as info:
        embed(handler, approval=SyntheticOnlyEmbeddingApproval())
    assert str(info.value) == code
    assert info.value.__cause__ is None


def test_timeouts_and_transport_errors_are_redacted(live_env: None) -> None:
    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret detail", request=request)

    async def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret detail", request=request)

    with pytest.raises(EmbeddingUnavailableError, match="^EMBEDDING_TIMEOUT$"):
        embed(timeout, approval=SyntheticOnlyEmbeddingApproval())
    with pytest.raises(EmbeddingUnavailableError, match="^EMBEDDING_TRANSPORT_ERROR$"):
        embed(broken, approval=SyntheticOnlyEmbeddingApproval())


def test_invalid_json_and_oversize_bodies(live_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async def garbage(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_INVALID_RESPONSE"):
        embed(garbage, approval=SyntheticOnlyEmbeddingApproval())

    import llm_routing_eval.adapters.embeddings.remote as remote

    monkeypatch.setattr(remote, "_MAX_RESPONSE_BYTES", 10)
    with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_RESPONSE_TOO_LARGE"):
        embed(ok_handler([]), approval=SyntheticOnlyEmbeddingApproval())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "bad model"}, "EMBEDDING_MODEL_INVALID"),
        ({"api_key_env": "GROQ_API_KEY"}, "EMBEDDING_KEY_ENV_INVALID"),
        ({"max_batch_size": 0}, "EMBEDDING_BATCH_BUDGET_INVALID"),
        ({"max_text_chars": 0}, "EMBEDDING_TEXT_BUDGET_INVALID"),
        ({"timeout_seconds": 0}, "EMBEDDING_TIMEOUT_INVALID"),
        ({"expected_dimensions": 0}, "EMBEDDING_DIMENSIONS_INVALID"),
    ],
)
def test_options_validate(overrides: dict[str, object], message: str) -> None:
    values: dict[str, object] = {"model": MODEL}
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        RemoteEmbeddingOptions(**values)  # type: ignore[arg-type]


def test_endpoint_is_fixed_and_empty_input_rejected(live_env: None) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler([]))) as client:
            with pytest.raises(ValueError, match="EMBEDDING_ENDPOINT_NOT_ALLOWLISTED"):
                RemoteSemanticEmbedding(
                    RemoteEmbeddingOptions(model=MODEL), client, endpoint="https://x.invalid"
                )
            embedder = RemoteSemanticEmbedding(
                RemoteEmbeddingOptions(model=MODEL, live_enabled=True), client
            )
            with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_EMPTY_INPUT"):
                await embedder.embed(())

    asyncio.run(run())


def test_knn_router_uses_semantic_embedding(live_env: None) -> None:
    """The kNN router composes with the semantic adapter purely through EmbeddingPort."""

    async def run() -> tuple[str | None, str]:
        async with managed_nvidia_embedding(
            RemoteEmbeddingOptions(model=MODEL, live_enabled=True),
            outbound_authorization=SyntheticOnlyEmbeddingApproval(),
            mock_transport=httpx.MockTransport(ok_handler([])),
        ) as embedder:
            router = await KnnRouter.prepare(
                ModelPair(small="m-small", large="m-large"),
                embedder,
                (
                    DevelopmentExample("dev-1", "synthetic account opening", "small"),
                    DevelopmentExample("dev-2", "synthetic loan restructuring", "large"),
                ),
                k=1,
                minimum_similarity=0.5,
            )
            case = BenchmarkCase(
                case_id="c",
                input_text="synthetic account balance",
                domain="d",
                scenario_type="s",
            )
            decision = await router.decide(
                case, RoutingContext(current_stage=0, remaining_model_calls=2)
            )
            return decision.selected_model, decision.reason_code

    assert asyncio.run(run()) == ("m-small", "KNN_SMALL")
