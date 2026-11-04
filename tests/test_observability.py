from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.services.observability import (
    NoOpCallbackHandler,
    NoOpGeneration,
    NoOpSpan,
    NoOpTrace,
    ObservabilityMiddleware,
    ObservabilityService,
)
from src.services.retrieval import LLMQueryRewriter, RetrievalService

# ============================================================================
# 1. Graceful Degradation / No-Op Tracing Tests
# ============================================================================


def test_noop_degradation_when_credentials_not_set():
    service = ObservabilityService(public_key=None, secret_key=None)
    assert service.is_enabled is False
    assert service.get_client() is None

    # Test trace, span, and generation creation
    trace = service.create_trace(name="noop_trace", user_id="user1", session_id="sess1")
    assert isinstance(trace, NoOpTrace)

    span = trace.span(name="noop_span", input={"key": "val"})
    assert isinstance(span, NoOpSpan)
    assert span.end(output={"res": "ok"}) is span
    assert span.update(status_message="ok") is span

    generation = trace.generation(name="noop_gen", model="gpt-4o-mini")
    assert isinstance(generation, NoOpGeneration)
    assert generation.end(output="text", usage={"total_tokens": 10}) is generation
    assert generation.update(output="text") is generation

    # Test context manager support
    with trace.span("context_span") as s:
        assert isinstance(s, NoOpSpan)

    with trace.generation("context_gen") as g:
        assert isinstance(g, NoOpGeneration)

    # Test trace update & end
    assert trace.update(status_message="done") is trace
    assert trace.end() is trace

    # Test callback handler
    callback = service.get_langchain_callback(trace_name="test")
    assert isinstance(callback, NoOpCallbackHandler)
    # Callback methods should run without errors
    callback.on_chain_start({}, {"input": "hello"})
    callback.on_chain_end({"output": "world"})
    callback.on_llm_start({}, ["prompt"])
    callback.on_llm_end(MagicMock())
    callback.on_retriever_start({}, "query")
    callback.on_retriever_end([])

    # Flush should not error
    service.flush()


def test_noop_degradation_with_empty_strings():
    service = ObservabilityService(public_key="", secret_key="", host="")
    assert service.is_enabled is False
    assert service.get_client() is None

    trace = service.create_trace(name="empty_trace")
    assert isinstance(trace, NoOpTrace)
    assert isinstance(service.get_langchain_callback(), NoOpCallbackHandler)


# ============================================================================
# 2. FastAPI Observability Middleware Tests
# ============================================================================


class MockLangfuseSpan:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.kwargs["name"] = name
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self

    def end(self, **kwargs):
        self.ended = True
        if kwargs:
            self.updates.append(kwargs)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class MockLangfuseTrace(MockLangfuseSpan):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.spans = []
        self.generations = []

    def span(self, name: str, **kwargs):
        s = MockLangfuseSpan(name, **kwargs)
        self.spans.append(s)
        return s

    def generation(self, name: str, **kwargs):
        g = MockLangfuseSpan(name, **kwargs)
        self.generations.append(g)
        return g


class MockLangfuseClient:
    def __init__(self):
        self.traces = []
        self.flushed = False

    def trace(self, **kwargs):
        t = MockLangfuseTrace(**kwargs)
        self.traces.append(t)
        return t

    def flush(self):
        self.flushed = True


@pytest.fixture
def mock_langfuse_client():
    return MockLangfuseClient()


@pytest.fixture
def mock_observability_service(mock_langfuse_client):
    service = ObservabilityService(
        public_key="pk_test",
        secret_key="sk_test",
        client=mock_langfuse_client,
    )
    return service


def create_test_app(obs_service: ObservabilityService) -> FastAPI:
    test_app = FastAPI()
    test_app.add_middleware(ObservabilityMiddleware, observability_service=obs_service)

    @test_app.get("/health")
    async def health_endpoint():
        return {"status": "ok"}

    @test_app.get("/items/{item_id}")
    async def item_endpoint(item_id: str):
        return {"item_id": item_id}

    @test_app.get("/error")
    async def error_endpoint():
        raise RuntimeError("Test server exception")

    return test_app


@pytest.mark.asyncio
async def test_middleware_disabled_mode_passthrough():
    disabled_service = ObservabilityService(public_key=None, secret_key=None)
    app = create_test_app(disabled_service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_middleware_traces_headers(
    mock_observability_service: ObservabilityService,
    mock_langfuse_client: MockLangfuseClient,
):
    app = create_test_app(mock_observability_service)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/health",
            headers={
                "X-User-Id": "user_42",
                "X-Session-Id": "sess_99",
                "X-Tenant-Id": "tenant_omega",
            },
        )
        assert response.status_code == 200

    assert len(mock_langfuse_client.traces) == 1
    t = mock_langfuse_client.traces[0]
    assert t.kwargs["name"] == "GET /health"
    assert t.kwargs["user_id"] == "user_42"
    assert t.kwargs["session_id"] == "sess_99"
    assert t.ended is True

    # Check metadata and tags updated
    assert any("latency_ms" in u.get("metadata", {}) for u in t.updates)
    last_update = t.updates[-1]
    assert last_update["metadata"]["status_code"] == 200
    assert last_update["metadata"]["tenant_id"] == "tenant_omega"
    assert "200" in last_update["tags"] or any("200" in tag for tag in last_update["tags"])


@pytest.mark.asyncio
async def test_middleware_traces_query_params(
    mock_observability_service: ObservabilityService,
    mock_langfuse_client: MockLangfuseClient,
):
    app = create_test_app(mock_observability_service)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/items/42?user_id=query_user&session_id=query_sess&namespace=query_ns"
        )
        assert response.status_code == 200

    assert len(mock_langfuse_client.traces) == 1
    t = mock_langfuse_client.traces[0]
    assert t.kwargs["name"] == "GET /items/42"
    assert t.kwargs["user_id"] == "query_user"
    assert t.kwargs["session_id"] == "query_sess"
    last_update = t.updates[-1]
    assert last_update["metadata"]["tenant_id"] == "query_ns"


@pytest.mark.asyncio
async def test_middleware_traces_error_status(
    mock_observability_service: ObservabilityService,
    mock_langfuse_client: MockLangfuseClient,
):
    app = create_test_app(mock_observability_service)
    transport = ASGITransport(app=app, raise_app_exceptions=False)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 404 Not Found
        res_404 = await client.get("/nonexistent")
        assert res_404.status_code == 404

        # 500 Server Error
        res_500 = await client.get("/error")
        assert res_500.status_code == 500

    assert len(mock_langfuse_client.traces) == 2

    # Verify 404 trace
    t_404 = mock_langfuse_client.traces[0]
    last_404 = t_404.updates[-1]
    assert last_404["metadata"]["status_code"] == 404

    # Verify 500 trace
    t_500 = mock_langfuse_client.traces[1]
    last_500 = t_500.updates[-1]
    assert last_500["metadata"]["status_code"] == 500
    assert any(u.get("level") == "ERROR" for u in t_500.updates)


# ============================================================================
# 3. Retrieval Tracing & LangChain Callback Propagation Tests
# ============================================================================


class MockCallbackHandler(NoOpCallbackHandler):
    def __init__(self):
        super().__init__()
        self.chain_starts = []
        self.chain_ends = []
        self.retriever_starts = []
        self.retriever_ends = []
        self.llm_starts = []
        self.llm_ends = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.chain_starts.append((serialized, inputs))

    def on_chain_end(self, outputs, **kwargs):
        self.chain_ends.append(outputs)

    def on_retriever_start(self, serialized, query, **kwargs):
        self.retriever_starts.append((serialized, query))

    def on_retriever_end(self, documents, **kwargs):
        self.retriever_ends.append(documents)

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_starts.append((serialized, prompts))

    def on_llm_end(self, response, **kwargs):
        self.llm_ends.append(response)


class FakeVectorStore:
    def __init__(self, matches=None):
        self.matches = matches or []

    async def query_chunks(self, namespace: str, query_text: str, top_k: int = 4):
        return {"matches": self.matches}


class FakeRewriter(LLMQueryRewriter):
    def __init__(self):
        super().__init__()
        self.reformulate_count = 0

    async def rewrite(self, query_text: str, conversation_context: list[str], **kwargs) -> str:
        return f"rewritten: {query_text}"

    async def reformulate(self, query_text: str, reason: str | None = None, **kwargs) -> str:
        self.reformulate_count += 1
        return f"reformulated_{self.reformulate_count}: {query_text}"


@pytest.mark.asyncio
async def test_retrieval_tracing_spans_and_metadata(
    mock_observability_service: ObservabilityService,
):
    matches = [
        {
            "chunk_text": "Sample policy documentation text.",
            "score": 0.85,
            "metadata": {"source": "policy.pdf", "page_num": 1},
        }
    ]
    service = RetrievalService(
        vector_store=FakeVectorStore(matches=matches),
        rewriter=FakeRewriter(),
        observability=mock_observability_service,
    )

    trace = mock_observability_service.create_trace(name="test_retrieval")
    response = await service.retrieve(
        namespace="tenant_test",
        query_text="policy query",
        conversation_context=["Context line 1"],
        top_k=2,
        parent_trace=trace,
    )

    assert response.confidence == "high"
    assert len(trace.spans) >= 1
    span_names = [s.name for s in trace.spans]
    assert "vector_store_query" in span_names
    assert any(s.ended for s in trace.spans)


@pytest.mark.asyncio
async def test_retrieval_low_confidence_requery_spans(
    mock_observability_service: ObservabilityService,
):
    # Empty matches -> low confidence triggers re-query loop
    service = RetrievalService(
        vector_store=FakeVectorStore(matches=[]),
        rewriter=FakeRewriter(),
        observability=mock_observability_service,
    )

    trace = mock_observability_service.create_trace(name="requery_trace")
    response = await service.retrieve(
        namespace="tenant_test",
        query_text="unmatched query",
        top_k=2,
        confidence_threshold=0.6,
        max_requery_attempts=2,
        parent_trace=trace,
    )

    assert response.confidence == "low"
    assert response.requery_performed is True
    span_names = [s.name for s in trace.spans]
    # Check that requery attempt spans were recorded
    assert any("requery_attempt" in name for name in span_names)


@pytest.mark.asyncio
async def test_retrieval_callback_propagation():
    handler = MockCallbackHandler()
    matches = [
        {
            "chunk_text": "Some text for callback test",
            "score": 0.9,
            "metadata": {"source": "doc.pdf", "page_num": 2},
        }
    ]
    service = RetrievalService(
        vector_store=FakeVectorStore(matches=matches),
        rewriter=FakeRewriter(),
    )

    response = await service.retrieve(
        namespace="tenant_cb",
        query_text="callback query",
        conversation_context=["context msg"],
        callbacks=[handler],
    )

    assert response.confidence == "high"
    # Verify callbacks were called
    assert len(handler.chain_starts) >= 1
    assert len(handler.chain_ends) >= 1
    assert len(handler.retriever_starts) >= 1
    assert len(handler.retriever_ends) >= 1
