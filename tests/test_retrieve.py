import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import QdrantClient

from src.main import app
from src.routers.retrieve import get_retrieval_service
from src.services.retrieval import LLMQueryRewriter, RetrievalService
from src.services.vector_store import VectorStoreService


class MockDenseEmbedder:
    def __init__(self, dim: int = 1536):
        self.dim = dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, ch in enumerate(text[: self.dim]):
                vec[i % self.dim] = float(ord(ch) % 100) / 100.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            results.append([x / norm for x in vec])
        return results

    async def embed_query(self, text: str) -> list[float]:
        docs = await self.embed_documents([text])
        return docs[0]


class MockSparseEmbedder:
    def embed(self, texts: list[str]):
        for text in texts:
            words = text.lower().split()
            indices = []
            values = []
            for w in sorted(set(words)):
                idx = sum(ord(c) for c in w) % 1000
                indices.append(idx)
                values.append(float(len(w)) / 10.0)
            yield type("SparseOutput", (), {"indices": indices, "values": values})()

    def query_embed(self, query: str):
        return self.embed([query])


class MockRewriter(LLMQueryRewriter):
    def __init__(self):
        super().__init__()
        self.rewrite_calls = []
        self.reformulate_calls = []
        self.reformulate_return_value: str | None = None

    async def rewrite(self, query_text: str, conversation_context: list[str]) -> str:
        self.rewrite_calls.append((query_text, conversation_context))
        if not conversation_context:
            return query_text
        # Deterministic mock rewrite resolving pronouns
        return f"accreditation data retention policy for {query_text}"

    async def reformulate(self, query_text: str, reason: str | None = None) -> str:
        self.reformulate_calls.append((query_text, reason))
        if self.reformulate_return_value is not None:
            return self.reformulate_return_value
        return f"{query_text} data retention mandatory"


@pytest.fixture
def mock_vector_store():
    client = QdrantClient(location=":memory:")
    service = VectorStoreService(
        client=client,
        dense_embedder=MockDenseEmbedder(),
        sparse_embedder=MockSparseEmbedder(),
        collection_name="test_retrieve_chunks",
    )
    return service


@pytest.fixture
def mock_rewriter():
    return MockRewriter()


@pytest.fixture
def retrieval_service(mock_vector_store, mock_rewriter):
    return RetrievalService(vector_store=mock_vector_store, rewriter=mock_rewriter)


@pytest.fixture
def client(retrieval_service):
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test")
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_rewrite_endpoint_empty_context(client: AsyncClient):
    response = await client.post(
        "/retrieve/rewrite",
        json={"query_text": "What are the rules?", "conversation_context": []},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["rewritten_query"] == "What are the rules?"


@pytest.mark.asyncio
async def test_rewrite_endpoint_with_context(client: AsyncClient, mock_rewriter: MockRewriter):
    response = await client.post(
        "/retrieve/rewrite",
        json={
            "query_text": "Does it require 7 years?",
            "conversation_context": [
                "User: What is the accreditation policy?",
                "Assistant: Standard 1 governs data retention.",
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "accreditation" in data["rewritten_query"]
    assert len(mock_rewriter.rewrite_calls) == 1


@pytest.mark.asyncio
async def test_retrieve_high_confidence_no_requery(
    client: AsyncClient, mock_vector_store: VectorStoreService, mock_rewriter: MockRewriter
):
    # Seed chunks
    await mock_vector_store.upsert_chunks(
        namespace="tenant_test",
        chunks=[
            {
                "chunk_text": "Accreditation standard 1: Data retention is mandatory for 7 years.",
                "metadata": {"source": "accreditation_policy.pdf", "page_num": 1},
            },
            {
                "chunk_text": "Accreditation standard 2: All audit logs must be encrypted at rest.",
                "metadata": {"source": "accreditation_policy.pdf", "page_num": 2},
            },
        ],
    )

    payload = {
        "namespace": "tenant_test",
        "query_text": "Accreditation standard 1 data retention",
        "conversation_context": [],
        "top_k": 2,
        "confidence_threshold": 0.50,
        "max_requery_attempts": 1,
    }

    response = await client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["confidence"] == "high"
    assert data["requery_performed"] is False
    assert data["requery_reason"] is None
    assert data["rewritten_query"] == "Accreditation standard 1 data retention"
    assert len(data["results"]) > 0

    first_result = data["results"][0]
    assert "Accreditation standard" in first_result["chunk_text"]
    assert first_result["source"]["filename"] == "accreditation_policy.pdf"
    assert first_result["source"]["page_num"] in (1, 2)
    assert "quoted_span" in first_result["source"]
    assert len(first_result["source"]["quoted_span"]) > 0
    assert "explanation" in first_result
    assert "Matches terms" in first_result["explanation"]
    assert len(mock_rewriter.reformulate_calls) == 0


@pytest.mark.asyncio
async def test_retrieve_with_conversational_context(
    client: AsyncClient, mock_vector_store: VectorStoreService, mock_rewriter: MockRewriter
):
    await mock_vector_store.upsert_chunks(
        namespace="tenant_conv",
        chunks=[
            {
                "chunk_text": "Accreditation standard 1: Data retention is mandatory for 7 years.",
                "metadata": {"source": "accreditation_policy.pdf", "page_num": 1},
            }
        ],
    )

    payload = {
        "namespace": "tenant_conv",
        "query_text": "retention period",
        "conversation_context": [
            "What does accreditation say?",
            "It outlines data compliance.",
        ],
        "top_k": 2,
    }

    response = await client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(mock_rewriter.rewrite_calls) == 1
    assert "accreditation" in data["rewritten_query"]


@pytest.mark.asyncio
async def test_retrieve_low_confidence_triggers_requery(
    client: AsyncClient, mock_vector_store: VectorStoreService, mock_rewriter: MockRewriter
):
    # Upsert specific content
    await mock_vector_store.upsert_chunks(
        namespace="tenant_requery",
        chunks=[
            {
                "chunk_text": "Data retention mandatory rules require 7 years archival.",
                "metadata": {"source": "retention_spec.pdf", "page_num": 4},
            }
        ],
    )

    # Mock reformulate to produce terms that match the indexed chunk
    mock_rewriter.reformulate_return_value = "Data retention mandatory archival"

    # Query with ambiguous / low-overlap query
    payload = {
        "namespace": "tenant_requery",
        "query_text": "unrelated storage duration question",
        "conversation_context": [],
        "top_k": 2,
        "confidence_threshold": 0.55,
        "max_requery_attempts": 1,
    }

    response = await client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["requery_performed"] is True
    assert data["requery_reason"] is not None
    assert "Initial confidence" in data["requery_reason"]
    assert "below threshold" in data["requery_reason"]
    assert len(mock_rewriter.reformulate_calls) == 1
    assert len(data["results"]) > 0


@pytest.mark.asyncio
async def test_retrieve_bounded_attempts_low_confidence_fallback(
    client: AsyncClient, mock_vector_store: VectorStoreService, mock_rewriter: MockRewriter
):
    # Even after reformulation, no matching content exists for this tenant
    mock_rewriter.reformulate_return_value = "Still no matching content anywhere"

    payload = {
        "namespace": "tenant_empty",
        "query_text": "Non-existent topics",
        "conversation_context": [],
        "top_k": 2,
        "confidence_threshold": 0.70,
        "max_requery_attempts": 2,
    }

    response = await client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Never crash or 500, return confidence: "low"
    assert data["confidence"] == "low"
    assert data["requery_performed"] is True
    assert len(mock_rewriter.reformulate_calls) == 2


@pytest.mark.asyncio
async def test_citation_and_explanation_structure(
    client: AsyncClient, mock_vector_store: VectorStoreService
):
    long_text = (
        "Article 42. Compliance protocols for external cloud systems.\n"
        "All data stores must configure encryption and multi-factor authentication."
    )
    await mock_vector_store.upsert_chunks(
        namespace="tenant_citation",
        chunks=[
            {
                "chunk_text": long_text,
                "metadata": {"source": "security/protocols_v2.pdf", "page_num": 12},
            }
        ],
    )

    payload = {
        "namespace": "tenant_citation",
        "query_text": "compliance protocols cloud systems",
        "top_k": 1,
    }

    response = await client.post("/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1

    res = data["results"][0]
    assert res["source"]["filename"] == "protocols_v2.pdf"
    assert res["source"]["page_num"] == 12
    expected_span = "Article 42. Compliance protocols for external cloud systems."
    assert res["source"]["quoted_span"] == expected_span
    assert "compliance" in res["explanation"].lower() or "protocols" in res["explanation"].lower()
