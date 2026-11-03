import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import QdrantClient

from src.main import app
from src.routers.chunks import get_vector_store
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


@pytest.fixture
def mock_vector_store():
    client = QdrantClient(location=":memory:")
    service = VectorStoreService(
        client=client,
        dense_embedder=MockDenseEmbedder(),
        sparse_embedder=MockSparseEmbedder(),
        collection_name="test_chunks",
    )
    return service


@pytest.fixture
def client(mock_vector_store):
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test")
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upsert_chunks_success(client: AsyncClient):
    payload = {
        "namespace": "tenant_1",
        "chunks": [
            {
                "chunk_text": "Accreditation standard 1: Data retention is mandatory for 7 years.",
                "metadata": {
                    "source": "accreditation_policy.pdf",
                    "chunk": 0,
                    "page_num": 1,
                    "empty_field": None,  # Should be omitted from storage
                },
            },
            {
                "chunk_text": "Accreditation standard 2: All logs must be encrypted at rest.",
                "metadata": {
                    "source": "accreditation_policy.pdf",
                    "chunk": 1,
                    "page_num": 2,
                },
            },
        ],
    }

    response = await client.post("/chunks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["namespace"] == "tenant_1"
    assert data["upserted_count"] == 2
    assert data["failed"] == []


@pytest.mark.asyncio
async def test_upsert_omits_null_metadata_fields(client: AsyncClient):
    payload = {
        "namespace": "tenant_clean_meta",
        "chunks": [
            {
                "chunk_text": "Sample text with null metadata values.",
                "metadata": {
                    "source": "test.pdf",
                    "valid_key": "valid_value",
                    "null_key": None,
                    "another_null": None,
                },
            }
        ],
    }

    response = await client.post("/chunks", json=payload)
    assert response.status_code == 200

    # Query back to verify null_key was omitted
    query_payload = {
        "namespace": "tenant_clean_meta",
        "query_text": "Sample text",
        "top_k": 1,
    }
    q_resp = await client.post("/chunks/query", json=query_payload)
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert len(q_data["matches"]) == 1
    match_meta = q_data["matches"][0]["metadata"]
    assert "null_key" not in match_meta
    assert "another_null" not in match_meta
    assert match_meta["valid_key"] == "valid_value"


@pytest.mark.asyncio
async def test_query_hybrid_and_namespace_isolation(client: AsyncClient):
    # Upsert to tenant_1
    await client.post(
        "/chunks",
        json={
            "namespace": "tenant_alpha",
            "chunks": [
                {
                    "chunk_text": "Alpha tenant retention requirements are 5 years.",
                    "metadata": {"source": "alpha_policy.pdf", "chunk": 0},
                }
            ],
        },
    )

    # Upsert to tenant_2
    await client.post(
        "/chunks",
        json={
            "namespace": "tenant_beta",
            "chunks": [
                {
                    "chunk_text": "Beta tenant retention requirements are 10 years.",
                    "metadata": {"source": "beta_policy.pdf", "chunk": 0},
                }
            ],
        },
    )

    # Query tenant_alpha
    resp_alpha = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_alpha",
            "query_text": "retention requirements",
            "top_k": 4,
        },
    )
    assert resp_alpha.status_code == 200
    data_alpha = resp_alpha.json()
    assert len(data_alpha["matches"]) == 1
    assert data_alpha["matches"][0]["metadata"]["source"] == "alpha_policy.pdf"
    assert "score" in data_alpha["matches"][0]
    assert (
        data_alpha["matches"][0]["chunk_text"]
        == "Alpha tenant retention requirements are 5 years."
    )

    # Query tenant_beta
    resp_beta = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_beta",
            "query_text": "retention requirements",
            "top_k": 4,
        },
    )
    assert resp_beta.status_code == 200
    data_beta = resp_beta.json()
    assert len(data_beta["matches"]) == 1
    assert data_beta["matches"][0]["metadata"]["source"] == "beta_policy.pdf"

    # Query nonexistent tenant
    resp_empty = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_gamma",
            "query_text": "retention requirements",
            "top_k": 4,
        },
    )
    assert resp_empty.status_code == 200
    assert resp_empty.json()["matches"] == []


@pytest.mark.asyncio
async def test_query_metadata_filter(client: AsyncClient):
    # Upsert multiple files to same tenant
    await client.post(
        "/chunks",
        json={
            "namespace": "tenant_filtered",
            "chunks": [
                {
                    "chunk_text": "Financial audit checklist section 1.",
                    "metadata": {"source": "finance.pdf", "chunk": 0},
                },
                {
                    "chunk_text": "Security audit checklist section 1.",
                    "metadata": {"source": "security.pdf", "chunk": 0},
                },
            ],
        },
    )

    # Filter by source: ["finance.pdf"]
    resp = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_filtered",
            "query_text": "audit checklist",
            "top_k": 10,
            "metadata_filter": {"source": ["finance.pdf"]},
        },
    )
    assert resp.status_code == 200
    matches = resp.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["metadata"]["source"] == "finance.pdf"


@pytest.mark.asyncio
async def test_delete_chunks_by_document_and_namespace(client: AsyncClient):
    # Index 2 chunks for doc1 and 1 chunk for doc2
    await client.post(
        "/chunks",
        json={
            "namespace": "tenant_del",
            "chunks": [
                {
                    "chunk_text": "Doc 1 Chunk 0",
                    "metadata": {"source": "doc1.pdf", "chunk": 0},
                },
                {
                    "chunk_text": "Doc 1 Chunk 1",
                    "metadata": {"source": "doc1.pdf", "chunk": 1},
                },
                {
                    "chunk_text": "Doc 2 Chunk 0",
                    "metadata": {"source": "doc2.pdf", "chunk": 0},
                },
            ],
        },
    )

    # Delete doc1 only
    resp = await client.post(
        "/chunks/delete",
        json={
            "namespace": "tenant_del",
            "scope": "document",
            "source": "doc1.pdf",
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"namespace": "tenant_del", "deleted_count": 2}

    # Verify doc1 is gone but doc2 remains
    q_resp = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_del",
            "query_text": "Chunk",
            "top_k": 10,
        },
    )
    matches = q_resp.json()["matches"]
    assert len(matches) == 1
    assert matches[0]["metadata"]["source"] == "doc2.pdf"

    # Delete entire namespace
    resp_ns = await client.post(
        "/chunks/delete",
        json={
            "namespace": "tenant_del",
            "scope": "namespace",
        },
    )
    assert resp_ns.status_code == 200
    assert resp_ns.json() == {"namespace": "tenant_del", "deleted_count": 1}

    # Verify namespace is empty
    q_empty = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_del",
            "query_text": "Chunk",
            "top_k": 10,
        },
    )
    assert q_empty.json()["matches"] == []


@pytest.mark.asyncio
async def test_delete_document_without_source_fails(client: AsyncClient):
    resp = await client.post(
        "/chunks/delete",
        json={
            "namespace": "tenant_del",
            "scope": "document",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chunk_and_upsert_document(client: AsyncClient):
    long_doc = (
        "Introduction to Policy Guidelines.\n\n"
        "Section 1: General provisions for AI governance and review.\n\n"
        "Section 2: Accreditation standards require periodic audits.\n\n"
        "Section 3: Archival of records must occur annually."
    )

    resp = await client.post(
        "/documents/chunk-and-upsert",
        json={
            "namespace": "tenant_chunk_upsert",
            "document_text": long_doc,
            "metadata": {"source": "guidelines.pdf", "category": "policy"},
            "chunk_size": 80,
            "chunk_overlap": 20,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["namespace"] == "tenant_chunk_upsert"
    assert data["upserted_count"] > 1
    assert data["failed"] == []

    # Query for section 2
    q_resp = await client.post(
        "/chunks/query",
        json={
            "namespace": "tenant_chunk_upsert",
            "query_text": "Accreditation standards",
            "top_k": 2,
        },
    )
    assert q_resp.status_code == 200
    matches = q_resp.json()["matches"]
    assert len(matches) > 0
    assert matches[0]["metadata"]["source"] == "guidelines.pdf"
    assert matches[0]["metadata"]["category"] == "policy"
    assert "chunk" in matches[0]["metadata"]
