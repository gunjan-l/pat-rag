import uuid
import warnings
from typing import Any

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from src.config import settings


def clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively remove keys with None values so non-applicable metadata is omitted."""
    if not metadata:
        return {}
    cleaned = {}
    for k, v in metadata.items():
        if v is not None:
            if isinstance(v, dict):
                cleaned[k] = clean_metadata(v)
            else:
                cleaned[k] = v
    return cleaned


class OpenAIDenseEmbedder:
    """Generates dense vector embeddings using OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key or ''}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json={"input": texts, "model": self.model},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data]

    async def embed_query(self, text: str) -> list[float]:
        docs = await self.embed_documents([text])
        return docs[0]


class FastEmbedSparseEmbedder:
    """Generates sparse BM25 vectors using FastEmbed."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: list[str]):
        return self.model.embed(texts)

    def query_embed(self, query: str):
        return self.model.query_embed(query)


class VectorStoreService:
    """Hybrid vector storage and retrieval service backed by Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        dense_embedder: Any | None = None,
        sparse_embedder: Any | None = None,
        collection_name: str = "chunks",
    ):
        self.collection_name = collection_name
        if client is not None:
            self.client = client
        elif settings.qdrant_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

        self.dense_embedder = dense_embedder or OpenAIDenseEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
        self.sparse_embedder = sparse_embedder or FastEmbedSparseEmbedder()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        dim = getattr(self.dense_embedder, "dim", 1536)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                },
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="namespace",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="metadata.source",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    async def upsert_chunks(
        self,
        namespace: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not chunks:
            return {"namespace": namespace, "upserted_count": 0, "failed": []}

        texts = [chunk["chunk_text"] for chunk in chunks]
        dense_vectors = await self.dense_embedder.embed_documents(texts)
        sparse_vectors = list(self.sparse_embedder.embed(texts))

        points = []
        for i, chunk in enumerate(chunks):
            sparse_res = sparse_vectors[i]
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vectors[i],
                        "sparse": models.SparseVector(
                            indices=list(sparse_res.indices),
                            values=[float(v) for v in sparse_res.values],
                        ),
                    },
                    payload={
                        "namespace": namespace,
                        "chunk_text": chunk["chunk_text"],
                        "metadata": clean_metadata(chunk.get("metadata", {})),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return {
            "namespace": namespace,
            "upserted_count": len(points),
            "failed": [],
        }

    async def query_chunks(
        self,
        namespace: str,
        query_text: str,
        top_k: int = 4,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection_name):
            return {"matches": []}

        must_conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="namespace",
                match=models.MatchValue(value=namespace),
            )
        ]

        if metadata_filter:
            for k, val in metadata_filter.items():
                field_key = f"metadata.{k}" if not k.startswith("metadata.") else k
                if isinstance(val, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchAny(any=val),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchValue(value=val),
                        )
                    )

        qdrant_filter = models.Filter(must=must_conditions)

        dense_query = await self.dense_embedder.embed_query(query_text)
        sparse_query_gen = self.sparse_embedder.query_embed(query_text)
        sparse_list = list(sparse_query_gen)
        if sparse_list:
            sparse_item = sparse_list[0]
            sparse_query = models.SparseVector(
                indices=list(sparse_item.indices),
                values=[float(v) for v in sparse_item.values],
            )
        else:
            sparse_query = models.SparseVector(indices=[], values=[])

        prefetch = [
            models.Prefetch(
                query=dense_query,
                using="dense",
                limit=top_k * 2,
                filter=qdrant_filter,
            ),
            models.Prefetch(
                query=sparse_query,
                using="sparse",
                limit=top_k * 2,
                filter=qdrant_filter,
            ),
        ]

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        matches = []
        for pt in results.points:
            matches.append(
                {
                    "score": round(float(pt.score), 4),
                    "chunk_text": pt.payload.get("chunk_text", "") if pt.payload else "",
                    "metadata": pt.payload.get("metadata", {}) if pt.payload else {},
                }
            )

        return {"matches": matches}

    async def delete_chunks(
        self,
        namespace: str,
        scope: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        if scope not in ("namespace", "document"):
            raise ValueError(f"Invalid scope '{scope}'. Must be 'namespace' or 'document'.")
        if scope == "document" and not source:
            raise ValueError("source is required when scope is 'document'")

        if not self.client.collection_exists(self.collection_name):
            return {"namespace": namespace, "deleted_count": 0}

        must_conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="namespace",
                match=models.MatchValue(value=namespace),
            )
        ]
        if scope == "document":
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source),
                )
            )

        delete_filter = models.Filter(must=must_conditions)

        count_res = self.client.count(
            collection_name=self.collection_name,
            count_filter=delete_filter,
            exact=True,
        )
        deleted_count = count_res.count

        if deleted_count > 0:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=delete_filter),
            )

        return {
            "namespace": namespace,
            "deleted_count": deleted_count,
        }

    async def chunk_and_upsert_document(
        self,
        namespace: str,
        document_text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> dict[str, Any]:
        clean_meta = clean_metadata(metadata)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_texts = splitter.split_text(document_text)
        if not split_texts:
            return {"namespace": namespace, "upserted_count": 0, "failed": []}

        chunks = []
        for i, txt in enumerate(split_texts):
            chunk_meta = dict(clean_meta)
            chunk_meta["chunk"] = i
            chunk_meta["text"] = txt
            chunks.append({"chunk_text": txt, "metadata": chunk_meta})

        return await self.upsert_chunks(namespace=namespace, chunks=chunks)
