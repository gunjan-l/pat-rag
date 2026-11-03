from enum import StrEnum
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from src.services.vector_store import VectorStoreService

router = APIRouter()

_vector_store: VectorStoreService | None = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


class ChunkItem(BaseModel):
    chunk_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertChunksRequest(BaseModel):
    namespace: str
    chunks: list[ChunkItem]


class UpsertResponse(BaseModel):
    namespace: str
    upserted_count: int
    failed: list[Any] = Field(default_factory=list)


class QueryRequest(BaseModel):
    namespace: str
    query_text: str
    top_k: int = 4
    metadata_filter: dict[str, Any] | None = None


class MatchItem(BaseModel):
    score: float
    chunk_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    matches: list[MatchItem]


class DeleteScope(StrEnum):
    namespace = "namespace"
    document = "document"


class DeleteChunksRequest(BaseModel):
    namespace: str
    scope: DeleteScope
    source: str | None = None

    @model_validator(mode="after")
    def validate_document_source(self) -> "DeleteChunksRequest":
        if self.scope == DeleteScope.document and not self.source:
            raise ValueError("source is required when scope is 'document'")
        return self


class DeleteResponse(BaseModel):
    namespace: str
    deleted_count: int


class ChunkAndUpsertRequest(BaseModel):
    namespace: str
    document_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 500
    chunk_overlap: int = 50


@router.post("/chunks", response_model=UpsertResponse)
async def upsert_chunks(
    body: UpsertChunksRequest,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> UpsertResponse:
    chunks_data = [chunk.model_dump() for chunk in body.chunks]
    result = await vector_store.upsert_chunks(
        namespace=body.namespace,
        chunks=chunks_data,
    )
    return UpsertResponse(**result)


@router.post("/chunks/query", response_model=QueryResponse)
async def query_chunks(
    body: QueryRequest,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> QueryResponse:
    result = await vector_store.query_chunks(
        namespace=body.namespace,
        query_text=body.query_text,
        top_k=body.top_k,
        metadata_filter=body.metadata_filter,
    )
    return QueryResponse(**result)


@router.post("/chunks/delete", response_model=DeleteResponse)
async def delete_chunks(
    body: DeleteChunksRequest,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DeleteResponse:
    try:
        result = await vector_store.delete_chunks(
            namespace=body.namespace,
            scope=body.scope.value,
            source=body.source,
        )
        return DeleteResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.post("/documents/chunk-and-upsert", response_model=UpsertResponse)
async def chunk_and_upsert_document(
    body: ChunkAndUpsertRequest,
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> UpsertResponse:
    result = await vector_store.chunk_and_upsert_document(
        namespace=body.namespace,
        document_text=body.document_text,
        metadata=body.metadata,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    return UpsertResponse(**result)
