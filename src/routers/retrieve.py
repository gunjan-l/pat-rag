from fastapi import APIRouter, Depends

from src.services.retrieval import (
    RetrievalService,
    RetrieveRequest,
    RetrieveResponse,
    RewriteRequest,
    RewriteResponse,
)

router = APIRouter()

_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


@router.post("/retrieve/rewrite", response_model=RewriteResponse)
async def rewrite_query(
    body: RewriteRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RewriteResponse:
    rewritten = await service.rewrite_query(
        query_text=body.query_text,
        conversation_context=body.conversation_context,
    )
    return RewriteResponse(rewritten_query=rewritten)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    body: RetrieveRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> RetrieveResponse:
    return await service.retrieve(
        namespace=body.namespace,
        query_text=body.query_text,
        conversation_context=body.conversation_context,
        top_k=body.top_k,
        confidence_threshold=body.confidence_threshold,
        max_requery_attempts=body.max_requery_attempts,
    )
