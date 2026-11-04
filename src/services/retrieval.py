import inspect
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.services.observability import ObservabilityService, get_observability_service
from src.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def extract_filename(metadata: dict[str, Any]) -> str:
    raw = metadata.get("source") or metadata.get("filename") or "unknown"
    name = Path(str(raw)).name
    return name if name else str(raw)


def extract_quoted_span(chunk_text: str, max_chars: int = 150) -> str:
    text = chunk_text.strip()
    if not text:
        return ""
    if "\n" in text:
        first_line = text.split("\n", 1)[0].strip()
        if 15 <= len(first_line) <= max_chars:
            return first_line
    for delim in [". ", "? ", "! "]:
        start = 0
        while delim in text[start:]:
            idx = start + text[start:].index(delim)
            candidate = text[: idx + 1].strip()
            if len(candidate) >= 20 and len(candidate) <= max_chars:
                return candidate
            start = idx + len(delim)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 30:
        return truncated[:last_space].strip() + "..."
    return truncated.strip() + "..."


def extract_query_terms(text: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    meaningful = [w for w in words if w not in STOPWORDS]
    return meaningful if meaningful else words


def generate_explanation(query_terms: list[str], chunk_text: str) -> str:
    chunk_lower = chunk_text.lower()
    matched = [t for t in set(query_terms) if t in chunk_lower]
    if matched:
        sorted_matched = sorted(matched)
        terms_str = ", ".join(f"'{m}'" for m in sorted_matched[:4])
        return f"Matches terms {terms_str} with high semantic similarity"
    return "Retrieved via dense semantic similarity search"


def compute_composite_confidence(
    query_terms: list[str],
    matches: list[dict[str, Any]],
    score_weight: float = 0.6,
    coverage_weight: float = 0.4,
) -> float:
    if not matches:
        return 0.0

    top_score = float(matches[0].get("score", 0.0))
    normalized_score = max(0.0, min(1.0, top_score))

    if not query_terms:
        coverage = 1.0
    else:
        all_chunk_text = " ".join(m.get("chunk_text", "").lower() for m in matches)
        unique_terms = set(query_terms)
        covered_count = sum(1 for term in unique_terms if term in all_chunk_text)
        coverage = covered_count / len(unique_terms)

    # When query terms exist but none are present in retrieved chunks, retrieval quality is low
    if query_terms and coverage == 0.0:
        composite = 0.5 * normalized_score * score_weight
    else:
        composite = (score_weight * normalized_score) + (coverage_weight * coverage)

    return round(composite, 4)


class SourceCitation(BaseModel):
    filename: str
    page_num: int
    quoted_span: str


class RetrievalResultItem(BaseModel):
    chunk_text: str
    score: float
    source: SourceCitation
    explanation: str


class RewriteRequest(BaseModel):
    query_text: str
    conversation_context: list[str] = Field(default_factory=list)


class RewriteResponse(BaseModel):
    rewritten_query: str


class RetrieveRequest(BaseModel):
    namespace: str
    query_text: str
    conversation_context: list[str] = Field(default_factory=list)
    top_k: int = 4
    confidence_threshold: float = 0.55
    max_requery_attempts: int = 1


class RetrieveResponse(BaseModel):
    results: list[RetrievalResultItem]
    confidence: Literal["high", "low"]
    requery_performed: bool
    requery_reason: str | None = None
    rewritten_query: str


class LLMQueryRewriter:
    """Handles conversational query rewriting and fallback query reformulation."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        observability: ObservabilityService | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.completion_model
        self.observability = observability or get_observability_service()

    async def rewrite(
        self,
        query_text: str,
        conversation_context: list[str],
        parent_trace: Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        if not conversation_context:
            return query_text

        trace = parent_trace or self.observability.create_trace(
            name="query_rewrite",
            metadata={"query_text": query_text, "context_length": len(conversation_context)},
        )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_chain_start"):
                    cb.on_chain_start(
                        {"name": "query_rewrite"},
                        {"query_text": query_text, "conversation_context": conversation_context},
                    )

        if not self.api_key:
            result = f"{query_text} ({' '.join(conversation_context)})"
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_chain_end"):
                        cb.on_chain_end({"rewritten_query": result})
            return result

        headers = {"Authorization": f"Bearer {self.api_key}"}
        prompt = (
            "You are a search query reformulation assistant. Given the conversation history and a "
            "follow-up query, produce an explicit, self-contained search query that resolves all "
            "pronouns and conversational references. Return ONLY the rewritten query text, with no "
            "explanation, preamble, or quotes."
        )
        context_str = "\n".join(conversation_context)
        user_prompt = f"Conversation history:\n{context_str}\n\nFollow-up query: {query_text}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_prompt},
        ]

        gen = None
        if hasattr(trace, "generation"):
            gen = trace.generation(
                name="llm_query_rewrite",
                model=self.model,
                input=messages,
            )
        elif hasattr(trace, "span"):
            gen = trace.span(
                name="llm_query_rewrite",
                input=messages,
            )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_llm_start"):
                    cb.on_llm_start({"name": self.model}, [user_prompt])

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.0,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip().strip('"\'')
                result = content or query_text
                usage = data.get("usage", {})

                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                if gen:
                    if hasattr(gen, "end"):
                        gen.end(
                            output=result,
                            usage=usage,
                            metadata={"latency_ms": latency_ms},
                        )
                    elif hasattr(gen, "update"):
                        gen.update(output=result, usage=usage)

                if callbacks:
                    for cb in callbacks:
                        if hasattr(cb, "on_llm_end"):
                            cb.on_llm_end(data)
                        if hasattr(cb, "on_chain_end"):
                            cb.on_chain_end({"rewritten_query": result})

                return result
        except Exception as exc:
            logger.warning("LLM rewrite failed, falling back to deterministic rewrite: %s", exc)
            fallback = f"{query_text} ({' '.join(conversation_context)})"
            if gen and hasattr(gen, "end"):
                gen.end(output=fallback, status_message=str(exc), level="ERROR")
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_chain_end"):
                        cb.on_chain_end({"rewritten_query": fallback, "error": str(exc)})
            return fallback

    async def reformulate(
        self,
        query_text: str,
        reason: str | None = None,
        parent_trace: Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        trace = parent_trace or self.observability.create_trace(
            name="query_reformulate",
            metadata={"query_text": query_text, "reason": reason},
        )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_chain_start"):
                    cb.on_chain_start(
                        {"name": "query_reformulate"},
                        {"query_text": query_text, "reason": reason},
                    )

        if not self.api_key:
            fallback = f"{query_text} detailed specification"
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_chain_end"):
                        cb.on_chain_end({"reformulated_query": fallback})
            return fallback

        headers = {"Authorization": f"Bearer {self.api_key}"}
        prompt = (
            "You are an expert search query reformulator. The previous query retrieved "
            "low-confidence or incomplete results. Reformulate or expand the query to use clearer "
            "synonyms, technical terms, or standard terminology to improve retrieval quality. "
            "Return ONLY the reformulated search query, without explanation or quotes."
        )
        user_content = f"Query: {query_text}"
        if reason:
            user_content += f"\nContext/Reason: {reason}"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]

        gen = None
        if hasattr(trace, "generation"):
            gen = trace.generation(
                name="llm_query_reformulate",
                model=self.model,
                input=messages,
                metadata={"reason": reason},
            )
        elif hasattr(trace, "span"):
            gen = trace.span(
                name="llm_query_reformulate",
                input=messages,
                metadata={"reason": reason},
            )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_llm_start"):
                    cb.on_llm_start({"name": self.model}, [user_content])

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip().strip('"\'')
                result = content or query_text
                usage = data.get("usage", {})

                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                if gen:
                    if hasattr(gen, "end"):
                        gen.end(
                            output=result,
                            usage=usage,
                            metadata={"latency_ms": latency_ms},
                        )
                    elif hasattr(gen, "update"):
                        gen.update(output=result, usage=usage)

                if callbacks:
                    for cb in callbacks:
                        if hasattr(cb, "on_llm_end"):
                            cb.on_llm_end(data)
                        if hasattr(cb, "on_chain_end"):
                            cb.on_chain_end({"reformulated_query": result})

                return result
        except Exception as exc:
            logger.warning(
                "LLM reformulate failed, falling back to deterministic expansion: %s", exc
            )
            fallback = f"{query_text} detailed specification"
            if gen and hasattr(gen, "end"):
                gen.end(output=fallback, status_message=str(exc), level="ERROR")
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_chain_end"):
                        cb.on_chain_end({"reformulated_query": fallback, "error": str(exc)})
            return fallback


class RetrievalService:
    """Agentic retrieval service with conversational rewriting and confidence fallback."""

    def __init__(
        self,
        vector_store: Any | None = None,
        rewriter: LLMQueryRewriter | None = None,
        observability: ObservabilityService | None = None,
    ):
        self.vector_store = vector_store or vector_store_service
        self.rewriter = rewriter or LLMQueryRewriter()
        self.observability = observability or get_observability_service()

    async def _safe_rewrite(
        self,
        query_text: str,
        conversation_context: list[str],
        parent_trace: Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        sig = inspect.signature(self.rewriter.rewrite)
        kwargs: dict[str, Any] = {}
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "parent_trace" in sig.parameters or has_varkw:
            kwargs["parent_trace"] = parent_trace
        if "callbacks" in sig.parameters or has_varkw:
            kwargs["callbacks"] = callbacks
        return await self.rewriter.rewrite(query_text, conversation_context, **kwargs)

    async def _safe_reformulate(
        self,
        query_text: str,
        reason: str | None = None,
        parent_trace: Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        sig = inspect.signature(self.rewriter.reformulate)
        kwargs: dict[str, Any] = {}
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "reason" in sig.parameters or has_varkw:
            kwargs["reason"] = reason
        if "parent_trace" in sig.parameters or has_varkw:
            kwargs["parent_trace"] = parent_trace
        if "callbacks" in sig.parameters or has_varkw:
            kwargs["callbacks"] = callbacks
        return await self.rewriter.reformulate(query_text, **kwargs)

    async def rewrite_query(
        self,
        query_text: str,
        conversation_context: list[str],
        parent_trace: Any | None = None,
        callbacks: list[Any] | None = None,
    ) -> str:
        return await self._safe_rewrite(
            query_text, conversation_context, parent_trace=parent_trace, callbacks=callbacks
        )

    async def retrieve(
        self,
        namespace: str,
        query_text: str,
        conversation_context: list[str] | None = None,
        top_k: int = 4,
        confidence_threshold: float = 0.55,
        max_requery_attempts: int = 1,
        callbacks: list[Any] | None = None,
        parent_trace: Any | None = None,
    ) -> RetrieveResponse:
        context = conversation_context or []

        trace = parent_trace or self.observability.create_trace(
            name="retrieval",
            metadata={
                "namespace": namespace,
                "top_k": top_k,
                "confidence_threshold": confidence_threshold,
                "max_requery_attempts": max_requery_attempts,
            },
            tags=["retrieval", f"namespace:{namespace}"],
        )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_chain_start"):
                    cb.on_chain_start(
                        {"name": "retrieval"},
                        {"namespace": namespace, "query_text": query_text, "context": context},
                    )

        # 1. Automatically rewrite conversational query if context is non-empty
        if context:
            rewritten_query = await self._safe_rewrite(
                query_text, context, parent_trace=trace, callbacks=callbacks
            )
        else:
            rewritten_query = query_text

        current_query = rewritten_query

        # 2. Query vector store
        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_retriever_start"):
                    cb.on_retriever_start({"name": "vector_store"}, current_query)

        vector_span = None
        if hasattr(trace, "span"):
            vector_span = trace.span(
                name="vector_store_query",
                input={"namespace": namespace, "query": current_query, "top_k": top_k},
            )

        query_res = await self.vector_store.query_chunks(
            namespace=namespace,
            query_text=current_query,
            top_k=top_k,
        )
        matches = query_res.get("matches", [])
        query_terms = extract_query_terms(current_query)
        initial_confidence = compute_composite_confidence(query_terms, matches)

        if vector_span and hasattr(vector_span, "end"):
            vector_span.end(
                output={"match_count": len(matches), "confidence": initial_confidence}
            )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_retriever_end"):
                    cb.on_retriever_end(matches)

        best_matches = matches
        best_confidence = initial_confidence
        best_query = current_query

        # 3. Fallback requery check
        requery_performed = False
        requery_reason = None
        attempts = 0

        while best_confidence < confidence_threshold and attempts < max_requery_attempts:
            if not requery_performed:
                requery_performed = True
                requery_reason = (
                    f"Initial confidence {initial_confidence:.2f} "
                    f"below threshold {confidence_threshold:.2f}"
                )

            attempts += 1
            requery_span = None
            if hasattr(trace, "span"):
                requery_span = trace.span(
                    name=f"requery_attempt_{attempts}",
                    input={
                        "attempt": attempts,
                        "current_query": current_query,
                        "reason": requery_reason,
                    },
                )

            reformulated_query = await self._safe_reformulate(
                current_query,
                reason=requery_reason,
                parent_trace=trace,
                callbacks=callbacks,
            )
            current_query = reformulated_query

            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_retriever_start"):
                        cb.on_retriever_start({"name": "vector_store_requery"}, current_query)

            new_query_res = await self.vector_store.query_chunks(
                namespace=namespace,
                query_text=current_query,
                top_k=top_k,
            )
            new_matches = new_query_res.get("matches", [])
            new_terms = extract_query_terms(current_query)
            new_confidence = compute_composite_confidence(new_terms, new_matches)

            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_retriever_end"):
                        cb.on_retriever_end(new_matches)

            if requery_span and hasattr(requery_span, "end"):
                requery_span.end(
                    output={
                        "reformulated_query": current_query,
                        "match_count": len(new_matches),
                        "confidence": new_confidence,
                    }
                )

            # Keep candidate results that improved or when initial had no matches
            if new_confidence >= best_confidence or (not best_matches and new_matches):
                best_matches = new_matches
                best_confidence = new_confidence
                best_query = current_query

        confidence_label: Literal["high", "low"] = (
            "high" if best_confidence >= confidence_threshold else "low"
        )

        effective_terms = extract_query_terms(best_query)
        results: list[RetrievalResultItem] = []
        for match in best_matches:
            meta = match.get("metadata", {})
            chunk_text = match.get("chunk_text", "")
            results.append(
                RetrievalResultItem(
                    chunk_text=chunk_text,
                    score=float(match.get("score", 0.0)),
                    source=SourceCitation(
                        filename=extract_filename(meta),
                        page_num=int(meta.get("page_num", 1)),
                        quoted_span=extract_quoted_span(chunk_text),
                    ),
                    explanation=generate_explanation(effective_terms, chunk_text),
                )
            )

        response = RetrieveResponse(
            results=results,
            confidence=confidence_label,
            requery_performed=requery_performed,
            requery_reason=requery_reason,
            rewritten_query=rewritten_query,
        )

        if hasattr(trace, "update"):
            trace.update(
                output={
                    "confidence": confidence_label,
                    "results_count": len(results),
                    "requery_performed": requery_performed,
                }
            )

        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_chain_end"):
                    cb.on_chain_end(response.model_dump())

        return response
