import logging
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.services.vector_store import vector_store_service

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
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
    ):
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.model = model or settings.completion_model

    async def rewrite(self, query_text: str, conversation_context: list[str]) -> str:
        if not conversation_context:
            return query_text

        if not self.api_key:
            return f"{query_text} ({' '.join(conversation_context)})"

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
                content = data["choices"][0]["message"]["content"].strip().strip("\"'")
                return content or query_text
        except Exception as exc:
            logger.warning("LLM rewrite failed, falling back to deterministic rewrite: %s", exc)
            return f"{query_text} ({' '.join(conversation_context)})"

    async def reformulate(self, query_text: str, reason: str | None = None) -> str:
        if not self.api_key:
            return f"{query_text} detailed specification"

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
                content = data["choices"][0]["message"]["content"].strip().strip("\"'")
                return content or query_text
        except Exception as exc:
            logger.warning(
                "LLM reformulate failed, falling back to deterministic expansion: %s", exc
            )
            return f"{query_text} detailed specification"


class RetrievalService:
    """Agentic retrieval service with conversational rewriting and confidence fallback."""

    def __init__(
        self,
        vector_store: Any | None = None,
        rewriter: LLMQueryRewriter | None = None,
    ):
        self.vector_store = vector_store or vector_store_service
        self.rewriter = rewriter or LLMQueryRewriter()

    async def rewrite_query(self, query_text: str, conversation_context: list[str]) -> str:
        return await self.rewriter.rewrite(query_text, conversation_context)

    async def retrieve(
        self,
        namespace: str,
        query_text: str,
        conversation_context: list[str] | None = None,
        top_k: int = 4,
        confidence_threshold: float = 0.55,
        max_requery_attempts: int = 1,
    ) -> RetrieveResponse:
        context = conversation_context or []

        # 1. Automatically rewrite conversational query if context is non-empty
        if context:
            rewritten_query = await self.rewriter.rewrite(query_text, context)
        else:
            rewritten_query = query_text

        current_query = rewritten_query

        # 2. Query vector store
        query_res = await self.vector_store.query_chunks(
            namespace=namespace,
            query_text=current_query,
            top_k=top_k,
        )
        matches = query_res.get("matches", [])
        query_terms = extract_query_terms(current_query)
        initial_confidence = compute_composite_confidence(query_terms, matches)

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
            reformulated_query = await self.rewriter.reformulate(
                current_query, reason=requery_reason
            )
            current_query = reformulated_query

            new_query_res = await self.vector_store.query_chunks(
                namespace=namespace,
                query_text=current_query,
                top_k=top_k,
            )
            new_matches = new_query_res.get("matches", [])
            new_terms = extract_query_terms(current_query)
            new_confidence = compute_composite_confidence(new_terms, new_matches)

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

        return RetrieveResponse(
            results=results,
            confidence=confidence_label,
            requery_performed=requery_performed,
            requery_reason=requery_reason,
            rewritten_query=rewritten_query,
        )
