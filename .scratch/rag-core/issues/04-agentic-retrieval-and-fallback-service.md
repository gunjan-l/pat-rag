# 04: Agentic Retrieval, Confidence Fallback & Query Rewrite (Component 3)

**What to build:** An HTTP agentic retrieval service (`POST /retrieve`, `POST /retrieve/rewrite`) that queries the vector store, scores retrieval confidence using a composite metric, automatically triggers an agentic query reformulation fallback pass when confidence is low, and returns verifiable source citations and plain-language explanations.

**Blocked by:** 03: Hybrid Vector Storage & Indexing Service (Component 2)

**Status:** ready-for-agent

- [ ] `POST /retrieve` accepts query text, tenant namespace, optional conversation context, and confidence thresholds.
- [ ] Multi-turn conversational questions with ambiguous references are automatically rewritten into self-contained search queries prior to retrieval.
- [ ] Standalone `POST /retrieve/rewrite` exposes conversational query rewriting independently.
- [ ] Composite confidence score is computed from normalized hybrid retrieval scores combined with lexical query term coverage across retrieved chunks.
- [ ] If confidence falls below `confidence_threshold`, an automated LLM query reformulation re-query loop is executed, bounded by `max_requery_attempts`.
- [ ] When confidence remains low after retries, results are returned gracefully flagged as `confidence: "low"` rather than returning empty results or crashing.
- [ ] Every result contains a source citation (filename, page number, quoted span) and a concise plain-language explanation for why the chunk was surfaced.
- [ ] Automated contract tests verify confidence degradation, query rewriting, re-query execution, and output contract conformity.
