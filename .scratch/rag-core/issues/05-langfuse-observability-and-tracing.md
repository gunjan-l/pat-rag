# 05: Langfuse Observability & Tracing Instrumentation

**What to build:** Comprehensive request tracing and observability across FastAPI endpoints, LangChain retrieval chains, and LLM calls using Langfuse running in Podman, capturing execution graphs, latency breakdowns, and token metrics.

**Blocked by:** 02: Document Extraction Service (Component 1), 04: Agentic Retrieval, Confidence Fallback & Query Rewrite (Component 3)

**Status:** ready-for-agent

- [ ] Langfuse client and callback handlers are initialized from environment settings (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- [ ] Tracing degrades gracefully to a no-op if Langfuse credentials are not provided, preventing local development crashes.
- [ ] FastAPI middleware traces incoming HTTP requests, associating traces with user IDs, session IDs, and tenant namespaces where available.
- [ ] LangChain retrieval, query rewrites, and re-query loops propagate Langfuse callbacks, recording nested spans, input prompts, output completions, token costs, and latency.
- [ ] Automated tests verify traces are dispatched with expected span tags and without adding blocking latency overhead.
