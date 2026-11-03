# ADR 0002: Observability via Langfuse, Evaluations via Ragas, and GitHub Actions CI/CD

## Status
Accepted

## Context
Production RAG systems require:
1. End-to-end tracing and latency/cost attribution across retrieval and generation steps.
2. Automated evaluation of retrieval quality and faithfulness without relying only on manual testing.
3. Automated continuous integration and validation.

Azure Monitor/Application Insights and Azure AI Evaluation SDK were originally proposed in `requirements.md` Component 7. Open-source replacements must be deployable locally via Podman.

## Decision
1. **Observability & Tracing**: Deploy Langfuse via Podman. Integrate `langfuse-fastapi` / LangChain `CallbackHandler` into the FastAPI request lifecycle and LangChain chains. Every request, query rewrite, Qdrant search, and LLM call gets traced with span timings and token counts.
2. **Evaluation Framework**: Use `ragas` for evaluating the pipeline. Measure:
   - `context_precision` and `context_recall` on Component 2 & 3 retrieval.
   - `faithfulness` and `answer_relevancy` on synthesized outputs.
   - Run evaluation suite programmatically or as an offline CLI/test command.
3. **CI/CD Automation**: Provide GitHub Actions workflows (`.github/workflows/ci.yml`) to run linting (ruff), type checks, test suites with mocked external APIs, and container build checks.

## Consequences
- **Positive**: Complete observability with zero cloud lock-in. Full visibility into latency bottlenecks and token costs. Automated RAG regression testing with standard open-source metric framework.
- **Negative**: Langfuse container requires PostgreSQL in Podman. Langfuse credentials (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) must be managed in `.env`.
