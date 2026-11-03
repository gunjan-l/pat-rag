# Spec: Core RAG Components 1, 2, and 3 (Open-Source Architecture)

## Problem Statement

Users require an open-source, local-first retrieval-augmented generation (RAG) backend implementing document extraction, vector indexing, agentic retrieval, full observability, and automated evaluations. The original specification relied on proprietary Azure PaaS services (Azure AI Document Intelligence, Azure AI Search, Azure AI Foundry IQ, Application Insights, Azure AI Evaluation SDK), creating vendor lock-in and high cloud dependencies. The system needs to run on local containerized infrastructure (Podman) with standard open-source libraries (Docling, LangChain, Qdrant, Langfuse, Ragas) while interfacing with OpenAI-compatible APIs for embeddings and LLM reasoning, backed by GitHub Actions CI/CD automation.

## Solution

A modular FastAPI service implementing the first three core components of the RAG pipeline plus observability, evaluation, and CI/CD:
1. **Component 1 (Extraction - `/extract`)**: Ingests PDF, DOCX, and scanned documents via Docling, returning clean 1-based page-indexed text, structured `[Table N]` pipe-delimited tables, and document metadata.
2. **Component 2 (Indexing & Vector Store - `/chunks`)**: Upserts chunked text into Qdrant running in Podman, combining dense API embeddings with local FastEmbed BM25 sparse vectors for hybrid retrieval with RRF rank fusion, strictly isolating data by tenant namespace.
3. **Component 3 (Retrieval & Fallback - `/retrieve`)**: Executes hybrid search against Qdrant, calculates composite confidence scores, triggers automated agentic re-querying when confidence is low, rewrites conversational queries, and returns citations and plain-language explanations.
4. **Observability & Tracing**: Integrated Langfuse tracing via LangChain callbacks and FastAPI middleware, capturing latency, token consumption, query rewrites, and retrieval spans. Deployable in Podman.
5. **Evaluation Framework**: Automated evaluation suite using Ragas measuring Context Precision, Context Recall, Faithfulness, and Answer Relevance.
6. **CI/CD Automation**: GitHub Actions workflow running tests, lint checks, and container smoke tests.

## User Stories

1. As an API client, I want to upload a native or scanned PDF/DOCX to `/extract`, so that I receive clean plain text without configuring external OCR engines.
2. As a downstream chunking system, I want extracted text to preserve 1-based page numbers, so that all generated chunks are attributed to verifiable physical pages.
3. As a downstream chunking system, I want detected tables rendered as `[Table N]` markers followed by pipe-delimited rows, so that table structures remain intact without requiring custom parsers.
4. As an ingestion pipeline, I want document-level metadata (title, author, creation date, revision) extracted into a dedicated object, so that document records can be cataloged automatically.
5. As an API consumer, I want malformed or corrupt files to return structured error codes (`UNSUPPORTED_FORMAT`, `CORRUPT_FILE`, `OCR_TIMEOUT`) instead of 500 errors, so that client applications can handle ingest failures gracefully.
6. As a multi-tenant application, I want to upsert chunk text and metadata into isolated namespaces, so that tenants cannot view or search other tenants' documents.
7. As an index maintainer, I want metadata fields not applicable to a chunk to be omitted rather than stored as null, so that the index remains compact and clean.
8. As a search client, I want hybrid search combining dense semantic vectors and sparse lexical BM25 matching via Reciprocal Rank Fusion (RRF), so that both conceptual meaning and exact terminology are matched accurately.
9. As a search client, I want to apply metadata filters (such as filtering by source document name) to my queries, so that search results can be restricted to specific documents.
10. As a compliance administrator, I want to delete all chunks belonging to a specific namespace or a specific document within a namespace, so that obsolete or requested data can be purged cleanly.
11. As an ingestion pipeline, I want a helper endpoint to chunk raw document text and upsert it into the vector database in a single call, so that end-to-end ingestion from extraction to storage is streamlined.
12. As a reviewer, I want every retrieved chunk to include source filename, page number, and quoted span, so that findings can be verified directly against the source document.
13. As a reviewer, I want a short, plain-language explanation for why each chunk was retrieved, so that I can understand why the system surfaced that specific text.
14. As an application, I want retrieval results to explicitly indicate overall confidence (`high` or `low`), so that low-confidence results are never silently trusted as authoritative.
15. As a search client, I want an automatic agentic re-query loop when retrieval confidence falls below a configured threshold, so that suboptimal initial queries are reformulated to recover relevant context.
16. As a search client, I want the agentic re-query fallback loop to be strictly bounded by `max_requery_attempts`, so that low-confidence queries do not cause runaway latency.
17. As an interactive conversational client, I want multi-turn questions containing pronouns ("what does it say about...") rewritten into self-contained search queries, so that conversational context is preserved during retrieval.
18. As an API client, I want a standalone `/retrieve/rewrite` endpoint, so that I can perform conversational query rewriting independently of retrieval execution.
19. As an operator, I want all retrieval calls, query rewrites, and LLM inferences traced in Langfuse, so that I can inspect latency breakdowns, token usage, and execution graphs in real time.
20. As an engineer, I want automated Ragas evaluation suites, so that I can benchmark context recall, precision, and faithfulness across prompt and embedding iterations.
21. As a team, I want GitHub Actions CI pipelines, so that pull requests are automatically validated for linting, typing, and test regressions.
22. As an infrastructure engineer, I want services (Qdrant, Langfuse) orchestrated via Podman, so that deployment is reproducible and lightweight.
23. As a system consumer, I want all endpoints to communicate via plain HTTP JSON REST, so that integration requires no vendor-specific SDKs.

## Implementation Decisions

### Architectural Decisions & Modules
- **Application Core**: Single modular FastAPI service.
- **Component 1 Module**: Docling-powered extraction service handling PDF/DOCX conversion, OCR layer checks, table formatting, and document metadata extraction. Tempfile buffers used for file processing.
- **Component 2 Module**: Qdrant vector database client in Podman. Dense embeddings generated via OpenAI-compatible API; sparse BM25 vectors generated in-process via FastEmbed. Namespace isolation enforced via Qdrant payload filters. Bulk deletion by namespace or document.
- **Component 3 Module**: LangChain agentic retrieval pipeline. Composite confidence calculation combining normalized hybrid retrieval scores with query keyword lexical overlap. Automated query reformulation loop on low confidence. Conversational query rewrite supporting history context.
- **Observability Module**: Langfuse integration with LangChain `CallbackHandler` passed to LLM calls and retrieval traces. Enabled conditionally via `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`.
- **Evaluation Module**: Ragas test runner script (`evals/run_evals.py`) executing dataset evaluations against Component 2 & 3.
- **CI/CD Automation**: GitHub Actions workflow `.github/workflows/ci.yml` running pytest and code formatting checks.

### API Contracts

**Component 1: Document Extraction (`POST /extract`)**
- Input: `multipart/form-data` with `file`, `filename`, optional `force_ocr` (boolean); or JSON with `blob_url`, `filename`, `force_ocr`.
- Output:
  ```json
  {
    "filename": "string",
    "extraction_method": "ocr | native_text",
    "document_metadata": {
      "title": "string",
      "author": "string",
      "created_date": "string",
      "revision": "string",
      "revision_date": "string"
    },
    "pages": [
      {
        "page_num": 1,
        "text": "string",
        "tables": [
          {
            "table_index": 0,
            "rows": [["Header1", "Header2"], ["Val1", "Val2"]]
          }
        ],
        "confidence": 0.95
      }
    ],
    "warnings": []
  }
  ```

**Component 2: Chunk Upsert & Query (`POST /chunks`, `POST /chunks/query`, `POST /chunks/delete`)**
- Upsert Input:
  ```json
  {
    "namespace": "tenant_id",
    "chunks": [
      {
        "chunk_text": "text content",
        "metadata": {
          "source": "filename.pdf",
          "chunk": 0,
          "page_num": 1
        }
      }
    ]
  }
  ```
- Upsert Output: `{"namespace": "tenant_id", "upserted_count": 1, "failed": []}`
- Query Input: `{"namespace": "tenant_id", "query_text": "query", "top_k": 4, "metadata_filter": {"source": ["file.pdf"]}}`
- Query Output: `{"matches": [{"score": 0.88, "chunk_text": "...", "metadata": {"source": "file.pdf", "chunk": 0, "page_num": 1}}]}`
- Delete Input: `{"namespace": "tenant_id", "scope": "namespace | document", "source": "file.pdf"}`
- Delete Output: `{"namespace": "tenant_id", "deleted_count": 1}`

**Component 3: Agentic Retrieval (`POST /retrieve`, `POST /retrieve/rewrite`)**
- Retrieve Input:
  ```json
  {
    "namespace": "tenant_id",
    "query_text": "search query",
    "conversation_context": ["prior turn 1"],
    "top_k": 4,
    "confidence_threshold": 0.55,
    "max_requery_attempts": 1
  }
  ```
- Retrieve Output:
  ```json
  {
    "results": [
      {
        "chunk_text": "...",
        "score": 0.85,
        "source": {
          "filename": "file.pdf",
          "page_num": 1,
          "quoted_span": "verbatim text..."
        },
        "explanation": "Matches terms '...' with high semantic similarity"
      }
    ],
    "confidence": "high | low",
    "requery_performed": false,
    "requery_reason": null,
    "rewritten_query": "search query"
  }
  ```
- Rewrite Input: `{"query_text": "what does it say", "conversation_context": ["prior context"]}`
- Rewrite Output: `{"rewritten_query": "expanded self-contained query"}`

## Testing Decisions

- **Good Test Criteria**: Tests must verify externally observable behaviors and API contracts over HTTP. Tests must not assert against private class methods or internal data structures.
- **Testing Seam**: The highest available seam is the FastAPI HTTP application interface via `httpx.AsyncClient` / `TestClient`.
- **Test Strategy**:
  - Unit/Component tests mock external OpenAI-compatible API calls (embeddings and chat completions) to ensure deterministic, offline execution.
  - Integration tests verify Qdrant storage operations and hybrid search against a test collection.
  - Contract validation tests ensure input/output JSON schemas strictly match `requirements.md`.
  - Evaluation tests run Ragas metrics against synthetic test ground-truth sets.

## Out of Scope

- Components 4 through 6, 8, 9 (Encryption at rest audit, Purview DLP, Audit trail sink, Foundry Agent state orchestration, Action plan generator).
- Proprietary Azure PaaS deployment scripts.
- Frontend user interfaces.

## Further Notes

- Container orchestration: Podman compose definition provided for Qdrant and Langfuse (`podman-compose.yml`).
- GitHub Actions CI in `.github/workflows/ci.yml`.
