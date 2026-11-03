# ADR 0001: Open-Source Architecture for Components 1, 2, and 3

## Status
Accepted

## Context
The platform requirements originally specified proprietary Azure PaaS services:
- Component 1: Azure AI Document Intelligence + Azure Blob Storage
- Component 2: Azure OpenAI Embeddings + Azure AI Search (Hybrid + RRF)
- Component 3: Azure AI Foundry IQ (Agentic Retrieval)

The project requirement is to replace all Azure services with open-source libraries and containerized services using LangChain, Docling, and Podman, while keeping external API integration for embedding and LLM inference.

## Decision
1. **Component 1 (Extraction)**: Use `docling` in-process. Support PDF, DOCX, and scanned documents with OCR fallback. Format tables into `[Table N]` pipe-delimited format and preserve 1-based page numbers. Direct local tempfile buffer instead of cloud object storage.
2. **Component 2 (Vector Database & Hybrid Search)**: Deploy Qdrant in a Podman container. Use OpenAI-compatible API for dense embeddings and in-process `fastembed` for sparse BM25 vectors. Qdrant handles multi-tenant namespace isolation via payload filters and merges results using Reciprocal Rank Fusion (RRF).
3. **Component 3 (Agentic Retrieval & Confidence)**: Built with LangChain using OpenAI-compatible LLM for query reformulation and explanations. Evaluate retrieval confidence via composite metric (hybrid score + query keyword coverage). Automatically trigger bounded fallback re-query when below threshold. Expose both `/retrieve` and standalone `/retrieve/rewrite`.
4. **Service Structure & Execution Environment**: Unified modular FastAPI application hosting all three components with strict contract adherence to the original REST specification. **Crucial rule**: No libraries or tools are installed on the host machine; all Python execution, `uv` dependency management, test suites, and background services run entirely inside Podman containers.

## Consequences
- **Positive**: Zero Azure cloud dependencies, fully containerized local development via Podman, fast in-memory table extraction via Docling, robust open-source hybrid search via Qdrant.
- **Negative**: Requires running Qdrant container locally via Podman. External API key required for dense embeddings and LLM query rewrites.
