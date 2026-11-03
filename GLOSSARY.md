# GLOSSARY

## Terms

### Namespace
Logical tenant identifier used to isolate document storage, vector chunks, and search queries. In multi-tenant retrieval, every operation (upsert, query, delete) is scoped to a namespace.

### Extraction
Process of taking raw document binary (PDF, DOCX, image) and producing page-indexed plain text, structured table text, and document-level metadata without chunking or embedding. Handled via Docling.

### Chunk
Discrete segment of document text with mandatory 1-based page number attribution, source identifier, and optional formatting metadata. Smallest unit of retrieval.

### Hybrid Search
Retrieval technique executing dense vector similarity search (via external embedding API) and sparse lexical search (BM25 via FastEmbed) simultaneously in Qdrant, merged via Reciprocal Rank Fusion (RRF).

### Reciprocal Rank Fusion (RRF)
Score merging algorithm that combines ranked lists from dense vector search and sparse keyword search without needing normalized vector distance calibrations.

### Confidence Score
Composite metric (0.0 to 1.0) evaluating retrieval quality by combining normalized hybrid similarity scores with lexical query keyword coverage across retrieved chunks.

### Agentic Retrieval & Fallback
Two-stage query execution: initial search evaluated against confidence threshold; if score or keyword coverage falls below threshold, automated LLM query reformulation triggers a second bounded retrieval pass.

### Conversational Query Rewrite
Pre-retrieval reformulation step that transforms a multi-turn conversational question containing ambiguous pronouns or shorthand into an explicit, self-contained search query.

### Observability & Tracing (Langfuse)
Instrumentation tracking requests, token counts, latencies, retrieval steps, and LLM inputs/outputs end-to-end via Langfuse containers in Podman and LangChain callbacks.

### RAG Evaluation (Ragas)
Framework executing automated quantitative evaluation on retrieval and generation pipelines, measuring metrics including Faithfulness, Answer Relevance, Context Precision, and Context Recall.
