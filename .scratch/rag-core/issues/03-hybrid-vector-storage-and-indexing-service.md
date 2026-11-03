# 03: Hybrid Vector Storage & Indexing Service (Component 2)

**What to build:** An HTTP indexing service (`POST /chunks`, `POST /chunks/query`, `POST /chunks/delete`, `POST /documents/chunk-and-upsert`) backed by Qdrant in Podman, implementing dense API embeddings combined with local FastEmbed BM25 sparse vectors for hybrid retrieval with Reciprocal Rank Fusion (RRF), strict multi-tenant namespace isolation, and bulk deletion.

**Blocked by:** 01: Project Scaffolding & Container Environment

**Status:** completed

- [x] `POST /chunks` accepts an array of chunk text objects with metadata and upserts them into Qdrant under a specific `namespace`.
- [x] Dense vectors are generated via OpenAI-compatible embedding API; sparse vectors are generated in-process via FastEmbed BM25.
- [x] Non-applicable metadata fields are omitted entirely from stored chunks rather than stored as null values.
- [x] `POST /chunks/query` runs hybrid search combining dense similarity and sparse lexical BM25 matching via RRF, returning ranked matches with scores and metadata.
- [x] Querying supports metadata filtering (e.g., filtering by `source` filename) and strictly isolates search within the specified `namespace`.
- [x] `POST /chunks/delete` supports deleting all chunks within a namespace, or all chunks belonging to a specific document `source` within a namespace.
- [x] `POST /documents/chunk-and-upsert` accepts full document text, chunks it using LangChain RecursiveCharacterTextSplitter while preserving page numbers, and upserts it in one step.
- [x] Automated contract tests verify upsert, hybrid query ranking, metadata filtering, and deletion against the Component 2 schema.

