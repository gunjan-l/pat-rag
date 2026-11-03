
**PAT AI Platform**

**Azure AI Component Build Requirements**

_Standalone technical specifications for nine independently-buildable AI/data components_

Version: 1.0

# How to Read This Document

This document specifies nine standalone AI/data components. Each component is written to be built independently, as a self-contained Azure-backed service (an HTTP API, an Azure Function, or a managed Azure resource configuration) that exposes a fixed input/output contract. No knowledge of any consuming system is required or assumed to build any component.

Every component section follows the same structure:

- **1\. Overview** — what the component does and the problem it solves.
- **2\. Azure Services** — the exact Azure resources to provision.
- **3\. Functional Requirements** — the detailed behaviour the component must implement.
- **4\. Input Contract** — exact request shape (JSON/binary) the component must accept.
- **5\. Output Contract** — exact response shape the component must return.
- **6\. Interoperability Requirements** — general-purpose API design rules the component must follow (identifier handling, metadata conventions, idempotency, error behaviour) so it can be consumed cleanly by any calling system without further negotiation.
- **7\. Non-Functional Requirements** — performance, security, error handling.
- **8\. Delivery Target** — the committed delivery timeframe for this component.

_Design philosophy: every component MUST be deliverable as a plain HTTP JSON API (REST, synchronous or async-polling) so it can be called from any backend language without requiring Azure SDKs on the caller's side. Where an Azure PaaS service already exposes a REST API (Document Intelligence, Content Safety, Azure AI Search, Azure OpenAI, Foundry Agent Service), the deliverable is the \*provisioned resource plus a thin adapter conforming to the contract below\*, not a hand-rolled wrapper service._

# Component 1: Document Upload & Text Extraction (OCR-augmented Ingestion) – ()

## 1. Overview

A service that accepts an uploaded document (PDF, DOCX, or scanned/image-only PDF) and returns clean, page-indexed plain text plus document-level metadata (title, author, dates, revision). It must reliably extract text even from scanned or image-only pages using OCR, and must preserve page numbers and table structure so the output can be fed directly into a downstream chunking/indexing stage without further layout processing.

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- **Azure AI Document Intelligence** (formerly Form Recognizer) — prebuilt "prebuilt-layout" / "prebuilt-read" model via the Analyze Document REST API.
- **Azure Blob Storage** — staging container for files being OCR'd (Document Intelligence can read directly from a SAS URL, avoiding a base64 round-trip for large files).

## 3. Praesidium Requirements:

• "Ingest Word, PDF, scanned, and text-based documents, including support for optical character recognition where needed."

• "Extract key information such as titles, dates, owners, revision history..."

## 4. Functional Requirements

1. **Accept a document and detect whether it has a usable text layer.** If the page/document has no extractable text layer (image-only/scanned), route it to Document Intelligence OCR; otherwise extract text natively.
2. **Call Azure AI Document Intelligence's "prebuilt-layout" model** to extract: full text per page, detected tables (as structured rows/columns, not flattened text), and reading order. Handwriting, barcodes, and rotated/skewed scans must be supported (Document Intelligence handles these natively).
3. **Preserve page numbers.** Every extracted text span must be attributable to a 1-based page number — this is mandatory, not optional.
4. **Render detected tables as text** using a consistent, documented convention: a \`\[Table N\]\` marker followed by pipe-delimited rows (\`Header1 | Header2\`), so table content is preserved as plain text without inventing a table representation a downstream consumer would need custom logic to parse.
5. **Extract key document-level metadata fields** where present in the document: title, author/owner, creation/revision date, and revision history — returned as a metadata block distinct from the page text.
6. **Support async processing for large files.** Document Intelligence's Analyze API is asynchronous (submit → poll operation-location → retrieve result). The component must expose this either as a blocking call that polls internally (bounded by a timeout, e.g. 120s) for typical file sizes, or as a submit/poll pair of endpoints for very large files.
7. **Return a structured, page-indexed text result** — never a single flattened string — so it can be handed directly to a chunker without re-splitting.

## 4. Input Contract

**Option A — multipart file upload:**

POST /extract
Content-Type: multipart/form-data

file: <binary PDF/DOCX/image>
filename: "policy\_v3.pdf" (string, required)
force\_ocr: false (bool, optional — skip native-text-layer check, always OCR)

**Option B — reference to an already-uploaded blob (preferred for large files, avoids double upload):**

{
"blob\_url": "https://<account></account>.blob.core.windows.net/<container></container>/<path></path>?<sas></sas>",
"filename": "policy\_v3.pdf",
"force\_ocr": false
}

## 5. Output Contract

**Synchronous / final result:**

{
"filename": "policy\_v3.pdf",
"extraction\_method": "ocr" | "native\_text",
"document\_metadata": {
"title": "Accreditation Policy 2026",
"author": "Jane Doe",
"created\_date": "2025-01-10",
"revision": "v3",
"revision\_date": "2026-03-01"
},
"pages": \[
{
"page\_num": 1,
"text": "extracted plain text for this page...",
"tables": \[
{
"table\_index": 0,
"rows": \[\["Header1","Header2"\], \["A","B"\]\]
}
\],
"confidence": 0.94
}
\],
"warnings": \["page 4 had low OCR confidence (0.41)"\]
}

**Error response (any failure):**

{
"error\_code": "UNSUPPORTED\_FORMAT" | "OCR\_TIMEOUT" | "CORRUPT\_FILE",
"message": "human-readable description"
}

## 6. Interoperability Requirements

- **This is a pure extraction service.** It must not perform chunking, embedding, or storage of the extracted content — its responsibility ends at producing clean, page-indexed text plus document metadata; any splitting into smaller retrieval units is a separate downstream concern.
- **Page numbers must always be 1-based integers.** Never 0-based, and never omitted for a PDF page — downstream consumers rely on unambiguous page attribution.
- **Table rendering convention must be fixed and documented** (\`\[Table N\]\` marker + pipe-delimited rows) so any consumer can rely on a single parsing rule regardless of source file type.
- **The \`document\_metadata\` block must always be present in the response** even when no fields could be extracted (return an empty object \`{}\`, not a missing key), so callers can rely on a stable response shape.

## 7. Non-Functional Requirements

- Must handle files up to at least 50MB / 200 pages within a bounded timeout
- Must never crash or 500 on a corrupt/malformed file — always return the structured error\_code contract above.

## 8. Delivery Target

# Component 2: Chunking, Indexing, Embeddings & Vector Database (needs discussion) – ()

## 1. Overview

A service that takes document text and creates chunks with metadata, generates dense vector embeddings for each chunk, and stores/queries them in a hybrid (vector + keyword) search index with tenant/namespace isolation. It must support precise, filterable retrieval with semantic re-ranking so a caller can retrieve exact matching passages rather than approximate paraphrases.

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- **Azure OpenAI — text-embedding-3-large / text-embedding-3-small** via the Foundry Model Catalog, for generating dense vector embeddings from chunked document text.
- **Azure AI Search** as the vector database — hybrid retrieval (dense vector + BM25 keyword search) with built-in Reciprocal Rank Fusion (RRF), plus an integrated skillset pipeline for chunking/embedding/index maintenance and semantic re-ranking.

## 3. Praesidium Requirements:

• "Compare submitted documents against configurable accreditation standards, internal policy templates, checklists, and review criteria."

• "Identify missing sections, outdated language, inconsistent terminology..."

## 4. Functional Requirements

1. **Accept document text and create chunks with metadata** and produce a dense vector embedding per chunk via Azure OpenAI's embedding endpoint.
2. **Upsert each chunk + its vector + its metadata into an Azure AI Search index**, isolated per logical tenant/namespace. Use Azure AI Search's \`filterable\` metadata fields to achieve namespace isolation within a single index, or provision one index per tenant if isolation must be physical — either approach must be supported behind the same API contract.
3. **Preserve every supplied metadata field verbatim**: e.g. \`source\` (filename), \`chunk\` (0-based index within file), \`text\` (duplicate of the chunk for retrieval preview), and format-specific fields such as \`page\_num\`, \`sheet\_name\`, \`slide\_number\`/\`slide\_title\`, \`file\_url\`. Fields that do not apply to a given chunk must be omitted entirely from that chunk's metadata, never set to null — this is a hard requirement.
4. **Support hybrid search queries** combining vector similarity and keyword (BM25) match with RRF fusion, and Azure AI Search's semantic re-ranker, exposed as a single query endpoint returning ranked chunks with relevance scores.
5. **Support metadata filtering on query** (e.g. restrict search to a specific \`source\` filename or set of filenames).
6. **Support per-namespace bulk delete and per-document delete** — namespace-level delete removes all vectors for a namespace/tenant; document-level delete removes all chunks belonging to one \`source\` filename within a namespace.

## 4. Input Contract

**Upsert chunks:**

{
"namespace": "tenant identifier",
"chunks": \[
{
"chunk\_text": "the actual chunk text",
"metadata": {
"source": "policy\_v3.pdf", "chunk": 0,
"text": "the actual chunk text", "page\_num": 1,
"file\_url": "https://...blob.../policy\_v3.pdf?sas"
}
}
\]
}

**Query:**

{
"namespace": "tenant identifier",
"query\_text": "data retention policy requirements",
"top\_k": 4,
"metadata\_filter": {"source": \["policy\_v3.pdf"\]}
}

**Delete:**

{
"namespace": "tenant identifier",
"scope": "namespace" | "document",
"source": "policy\_v3.pdf, required if scope=document"
}

## 5. Output Contract

**Upsert acknowledgement:**

{
"namespace": "...", "upserted\_count": 42, "failed": \[\]
}

**Query result:**

{
"matches": \[
{
"score": 0.87, "chunk\_text": "...",
"metadata": {"source": "policy\_v3.pdf", "chunk": 0, "page\_num": 1,
"file\_url": "https://..."}
}
\]
}

**Delete acknowledgement:**

{
"namespace": "...", "deleted\_count": 12
}

## 8. Delivery Target

# Component 3: Retrieval & Fallback Logic (Agentic Retrieval with Confidence Scoring) – ()

## 1. Overview

A retrieval service that layers confidence scoring, automatic re-query, and reviewer-facing explanations on top of a vector search index, so that low-confidence results are explicitly flagged rather than silently returned, and every result carries an exact source citation a reviewer can verify against the original document.

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- **Azure AI Foundry — Foundry IQ (agentic retrieval)** for multi-step search planning, dynamic context-quality evaluation, and automatic fallback/re-query loops, built on top of the Azure AI Search index from Component 5.

## 3. Praesidium Requirements:

- •"Provide confidence indicators, source references within the reviewed document, and explanations for AI-generated flags."
- Section 10 (AI Governance), Para 1:

• "provide mechanisms for reviewers to verify findings against source documents."

## 3. Functional Requirements

1. **Given a question/query and a namespace,** run an initial retrieval (delegating to Component 5's query endpoint or an equivalent Foundry-native retrieval call), then evaluate the \*confidence\* of the returned context (score distribution, coverage of the query's key terms) and decide: accept as-is, or trigger one additional agentic re-query pass with a reformulated search query, up to a configurable max of retries.
2. **Apply a numeric confidence/score threshold** below which results are flagged as low-confidence rather than silently returned — the caller must always be told when confidence is low, never left to infer it from raw scores.
3. **Attach a source reference to every returned chunk**: filename, page/sheet/slide locator, and (if available) a character offset or the verbatim quoted span, so a reviewer can jump to and verify the exact source passage.
4. **Generate a short, plain-language explanation for why a chunk was retrieved / why a flag was raised** (e.g. "retrieved because it matches the query terms 'data retention period' with 0.83 similarity").
5. **Support two distinct query-rewriting behaviors**: (a) rewriting a conversational follow-up question (e.g. containing "it"/"that"/short fragments) into a self-contained search query using recent conversation turns as context, and (b) a separate agentic re-query rewrite triggered specifically by low retrieval confidence. These are different triggers and must be exposed as distinct, independently callable behaviors even if they share an underlying rewrite mechanism.

## 4. Input Contract

**Retrieve with confidence + fallback:**

{
"namespace": "tenant identifier",
"query\_text": "what does the policy say about data retention",
"conversation\_context": \["prior user turn 1", "prior assistant turn 1"\],
"top\_k": 4, "confidence\_threshold": 0.55, "max\_requery\_attempts": 1
}

## 5. Output Contract

**Retrieval result with confidence:**

{
"results": \[
{
"chunk\_text": "...", "score": 0.81,
"source": {"filename": "policy\_v3.pdf", "page\_num": 4, "quoted\_span": "Records shall be retained for 7 years..."},
"explanation": "Matches query terms 'data retention' with high similarity"
}
\],
"confidence": "high" | "low",
"requery\_performed": false, "requery\_reason": null,
"rewritten\_query": "what is the data retention period specified in the policy"
}

## 6. Interoperability Requirements

- **Must degrade gracefully rather than returning empty/error** — if no result clears the confidence threshold even after retries, return the best available results explicitly flagged \`confidence: "low"\`, never an empty result set with no explanation.
- **\`explanation\` must always be plain natural language**, not a restatement of raw scores — callers rendering this to a human reviewer should not need to translate it further.
- **The re-query fallback path must be bounded** by \`max\_requery\_attempts\` and an overall timeout, so a single low-confidence query cannot stall a response indefinitely.

## 8. Delivery Target

# Component 4: Data Protection — Encryption in Transit & at Rest – ()

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- Azure Key Vault — customer-managed keys (CMK) for envelope encryption, key rotation, and RBAC/Managed-Identity-scoped access to secrets and keys.
- Azure Storage Service Encryption (SSE) — AES-256 encryption at rest for uploaded documents in Blob Storage.
- Database storage-level encryption at rest for the relational store holding chat history, user\_conversation\_history, and audit logs.
- Vector store / Azure AI Search encryption at rest for embeddings, with optional CMK to match the primary document store's key-management mode.
- TLS 1.2+ (TLS 1.3 preferred) enforced on every hop: client → BEI API, BEI → Azure OpenAI / Content Safety, BEI → storage, BEI → database.

## 3. Praesidium Requirements:

Protect data in transit and at rest using industry-standard encryption.

**4\. Input Contract** _(internal/audit query — this is an infrastructure control, not a per-request check)_

{
  "check\_type": "encryption\_status",
  "resource": "documents" | "database" | "vector\_store" | "backups"
}

**5\. Output Contract**

{
  "resource": "documents",
  "in\_transit": {"protocol": "TLS", "min\_version": "1.2"},
  "at\_rest": {
    "algorithm": "AES-256",
    "key\_management": "customer-managed" | "platform-managed",
    "key\_vault\_ref": "kv-bei-prod/keys/doc-store-key"
  },
  "last\_verified": "2026-08-05T10:15:00Z"
}

**9\. Delivery Target**

# Component 5: Data Governance & Compliance (Classification, DLP, Retention) – ()

## 1. Overview

A service that classifies uploaded documents (type, accreditation area, standard, category, review status), scans document text for data-loss-prevention risk categories (bulk PII, credentials, financial data), and evaluates configurable retention/archival/deletion rules per document category.

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- **Microsoft Purview** — Information Protection (sensitivity labels: Confidential/Internal/Public, PII auto-classification via built-in Sensitive Information Types) and Data Loss Prevention policies.
- **Azure Blob Storage lifecycle management** — for automated archival/deletion rules driven by retention metadata.

## 3. Praesidium Requirements:

• "Classify documents by type, source, accreditation area, standard, policy category, and review status."

• "Support configurable retention, deletion, and archival rules..."

• "Provide clear controls for whether organizational data may be used to train... models."

## 4. Functional Requirements

1. **Classify a document at ingestion time** into: document type (policy/standard/checklist/report/other), accreditation area, standard/framework name, policy category, and review status (draft/under-review/approved/expired) — using a combination of Purview's built-in sensitive-information-type detectors (for PII/Confidential auto-labelling) and a classification call against the document's extracted text using an LLM-based or rules-based classifier for domain-specific taxonomy fields, since Purview's built-in labels will not know a domain-specific accreditation taxonomy out of the box.

## 4. Input Contract

**Classify a document:**

{
"document\_id": "uuid, supplied by caller",
"filename": "policy\_v3.pdf",
"extracted\_text": "full text or first N pages",
"uploaded\_by": "user identifier",
"uploaded\_at": "2026-08-05T09:00:00Z"
}

## 5. Output Contract

**Classification result:**

{
"document\_id": "uuid",
"document\_type": "policy",
"accreditation\_area": "Clinical Governance",
"standard": "ISO 9001",
"policy\_category": "Data Retention",
"review\_status": "approved",
"classified\_at": "2026-08-05T09:00:05Z"
}

**Governance dashboard summary:**

{
"by\_review\_status": {"approved": 120, "expired": 8, "draft": 15},
"pending\_lifecycle\_actions": 8,
"generated\_at": "2026-08-05T09:00:00Z"
}

## 6. Interoperability Requirements

- **\`document\_id\` is always an opaque identifier supplied by the caller.** This component must never generate its own document identifiers or require the caller to look one up — every request that references a document must carry the ID as an input field.
- **Classification results must be retrievable by document\_id at any later time** (idempotent lookup), not just returned once at classification time.

## 8. Delivery Target

# Component 6: Audit Trail & Security Accountability – ()

## 1. Overview

An append-only, tamper-evident audit log service that records every significant platform event — document uploads, AI-generated outputs, reviewer decisions, guardrail triggers, configuration changes, and final dispositions — and supports full-text/structured search and export for eDiscovery and forensic review.

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- **Microsoft Purview Audit (Universal Audit Log)** — for cross-service, tamper-evident activity logging if the platform later spans multiple Microsoft 365 / Azure data surfaces.
- **Azure Monitor / Log Analytics + Application Insights** — for control-plane/API-level diagnostic logs (config changes, key usage) and for structured application event ingestion via the standard Application Insights SDK or REST Data Collector API.

## 3. Praesidium Requirements:

- "while maintaining human oversight and defensible audit trails."

• "Maintain complete audit logs of uploads, AI outputs, reviewer decisions, changes, and final dispositions."

## 3. Functional Requirements

1. **Provide a single "write an audit event" endpoint** that accepts a structured event from any caller (upload, AI output generated, reviewer decision, configuration change, disposition) and appends it immutably.
2. **Guarantee append-only, non-repudiable storage** — no update or delete API should be exposed for individual events; corrections are new events referencing the original (\`corrects\_event\_id\`), never in-place edits.
3. **Support full-text and structured search/export** across all logged events for eDiscovery/forensics: filter by event\_type, user\_id, document\_id, conversation\_id, date range; export matching events as JSON or CSV.
4. **Capture, at minimum, these event types**: \`document\_uploaded\`, \`ai\_output\_generated\`, \`reviewer\_decision\` (approve/reject/edit/escalate), \`guardrail\_triggered\`, \`config\_changed\`, \`final\_disposition\`.
5. **Timestamp every event with server-side UTC time** (never trust a client-supplied timestamp for the authoritative record — a client timestamp may be stored as a secondary reference field only).

## 4. Input Contract

**Write an audit event:**

{
"event\_type": "document\_uploaded" | "ai\_output\_generated" | "reviewer\_decision"
| "guardrail\_triggered" | "config\_changed" | "final\_disposition",
"actor": {"user\_id": "identifier", "role": "reviewer" | "system" | "admin"},
"document\_id": "uuid, optional",
"conversation\_id": "uuid, optional",
"payload": { "...event-type-specific structured detail..." },
"client\_timestamp": "2026-08-05T09:00:00Z, optional",
"corrects\_event\_id": "uuid, optional — only for corrections"
}

**Search/export events:**

{
"event\_type": "reviewer\_decision", "document\_id": "uuid, optional",
"from": "2026-07-01T00:00:00Z", "to": "2026-08-05T23:59:59Z",
"format": "json" | "csv"
}

## 5. Output Contract

**Write acknowledgement:**

{
"event\_id": "uuid", "recorded\_at": "2026-08-05T09:00:00.123Z",
"immutable": true
}

**Search result:**

{
"events": \[
{
"event\_id": "uuid", "event\_type": "reviewer\_decision",
"actor": {"user\_id": "reviewer@example.com", "role": "reviewer"},
"document\_id": "uuid", "payload": {"decision": "approve", "reason": "meets standard"},
"recorded\_at": "2026-08-05T09:00:00.123Z"
}
\],
"total\_count": 1, "next\_page\_token": null
}

## 6. Interoperability Requirements

- **This component is a pure audit sink** — it does not originate events itself; every event is written by a caller at the point that event occurs elsewhere.
- **Writes must be safe to call "fire-and-forget" from a caller's hot path** without requiring the caller to block on full durability confirmation.
- **Retention of audit events must be independent of any related document/conversation lifecycle** — this store must never cascade-delete an audit event when the document or conversation it references is deleted elsewhere.
- **Access to search/export must itself be access-controlled and logged** (an audit trail of who read the audit trail).

## 8. Delivery Target

# Component 7: Evaluation (Monitoring & Quality Audit) – ()

## 1. Overview

An evaluation harness that runs a set of test cases against a live endpoint, scores responses on groundedness/relevance/fluency/coherence, aggregates pass rates over time for regression detection, and emits distributed traces for latency attribution across the ingestion/safety/retrieval/generation pipeline.

## 2. Azure Services Suggested ( Please research on this and use the appropriate azure service)

- **Azure AI Evaluation SDK** — automated GenAI-specific evaluators: Groundedness, Relevance, Fluency, Coherence, and Risk & Safety (harmful content) evaluators, run against a set of test cases or sampled live traffic.
- **Azure Monitor / Application Insights** — end-to-end distributed tracing (via OpenTelemetry) for latency breakdowns (extraction time, retrieval time, token counts, LLM decision paths) and production alerting on anomalies.

## 3. Praesidium Requirements:

Testing plans, test results, issue logs, and remediation documentation.

## 3. Functional Requirements

1. **Accept a set of evaluation cases** (\`input\` question, \`expected\_contains\` substrings, \`expected\_tool\`, \`should\_not\_contain\`, \`is\_fallback\_expected\`) and run them against a live endpoint, computing: substring/tool-match pass/fail, plus Groundedness/Relevance/Fluency/Coherence scores via the Azure AI Evaluation SDK for each case's actual response.
2. **Aggregate results into a run summary**: total/passed/failed counts, pass rate, per-case latency, and mean scores per evaluator dimension — stored so historical runs can be compared over time (regression detection).
3. **Support promoting a human-corrected response into the eval case set** — i.e. when a human reviewer corrects an AI output, that correction should be capturable as a new eval case with \`source: human\_correction\`, closing the loop between review and regression testing.
4. **Emit end-to-end distributed traces** via OpenTelemetry for every request through the pipeline (ingest → safety → retrieve → generate), tagged with span-level timing so latency regressions can be attributed to a specific stage, and exported to Azure Monitor/Application Insights for dashboarding and alerting.
5. **Expose a formatted test-plan/test-result report** (pass rate, evaluator score trends, open issues list) suitable for inclusion in testing/acceptance documentation, not just raw JSON.

## 4. Input Contract

**Create an eval case:**

{
"input": "does the policy meet ISO 9001 clause 4.2?",
"expected\_contains": \["retention period"\], "expected\_tool": "retrieve",
"should\_not\_contain": \["I don't know"\], "is\_fallback\_expected": false,
"source": "manual" | "human\_correction"
}

**Run an eval suite:**

{
"target\_endpoint": "chat" | "policy\_review\_workflow",
"case\_ids": \["uuid", "..."\]
}

## 5. Output Contract

**Eval run result:**

{
"run\_id": "uuid", "total": 20, "passed": 17, "failed": 3, "pass\_rate": 0.85,
"evaluator\_scores": {"groundedness": 0.88, "relevance": 0.91, "fluency": 0.95, "coherence": 0.93},
"results": \[
{
"case\_id": "uuid", "passed": true, "latency\_ms": 1200,
"actual\_output": "...", "evaluator\_scores": {"groundedness": 0.9}
}
\],
"ran\_at": "2026-08-05T09:00:00Z"
}

**Formatted test-plan report:**

{
"report\_period": "2026-07-01..2026-08-05", "overall\_pass\_rate": 0.85,
"score\_trend": {"groundedness": \[0.82, 0.85, 0.88\]}, "open\_issues": \[{"case\_id": "uuid", "description": "..."}\]
}

## 6. Interoperability Requirements

- **Eval runs must execute in an isolated evaluation context**, never against production conversation/session data — running a suite must not pollute real usage history or analytics.
- **The \`evaluator\_scores\` field names (\`groundedness\`, \`relevance\`, \`fluency\`, \`coherence\`) must remain fixed** so any dashboard consuming this component's output can rely on a stable set of keys across runs.
- **Trace export must be additive and non-blocking** — must not meaningfully increase the latency of the request path being instrumented (target: <5ms overhead per span).

## 8. Delivery Target

# Component 8: Agent Orchestration & Backend Runtime (needs discussion-migration) ()

## 1. Overview

A managed orchestration layer that runs a multi-step document-review workflow (extract → classify → safety-check → retrieve-and-compare → generate action plan → human review → finalize), persisting workflow state, streaming live progress, and supporting pause-for-approval / resume-after-decision semantics for any step that requires human sign-off.

## 2. Azure Services

- **Azure AI Foundry Agent Service** — fully managed, serverless agent state and tool-calling orchestration, with native support for Microsoft Agent Framework and OpenAI-compatible \`/responses\`-style endpoints (so LangGraph/LangChain-based tooling can attach without a custom protocol adapter).

## 3. Functional Requirements

1. **Define an agent whose available tools are**: document ingestion/extraction, safety check, governance classification, retrieval, and structured generation — each wired in as a callable tool/function the orchestrated agent can invoke in sequence or in response to intermediate results.
2. **Manage multi-step state transitions automatically**: upload → extract → classify → safety-check → retrieve-and-compare → generate action plan → pause for human review → resume after decision → finalize disposition. The service must persist where a given workflow run currently is, without the caller needing to manage a state machine itself.
3. **Pause execution and emit a distinct "awaiting approval" state** whenever a tool call is classified as requiring human review (per a caller-supplied policy: e.g. any "finalize disposition" action always pauses; a "generate action plan" action never does) — and resume cleanly from that exact paused point once a decision is submitted, carrying the reviewer's decision (approve/reject/edit) into the next step.
4. **Stream intermediate progress events** ("extracting text", "checking safety", "retrieving standards", "generating action plan", "awaiting approval") to the caller in real time via server-sent events or an equivalent streaming transport, so a consuming frontend can render live progress rather than a single blocking spinner.
5. **Expose a \`resume\` endpoint** that accepts a paused workflow's run ID plus the human reviewer's decision payload (approve / reject / edit-with-corrected-input / escalate) and continues orchestration from that exact point.

## 4. Input Contract

**Start a workflow run:**

{
"workflow": "policy\_review",
"document\_id": "uuid",
"namespace": "tenant identifier",
"hitl\_policy": {"pause\_on": \["finalize\_disposition"\]}
}

**Resume a paused run:**

{
"run\_id": "uuid",
"decision": {
"action": "approve" | "reject" | "edit" | "escalate",
"edited\_input": {"...only if action=edit..."},
"reason": "string, optional"
}
}

## 5. Output Contract

**Streamed progress event (one of a sequence):**

{
"run\_id": "uuid", "type": "step\_started" | "step\_completed" | "awaiting\_approval" | "completed" | "error",
"step": "extract" | "safety\_check" | "classify" | "retrieve" | "generate\_plan" | "finalize",
"detail": { "...step-specific structured output..." },
"approval\_id": "uuid, only present when type=awaiting\_approval"
}

**Final run status (also queryable by run\_id at any time):**

{
"run\_id": "uuid", "status": "completed" | "awaiting\_approval" | "failed",
"current\_step": "finalize", "final\_disposition": "approved" | null
}

## 6. Interoperability Requirements

- **The streamed event \`type\` vocabulary must be fixed and documented** (\`step\_started\`, \`step\_completed\`, \`awaiting\_approval\`, \`completed\`, \`error\`) so any consuming frontend can render progress deterministically without special-casing.
- **The \`resume\` decision payload must always follow the approve/reject/edit/escalate convention** with \`edited\_input\` present only when \`action: edit\` — this shape must remain stable regardless of which workflow step is paused.
- **A paused run must survive backend restarts/redeploys** — state must be durable (Foundry Agent Service's managed state persistence is expected to satisfy this by design).
- **Approval timeout behavior must be configurable**: what happens if no reviewer decision arrives within a configurable window — auto-reject, auto-approve, or escalate — must be settable per workflow, not hardcoded.

## 8. Delivery Target

# Component 9: Prompting, Observation & Action Plan Generation (Structured LLM Reasoning) ()

## 1. Overview

A structured-generation service that takes a task description plus retrieved source chunks and returns a schema-conformant compliance summary, gap list, or action plan, with every claim validated against the supplied source material and explicitly labelled as decision-support information.

## 2. Azure Services

- **Azure OpenAI — GPT-4o / GPT-4.1** via the Foundry Model Catalog, called with structured-output / JSON-schema mode enabled so responses are guaranteed valid JSON conforming to a caller-supplied schema.
- **Azure AI Foundry Groundedness Detection API** to validate that a generated action plan or gap list is actually grounded in the retrieved source chunks rather than containing unsupported claims.

## 3. Functional Requirements

1. **Accept a structured request** containing: the question/task (e.g. "review this policy against Standard X"), the retrieved context chunks (with source references), and a target output schema name (e.g. \`compliance\_report\`, \`gap\_list\`, \`action\_plan\`).
2. **Generate output that strictly conforms to the requested JSON schema** every time — using the model provider's native structured-output enforcement, not just prompt instructions — so the caller never needs defensive JSON-parsing/repair logic.
3. **Support at minimum these three output schema types:** (a) \`review\_summary\` — a concise natural-language summary plus a \`compliant: bool\` verdict; (b) \`gap\_list\` — an array of \`{gap\_description, missing\_or\_outdated, related\_source\_chunk\_ids}\`; (c) \`action\_plan\` — an array of \`{recommended\_action, priority, rationale, supporting\_source\_chunk\_ids}\`. All three must also return \`suggested\_reviewer\_questions: \[string\]\`.
4. **Run every generated output through groundedness validation** before returning it — each claim/gap/action must be traceable to at least one of the supplied source chunks; flag (do not silently drop) any claim that fails groundedness, returning it with \`grounded: false\` so the caller can decide whether to surface, hide, or re-generate.
5. **Always include a fixed disclaimer field** (\`decision\_support\_disclaimer\`) in every response, making explicit that the output is decision-support information requiring human review, not an automated final decision — this must be a first-class structured field, not text buried inside prose.

## 4. Input Contract

**Generate structured output:**

{
"task": "review this policy section against Standard X clause 4.2",
"schema": "review\_summary" | "gap\_list" | "action\_plan",
"retrieved\_chunks": \[
{
"chunk\_id": "uuid",
"chunk\_text": "...",
"source": {"filename": "policy\_v3.pdf", "page\_num": 4}
}
\],
"model\_preference": "gpt-4o" | "gpt-4.1", "temperature": 0.2
}

## 5. Output Contract

**Example — gap\_list schema:**

{
"schema": "gap\_list",
"gaps": \[
{
"gap\_description": "No mention of annual review cadence",
"missing\_or\_outdated": "missing", "related\_source\_chunk\_ids": \["uuid1"\],
"grounded": true
}
\],
"suggested\_reviewer\_questions": \["Does the policy specify a review cycle?"\],
"decision\_support\_disclaimer": "This output is AI-generated decision-support information and requires human reviewer approval before any action is taken.",
"ungrounded\_claims\_removed\_or\_flagged": 0
}

## 6. Interoperability Requirements

- **Output must always match the exact schema requested** — the component must never return a shape other than the one named in \`schema\`, and must never omit the mandatory \`decision\_support\_disclaimer\` or \`suggested\_reviewer\_questions\` fields regardless of schema type.

## 8. Delivery Target
