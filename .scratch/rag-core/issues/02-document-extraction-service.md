# 02: Document Extraction Service (Component 1)

**What to build:** An HTTP extraction service endpoint (`POST /extract`) accepting PDF, DOCX, and scanned documents, using Docling to produce 1-based page-indexed plain text, structured tables formatted with `[Table N]` and pipe delimiters, and document metadata, with resilient structured error handling.

**Blocked by:** 01: Project Scaffolding & Container Environment

**Status:** ready-for-agent

- [ ] `POST /extract` accepts multipart document uploads (`file`, `filename`, optional `force_ocr`) and JSON payloads (`blob_url`, `filename`, `force_ocr`).
- [ ] Text extraction attributes every text span to a 1-based integer page number (`page_num >= 1`).
- [ ] Detected tables are rendered using the `[Table N]` marker convention followed by pipe-delimited text rows and structured row/column arrays.
- [ ] Document metadata block extracts title, author, creation date, and revision where available, returning an empty object `{}` when missing.
- [ ] Scanned pages without native text trigger OCR fallback automatically or when `force_ocr=true`.
- [ ] Malformed or corrupt files return HTTP 400/422 with structured error JSON (`UNSUPPORTED_FORMAT`, `CORRUPT_FILE`, `OCR_TIMEOUT`) instead of uncaught 500 exceptions.
- [ ] Automated contract tests verify responses match the Component 1 output schema.
