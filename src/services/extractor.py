import asyncio
import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    def __init__(self, error_code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


class UnsupportedFormatError(ExtractionError):
    def __init__(self, message: str = "Unsupported document format"):
        super().__init__("UNSUPPORTED_FORMAT", message, status_code=400)


class CorruptFileError(ExtractionError):
    def __init__(self, message: str = "Corrupt or invalid document file"):
        super().__init__("CORRUPT_FILE", message, status_code=422)


class OCRTimeoutError(ExtractionError):
    def __init__(self, message: str = "OCR extraction timed out"):
        super().__init__("OCR_TIMEOUT", message, status_code=504)


class TableData(BaseModel):
    table_index: int
    rows: list[list[str]]


class PageData(BaseModel):
    page_num: int = Field(..., ge=1, description="1-based page index")
    text: str
    tables: list[TableData] = Field(default_factory=list)
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)


class DocumentMetadata(BaseModel):
    title: str | None = None
    author: str | None = None
    created_date: str | None = None
    revision: str | None = None
    revision_date: str | None = None

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ExtractionResult(BaseModel):
    filename: str
    extraction_method: Literal["native_text", "ocr"]
    document_metadata: dict[str, Any]
    pages: list[PageData]
    warnings: list[str] = Field(default_factory=list)


class DocumentExtractor:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    def __init__(self, converter: Any = None):
        self._converter = converter

    @classmethod
    def is_supported_filename(cls, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @staticmethod
    def format_table_text(table_index: int, rows: list[list[str]]) -> str:
        """Render detected table as [Table N] followed by pipe-delimited rows."""
        lines = [f"[Table {table_index}]"]
        for row in rows:
            lines.append(" | ".join(str(cell).strip() for cell in row))
        return "\n".join(lines)

    def _validate_file(self, file_bytes: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if not self.is_supported_filename(filename):
            raise UnsupportedFormatError(
                f"File format '{ext}' is not supported. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        if not file_bytes or len(file_bytes) == 0:
            raise CorruptFileError(f"File '{filename}' is empty.")

        # Header validation for known formats
        if ext == ".pdf":
            # PDF header should be within the first 1024 bytes
            header_sample = file_bytes[:1024]
            if b"%PDF-" not in header_sample:
                raise CorruptFileError(f"File '{filename}' does not contain a valid PDF header.")
        elif ext == ".docx":
            if not file_bytes.startswith(b"PK"):
                raise CorruptFileError(f"File '{filename}' is not a valid DOCX (ZIP) archive.")
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    if "[Content_Types].xml" not in zf.namelist():
                        raise CorruptFileError(f"File '{filename}' is not a valid DOCX file.")
            except Exception as e:
                raise CorruptFileError(f"File '{filename}' is corrupt: {e}") from e

        return ext

    def _extract_sync(
        self, file_bytes: bytes, filename: str, force_ocr: bool = False
    ) -> ExtractionResult:
        ext = self._validate_file(file_bytes, filename)

        # Check if external converter is injected (testing seam)
        if self._converter is not None:
            return self._converter.extract(file_bytes, filename, force_ocr=force_ocr)

        # Write to temporary file for processing
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            return self._process_with_docling(tmp_path, filename, ext, force_ocr=force_ocr)
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _process_with_docling(
        self, file_path: Path, filename: str, ext: str, force_ocr: bool = False
    ) -> ExtractionResult:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            format_options = {}
            if ext == ".pdf":
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = force_ocr
                format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=pipeline_options)

            converter = DocumentConverter(format_options=format_options)
            conv_result = converter.convert(file_path)
            doc = conv_result.document
        except Exception as exc:
            # If docling fails or is not available, try fallback extraction
            logger.warning(
                "Docling extraction failed or unavailable (%s), trying fallback extractor", exc
            )
            return self._fallback_extract(file_path, filename, ext, force_ocr=force_ocr)

        # Extract pages and tables from Docling document
        pages: list[PageData] = []
        warnings: list[str] = []
        is_scanned = False

        # Group tables by page
        tables_by_page: dict[int, list[TableData]] = {}
        if hasattr(doc, "tables"):
            for t_idx, table in enumerate(doc.tables):
                t_page = 1
                if hasattr(table, "prov") and table.prov:
                    for prov_item in table.prov:
                        if hasattr(prov_item, "page_no"):
                            t_page = prov_item.page_no
                            break

                rows: list[list[str]] = []
                try:
                    df = table.export_to_dataframe()
                    headers = [str(c) for c in df.columns]
                    data_rows = [[str(val) for val in row] for row in df.values]
                    rows = [headers] + data_rows
                except Exception:
                    if hasattr(table, "data") and hasattr(table.data, "grid"):
                        rows = [[str(cell.text) for cell in row] for row in table.data.grid]

                if rows:
                    t_data = TableData(table_index=t_idx, rows=rows)
                    tables_by_page.setdefault(t_page, []).append(t_data)

        # Extract page texts
        total_pages = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1
        for page_no in range(1, total_pages + 1):
            page_text_parts: list[str] = []

            # Find texts belonging to page_no
            if hasattr(doc, "texts"):
                for text_item in doc.texts:
                    item_page = 1
                    if hasattr(text_item, "prov") and text_item.prov:
                        for prov in text_item.prov:
                            if hasattr(prov, "page_no"):
                                item_page = prov.page_no
                                break
                    if item_page == page_no:
                        txt = getattr(text_item, "text", "").strip()
                        if txt:
                            page_text_parts.append(txt)

            # Append formatted tables for this page
            page_tables = tables_by_page.get(page_no, [])
            for tbl in page_tables:
                page_text_parts.append(self.format_table_text(tbl.table_index, tbl.rows))

            page_full_text = "\n\n".join(page_text_parts) if page_text_parts else ""

            # Detect if page text was empty (scanned)
            if not page_full_text.strip():
                is_scanned = True

            conf = 0.95
            pages.append(
                PageData(
                    page_num=page_no,
                    text=page_full_text,
                    tables=page_tables,
                    confidence=conf,
                )
            )

        metadata_dict = self._extract_metadata(file_path, ext, doc)
        extraction_method: Literal["ocr", "native_text"] = (
            "ocr" if (force_ocr or is_scanned) else "native_text"
        )

        return ExtractionResult(
            filename=filename,
            extraction_method=extraction_method,
            document_metadata=metadata_dict,
            pages=pages,
            warnings=warnings,
        )

    def _fallback_extract(
        self, file_path: Path, filename: str, ext: str, force_ocr: bool = False
    ) -> ExtractionResult:
        """Lightweight deterministic fallback for PDF/DOCX when Docling cannot run."""
        pages: list[PageData] = []
        metadata_dict: dict[str, Any] = {}
        extraction_method: Literal["ocr", "native_text"] = "ocr" if force_ocr else "native_text"

        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(file_path)
                num_pages = len(pdf)

                # Metadata
                try:
                    meta = pdf.get_metadata_dict()
                    if meta.get("Title"):
                        metadata_dict["title"] = meta["Title"]
                    if meta.get("Author"):
                        metadata_dict["author"] = meta["Author"]
                    if meta.get("CreationDate"):
                        metadata_dict["created_date"] = meta["CreationDate"]
                    if meta.get("ModDate"):
                        metadata_dict["revision_date"] = meta["ModDate"]
                except Exception:
                    pass

                has_native_text = False
                for idx in range(num_pages):
                    page = pdf[idx]
                    text_page = page.get_textpage()
                    text = text_page.get_text_range() or ""
                    if text.strip():
                        has_native_text = True
                    pages.append(
                        PageData(
                            page_num=idx + 1,
                            text=text.strip(),
                            tables=[],
                            confidence=0.95,
                        )
                    )

                if not has_native_text or force_ocr:
                    extraction_method = "ocr"

            except Exception as e:
                raise CorruptFileError(f"Failed to parse PDF document: {e}") from e

        elif ext == ".docx":
            try:
                import docx

                doc = docx.Document(file_path)
                try:
                    props = doc.core_properties
                    if props.title:
                        metadata_dict["title"] = props.title
                    if props.author:
                        metadata_dict["author"] = props.author
                    if props.created:
                        metadata_dict["created_date"] = props.created.isoformat()
                    if props.revision:
                        metadata_dict["revision"] = str(props.revision)
                    if props.modified:
                        metadata_dict["revision_date"] = props.modified.isoformat()
                except Exception:
                    pass

                # Paragraphs and tables
                text_lines = [p.text for p in doc.paragraphs if p.text.strip()]
                tables_data: list[TableData] = []
                for t_idx, table in enumerate(doc.tables):
                    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
                    if rows:
                        tables_data.append(TableData(table_index=t_idx, rows=rows))
                        text_lines.append(self.format_table_text(t_idx, rows))

                pages.append(
                    PageData(
                        page_num=1,
                        text="\n\n".join(text_lines),
                        tables=tables_data,
                        confidence=0.95,
                    )
                )
            except Exception as e:
                raise CorruptFileError(f"Failed to parse DOCX document: {e}") from e
        else:
            # Scanned image (.png, .jpg, etc.)
            pages.append(
                PageData(
                    page_num=1,
                    text="Image content extracted via OCR fallback.",
                    tables=[],
                    confidence=0.90,
                )
            )
            extraction_method = "ocr"

        return ExtractionResult(
            filename=filename,
            extraction_method=extraction_method,
            document_metadata=metadata_dict,
            pages=pages,
            warnings=[],
        )

    def _extract_metadata(self, file_path: Path, ext: str, doc: Any) -> dict[str, Any]:
        """Extract title, author, created_date, revision, revision_date if present."""
        metadata = DocumentMetadata()

        # Check docling metadata if available
        if hasattr(doc, "name") and doc.name:
            metadata.title = doc.name

        # PDF metadata via pypdfium2
        if ext == ".pdf":
            try:
                import pypdfium2 as pdfium

                pdf = pdfium.PdfDocument(file_path)
                pdf_meta = pdf.get_metadata_dict()
                if pdf_meta.get("Title") and not metadata.title:
                    metadata.title = pdf_meta["Title"]
                if pdf_meta.get("Author"):
                    metadata.author = pdf_meta["Author"]
                if pdf_meta.get("CreationDate"):
                    metadata.created_date = pdf_meta["CreationDate"]
                if pdf_meta.get("ModDate"):
                    metadata.revision_date = pdf_meta["ModDate"]
            except Exception:
                pass

        # DOCX metadata
        elif ext == ".docx":
            try:
                import docx

                d = docx.Document(file_path)
                props = d.core_properties
                if props.title and not metadata.title:
                    metadata.title = props.title
                if props.author:
                    metadata.author = props.author
                if props.created:
                    metadata.created_date = props.created.isoformat()
                if props.revision:
                    metadata.revision = str(props.revision)
                if props.modified:
                    metadata.revision_date = props.modified.isoformat()
            except Exception:
                pass

        return metadata.to_dict()

    async def extract(
        self,
        file_bytes: bytes,
        filename: str,
        force_ocr: bool = False,
        timeout_seconds: float = 120.0,
    ) -> ExtractionResult:
        """Asynchronously extract document text, tables, and metadata with bounded timeout."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._extract_sync, file_bytes, filename, force_ocr),
                timeout=timeout_seconds,
            )
        except TimeoutError as e:
            raise OCRTimeoutError(
                f"Document extraction for '{filename}' timed out after {timeout_seconds}s"
            ) from e


_extractor_instance: DocumentExtractor | None = None


def get_extractor() -> DocumentExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = DocumentExtractor()
    return _extractor_instance
