import logging

import httpx
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from src.services.extractor import (
    CorruptFileError,
    DocumentExtractor,
    ExtractionResult,
    UnsupportedFormatError,
    get_extractor,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["extract"])


class ExtractBlobRequest(BaseModel):
    blob_url: str
    filename: str
    force_ocr: bool = False


@router.post("/extract", response_model=ExtractionResult)
async def extract_document(
    request: Request,
    extractor: DocumentExtractor = Depends(get_extractor),
) -> ExtractionResult:
    """Extract document text, tables, and metadata from uploaded file or blob reference."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            body = await request.json()
            payload = ExtractBlobRequest(**body)
        except Exception as e:
            raise UnsupportedFormatError(f"Invalid JSON payload: {e}") from e

        # Download file content from blob_url
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.get(str(payload.blob_url))
                res.raise_for_status()
                file_bytes = res.content
        except Exception as e:
            raise CorruptFileError(f"Failed to retrieve blob from '{payload.blob_url}': {e}") from e

        filename = payload.filename
        force_ocr = payload.force_ocr

    elif "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise CorruptFileError("No file uploaded in multipart form data")

        filename = form.get("filename")
        if not filename and hasattr(file, "filename"):
            filename = file.filename
        if not filename:
            raise UnsupportedFormatError("Missing filename in request")

        force_ocr_val = form.get("force_ocr", False)
        if isinstance(force_ocr_val, str):
            force_ocr = force_ocr_val.strip().lower() in ("true", "1", "yes")
        else:
            force_ocr = bool(force_ocr_val)

        if hasattr(file, "read"):
            file_bytes = await file.read()
        elif isinstance(file, bytes):
            file_bytes = file
        else:
            raise CorruptFileError("Invalid uploaded file payload")

    else:
        raise UnsupportedFormatError(
            f"Unsupported Content-Type '{content_type}'. "
            "Must be multipart/form-data or application/json."
        )

    return await extractor.extract(
        file_bytes=file_bytes, filename=str(filename), force_ocr=force_ocr
    )
