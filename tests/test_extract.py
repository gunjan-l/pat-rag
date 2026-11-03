from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.services.extractor import (
    CorruptFileError,
    DocumentExtractor,
    ExtractionResult,
    OCRTimeoutError,
    PageData,
    TableData,
    UnsupportedFormatError,
    get_extractor,
)


@pytest.fixture
def mock_extraction_result() -> ExtractionResult:
    return ExtractionResult(
        filename="policy_v3.pdf",
        extraction_method="native_text",
        document_metadata={
            "title": "Accreditation Policy 2026",
            "author": "Jane Doe",
            "created_date": "2025-01-10",
            "revision": "v3",
            "revision_date": "2026-03-01",
        },
        pages=[
            PageData(
                page_num=1,
                text="This is page one text.\n\n[Table 0]\nHeader1 | Header2\nVal1 | Val2",
                tables=[
                    TableData(
                        table_index=0,
                        rows=[["Header1", "Header2"], ["Val1", "Val2"]],
                    )
                ],
                confidence=0.95,
            ),
            PageData(
                page_num=2,
                text="This is page two text.",
                tables=[],
                confidence=0.98,
            ),
        ],
        warnings=[],
    )


@pytest.mark.asyncio
async def test_extract_multipart_pdf_contract(mock_extraction_result):
    """Test POST /extract with multipart/form-data upload."""
    mock_extractor = AsyncMock(spec=DocumentExtractor)
    mock_extractor.extract = AsyncMock(return_value=mock_extraction_result)

    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    try:
        pdf_bytes = b"%PDF-1.5 mock pdf content"
        files = {"file": ("policy_v3.pdf", pdf_bytes, "application/pdf")}
        data = {"filename": "policy_v3.pdf", "force_ocr": "false"}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/extract", files=files, data=data)

        assert response.status_code == 200
        res = response.json()

        # Contract checks
        assert res["filename"] == "policy_v3.pdf"
        assert res["extraction_method"] in ["native_text", "ocr"]
        assert isinstance(res["document_metadata"], dict)
        assert res["document_metadata"]["title"] == "Accreditation Policy 2026"
        assert res["document_metadata"]["author"] == "Jane Doe"
        assert res["document_metadata"]["revision"] == "v3"

        assert isinstance(res["pages"], list)
        assert len(res["pages"]) == 2

        # 1-based page numbers
        for idx, page in enumerate(res["pages"], start=1):
            assert page["page_num"] == idx
            assert page["page_num"] >= 1
            assert isinstance(page["text"], str)
            assert isinstance(page["tables"], list)
            assert 0.0 <= page["confidence"] <= 1.0

        # Table formatting check
        page1 = res["pages"][0]
        assert "[Table 0]" in page1["text"]
        assert "Header1 | Header2" in page1["text"]
        assert len(page1["tables"]) == 1
        assert page1["tables"][0]["table_index"] == 0
        assert page1["tables"][0]["rows"] == [["Header1", "Header2"], ["Val1", "Val2"]]

        assert isinstance(res["warnings"], list)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extract_json_blob_url(mock_extraction_result):
    """Test POST /extract with JSON payload referencing a blob_url."""
    mock_extractor = AsyncMock(spec=DocumentExtractor)
    mock_extractor.extract = AsyncMock(return_value=mock_extraction_result)

    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    try:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"%PDF-1.5 mock remote pdf"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            payload = {
                "blob_url": "https://storage.example.com/container/policy_v3.pdf?sas=token",
                "filename": "policy_v3.pdf",
                "force_ocr": False,
            }
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/extract", json=payload)

        assert response.status_code == 200
        res = response.json()
        assert res["filename"] == "policy_v3.pdf"
        assert len(res["pages"]) == 2
        mock_extractor.extract.assert_awaited_once()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extract_force_ocr(mock_extraction_result):
    """Test force_ocr parameter passes through and forces OCR extraction method."""
    ocr_result = ExtractionResult(
        filename="scanned.pdf",
        extraction_method="ocr",
        document_metadata={},
        pages=[
            PageData(
                page_num=1,
                text="OCR text content",
                tables=[],
                confidence=0.88,
            )
        ],
        warnings=["page 1 processed with OCR"],
    )

    mock_extractor = AsyncMock(spec=DocumentExtractor)
    mock_extractor.extract = AsyncMock(return_value=ocr_result)

    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    try:
        files = {"file": ("scanned.pdf", b"%PDF-1.5 scanned", "application/pdf")}
        data = {"filename": "scanned.pdf", "force_ocr": "true"}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/extract", files=files, data=data)

        assert response.status_code == 200
        res = response.json()
        assert res["extraction_method"] == "ocr"
        assert res["document_metadata"] == {}
        assert res["warnings"] == ["page 1 processed with OCR"]

        # Check call arguments to mock_extractor
        _, kwargs = mock_extractor.extract.call_args
        assert kwargs.get("force_ocr") is True or mock_extractor.extract.call_args[0][2] is True
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extract_empty_metadata_returns_empty_dict():
    """Verify document_metadata returns an empty object {} when no metadata is found."""
    empty_meta_result = ExtractionResult(
        filename="unnamed.pdf",
        extraction_method="native_text",
        document_metadata={},
        pages=[
            PageData(
                page_num=1,
                text="Some text",
                tables=[],
                confidence=0.99,
            )
        ],
        warnings=[],
    )

    mock_extractor = AsyncMock(spec=DocumentExtractor)
    mock_extractor.extract = AsyncMock(return_value=empty_meta_result)

    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    try:
        files = {"file": ("unnamed.pdf", b"%PDF-1.5 content", "application/pdf")}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/extract", files=files)

        assert response.status_code == 200
        res = response.json()
        assert "document_metadata" in res
        assert res["document_metadata"] == {}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extract_unsupported_format_error():
    """Verify HTTP 400 with UNSUPPORTED_FORMAT for unsupported file extensions."""
    files = {"file": ("script.exe", b"binary executable content", "application/octet-stream")}
    data = {"filename": "script.exe"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/extract", files=files, data=data)

    assert response.status_code == 400
    res = response.json()
    assert res["error_code"] == "UNSUPPORTED_FORMAT"
    assert "not supported" in res["message"].lower()


@pytest.mark.asyncio
async def test_extract_corrupt_file_error():
    """Verify HTTP 422 with CORRUPT_FILE for empty or corrupt file content."""
    mock_extractor = AsyncMock(spec=DocumentExtractor)
    mock_extractor.extract = AsyncMock(side_effect=CorruptFileError("File is empty or corrupt"))

    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    try:
        files = {"file": ("corrupt.pdf", b"", "application/pdf")}
        data = {"filename": "corrupt.pdf"}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/extract", files=files, data=data)

        assert response.status_code == 422
        res = response.json()
        assert res["error_code"] == "CORRUPT_FILE"
        assert "corrupt" in res["message"].lower() or "empty" in res["message"].lower()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extract_ocr_timeout_error():
    """Verify HTTP 504 with OCR_TIMEOUT when document extraction exceeds timeout."""
    mock_extractor = AsyncMock(spec=DocumentExtractor)
    mock_extractor.extract = AsyncMock(side_effect=OCRTimeoutError("Extraction exceeded timeout"))

    app.dependency_overrides[get_extractor] = lambda: mock_extractor
    try:
        files = {"file": ("huge.pdf", b"%PDF-1.5 large document", "application/pdf")}
        data = {"filename": "huge.pdf"}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/extract", files=files, data=data)

        assert response.status_code in [504, 408]
        res = response.json()
        assert res["error_code"] == "OCR_TIMEOUT"
        assert "timeout" in res["message"].lower()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_extract_blob_url_download_failure():
    """Verify HTTP 422 when downloading blob_url fails."""
    with patch("httpx.AsyncClient.get", side_effect=Exception("Failed to fetch blob")):
        payload = {
            "blob_url": "https://storage.example.com/missing.pdf",
            "filename": "missing.pdf",
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/extract", json=payload)

    assert response.status_code in [400, 422]
    res = response.json()
    assert res["error_code"] in ["CORRUPT_FILE", "UNSUPPORTED_FORMAT"]


# Unit tests for DocumentExtractor logic
def test_extractor_format_table():
    extractor = DocumentExtractor()
    rows = [["Name", "Role"], ["Alice", "Admin"], ["Bob", "User"]]
    formatted = extractor.format_table_text(0, rows)
    assert "[Table 0]" in formatted
    assert "Name | Role" in formatted
    assert "Alice | Admin" in formatted
    assert "Bob | User" in formatted


def test_extractor_supported_extensions():
    extractor = DocumentExtractor()
    assert extractor.is_supported_filename("policy.pdf") is True
    assert extractor.is_supported_filename("manual.docx") is True
    assert extractor.is_supported_filename("scan.png") is True
    assert extractor.is_supported_filename("scan.jpeg") is True
    assert extractor.is_supported_filename("scan.tiff") is True
    assert extractor.is_supported_filename("bad.exe") is False
    assert extractor.is_supported_filename("data.csv") is False


@pytest.mark.asyncio
async def test_extractor_empty_bytes_raises_corrupt():
    extractor = DocumentExtractor()
    with pytest.raises(CorruptFileError):
        await extractor.extract(b"", "empty.pdf")


@pytest.mark.asyncio
async def test_extractor_unsupported_raises_error():
    extractor = DocumentExtractor()
    with pytest.raises(UnsupportedFormatError):
        await extractor.extract(b"data", "script.sh")
