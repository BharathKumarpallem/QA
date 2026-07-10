from unittest.mock import MagicMock, patch
import pytest
from app.services.pdf_processor import PDFProcessor
from app.core.exceptions import InvalidDocumentException, ScannedDocumentException


def test_compute_sha256() -> None:
    processor = PDFProcessor()
    data = b"hello world"
    # Known SHA-256 for 'hello world'
    expected_hash = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert processor.compute_sha256(data) == expected_hash


@patch("pdfplumber.open")
def test_process_pdf_success(mock_open) -> None:
    processor = PDFProcessor()

    # Mock PDF Page 1 & 2
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "This is some readable text on page one. It needs to be long enough so it is not marked as scanned."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = (
        "This is the second page of our document. It contains valid English sentences."
    )

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page1, mock_page2]
    mock_open.return_value = mock_pdf

    chunks, total_pages = processor.process_pdf(b"mock_data", "test.pdf")

    assert total_pages == 2
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert "page one" in chunks[0]["text"]
    assert chunks[1]["page_number"] == 2
    assert "second page" in chunks[1]["text"]


@patch("pdfplumber.open")
def test_scanned_pdf_throws_exception(mock_open) -> None:
    processor = PDFProcessor()

    # Mock a page with nearly zero text (scanned page)
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Low"

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_open.return_value = mock_pdf

    with pytest.raises(ScannedDocumentException) as exc_info:
        processor.process_pdf(b"scanned_data", "scanned.pdf")

    assert "Scanned PDF detected" in str(exc_info.value)


@patch("pdfplumber.open")
def test_empty_pdf_throws_exception(mock_open) -> None:
    processor = PDFProcessor()

    # PDF with no pages
    mock_pdf = MagicMock()
    mock_pdf.pages = []
    mock_open.return_value = mock_pdf

    with pytest.raises(InvalidDocumentException) as exc_info:
        processor.process_pdf(b"empty_data", "empty.pdf")

    assert "does not contain any pages" in str(exc_info.value)


def test_chunk_text_long_page() -> None:
    processor = PDFProcessor()
    # Create text that exceeds settings.CHUNK_SIZE tokens (approx 500 words)
    long_text = "word " * 600
    chunks = processor._chunk_text(long_text, page_number=5)

    # Should have split into at least 2 chunks
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk["page_number"] == 5
        assert len(chunk["text"]) > 0
        assert chunk["token_count"] <= 500
