import hashlib
from io import BytesIO
from typing import List, Dict, Any, Tuple
import pdfplumber
import tiktoken
from app.core.config import settings
from app.core.exceptions import InvalidDocumentException, ScannedDocumentException
from loguru import logger


class PDFProcessor:
    """
    Service responsible for loading PDF files, verifying their validity,
    detecting scanned contents, and chunking text with page awareness and token budgets.
    """

    def __init__(self) -> None:
        # text-embedding-3-small uses cl100k_base tokenizer encoding
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def compute_sha256(self, file_content: bytes) -> str:
        """
        Computes the SHA-256 hash of the uploaded file bytes.
        """
        return hashlib.sha256(file_content).hexdigest()

    def process_pdf(
        self, file_content: bytes, filename: str
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Extracts page-by-page text from PDF, validates scanned content, and chunks the text.

        Returns:
            Tuple[List[Dict], int]: List of chunks and the total page count.
        """
        logger.info(f"Processing PDF '{filename}' (Size: {len(file_content)} bytes)")

        try:
            pdf = pdfplumber.open(BytesIO(file_content))
        except Exception as e:
            logger.error(f"Failed to open PDF '{filename}': {str(e)}")
            raise InvalidDocumentException(
                f"The uploaded file is not a valid PDF or is corrupted: {str(e)}"
            )

        pages_data: List[Tuple[int, str]] = []
        total_chars = 0
        total_pages = len(pdf.pages)

        if total_pages == 0:
            pdf.close()
            raise InvalidDocumentException(
                "The PDF document does not contain any pages."
            )

        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            try:
                text = page.extract_text() or ""
                text = text.strip()
                total_chars += len(text)
                pages_data.append((page_num, text))
            except Exception as e:
                logger.warning(
                    f"Failed to extract text from page {page_num} of '{filename}': {str(e)}"
                )
                pages_data.append((page_num, ""))

        pdf.close()

        # Scanned PDF detection
        avg_chars_per_page = total_chars / total_pages
        logger.info(
            f"PDF processed | Pages: {total_pages} | Avg characters/page: {avg_chars_per_page:.2f}"
        )

        # If the average characters per page is extremely low, it's likely a scanned PDF
        if avg_chars_per_page < 50:
            logger.error(
                f"Scanned PDF detected for '{filename}' | Avg characters/page: {avg_chars_per_page:.2f}"
            )
            raise ScannedDocumentException(
                f"Scanned PDF detected. Average characters per page is {avg_chars_per_page:.1f} "
                "(minimum required is 50.0). OCR is required."
            )

        chunks: List[Dict[str, Any]] = []
        for page_num, text in pages_data:
            if not text:
                continue

            page_chunks = self._chunk_text(text, page_num)
            chunks.extend(page_chunks)

        logger.info(
            f"Ingested '{filename}' | Ingested {len(chunks)} chunks across {total_pages} pages."
        )
        return chunks, total_pages

    def _chunk_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Splits text into chunks under CHUNK_SIZE tokens with CHUNK_OVERLAP token overlap.
        """
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP

        # If the text fits in one chunk, return immediately
        if total_tokens <= chunk_size:
            return [
                {"text": text, "page_number": page_number, "token_count": total_tokens}
            ]

        chunks: List[Dict[str, Any]] = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(
                {
                    "text": chunk_text,
                    "page_number": page_number,
                    "token_count": len(chunk_tokens),
                }
            )

            # Step size must be positive to prevent infinite loop
            step = chunk_size - chunk_overlap
            if step <= 0:
                logger.warning(
                    f"CHUNK_OVERLAP ({chunk_overlap}) is greater than or equal to CHUNK_SIZE ({chunk_size}). "
                    "Defaulting overlap to 0 for this step to make progress."
                )
                step = chunk_size

            start += step

        return chunks
