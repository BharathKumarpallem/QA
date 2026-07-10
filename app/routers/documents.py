import uuid
from fastapi import APIRouter, File, UploadFile, Depends
from app.core.config import settings
from app.core.exceptions import DuplicateDocumentException, InvalidDocumentException
from app.models.schemas import DocumentUploadResponse
from app.services.pdf_processor import PDFProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from loguru import logger

router = APIRouter(prefix="/documents", tags=["Document Management"])


# Dependency Providers
def get_pdf_processor() -> PDFProcessor:
    return PDFProcessor()


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload and Process PDF Document",
)
async def upload_document(
    file: UploadFile = File(...),
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """
    Ingests, validates, chunks, and creates vectors for an uploaded PDF document.
    """
    logger.info(f"Received upload request for file: '{file.filename}'")

    # 1. MIME Validation
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(
        ".pdf"
    ):
        logger.error(
            f"Upload failed: File '{file.filename}' is not a PDF (Content-Type: '{file.content_type}')"
        )
        raise InvalidDocumentException(
            detail=f"Unsupported file type '{file.content_type}'. Only PDF files (.pdf) are allowed.",
            status_code=400,
        )

    # Read content bytes
    content_bytes = await file.read()
    file_size_mb = len(content_bytes) / (1024 * 1024)

    # 2. File Size Validation
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        logger.error(
            f"Upload failed: File size ({file_size_mb:.2f}MB) exceeds limit ({settings.MAX_UPLOAD_SIZE_MB}MB)"
        )
        raise InvalidDocumentException(
            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB}MB. Uploaded size: {file_size_mb:.2f}MB.",
            status_code=413,
        )

    # 3. Compute Hash for Duplicate Detection
    sha256 = pdf_processor.compute_sha256(content_bytes)
    duplicate_info = vector_store.check_duplicate_by_hash(sha256)
    if duplicate_info:
        logger.warning(
            f"Duplicate file uploaded. SHA-256: {sha256} | "
            f"Existing document ID: {duplicate_info['document_id']}"
        )
        raise DuplicateDocumentException(
            detail=(
                f"Duplicate document detected. This file has already been processed "
                f"under Document ID: {duplicate_info['document_id']}."
            )
        )

    # 4. Extract Text & Chunk PDF
    chunks, total_pages = pdf_processor.process_pdf(content_bytes, file.filename)

    if not chunks:
        logger.error("Processing PDF generated 0 text chunks.")
        raise InvalidDocumentException(
            "No indexable text could be extracted from the PDF."
        )

    # 5. Generate Vector Embeddings
    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks of '{file.filename}'")
    embeddings = await embedding_service.get_embeddings(texts)

    # 6. Save vectors and chunk metadata to ChromaDB
    document_id = str(uuid.uuid4())
    vector_store.add_document_chunks(
        document_id=document_id,
        filename=file.filename,
        sha256=sha256,
        chunks=chunks,
        embeddings=embeddings,
        total_pages=total_pages,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        sha256=sha256,
        page_count=total_pages,
        chunk_count=len(chunks),
        status="processed",
    )


@router.delete("/{document_id}", summary="Delete Ingested Document vectors")
async def delete_document(
    document_id: str, vector_store: VectorStoreService = Depends(get_vector_store)
) -> dict:
    """
    Deletes all vectors and chunk records associated with the specified Document UUID.
    """
    logger.info(f"Received request to delete document: '{document_id}'")
    vector_store.delete_document(document_id)
    return {
        "detail": f"Document ID '{document_id}' and all associated vector chunks deleted successfully."
    }
