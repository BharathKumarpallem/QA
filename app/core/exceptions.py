class ServiceException(Exception):
    """
    Base exception for application business errors.
    """

    status_code: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str = None, status_code: int = None):
        if detail:
            self.detail = detail
        if status_code:
            self.status_code = status_code
        super().__init__(self.detail)


class DuplicateDocumentException(ServiceException):
    status_code: int = 409
    detail: str = "This document has already been uploaded."


class InvalidDocumentException(ServiceException):
    status_code: int = 400
    detail: str = "The uploaded file is not a valid PDF or is corrupted."


class ScannedDocumentException(ServiceException):
    status_code: int = 422
    detail: str = "Scanned PDF detected. OCR is required."


class DocumentNotFoundException(ServiceException):
    status_code: int = 404
    detail: str = "Document not found."


class LLMServiceException(ServiceException):
    status_code: int = 502
    detail: str = "Failed to communicate with LLM provider."


class VectorStoreException(ServiceException):
    status_code: int = 500
    detail: str = "Failed to interact with the vector store."
