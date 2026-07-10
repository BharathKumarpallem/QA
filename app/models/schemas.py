from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# Health and Version schemas
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    database: str = Field(..., example="connected")
    timestamp: datetime


class VersionResponse(BaseModel):
    name: str = Field(..., example="AI Document Q&A Service")
    version: str = Field(..., example="1.0.0")
    environment: str = Field(..., example="development")


# Document upload schema
class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique UUID assigned to the document")
    filename: str = Field(..., description="Name of the uploaded PDF file")
    sha256: str = Field(
        ..., description="SHA-256 hash of the file content for duplicate checking"
    )
    page_count: int = Field(..., description="Total pages processed in the document")
    chunk_count: int = Field(..., description="Total semantic chunks generated")
    status: str = Field("processed", description="Status of the ingestion pipeline")


# Semantic Search schemas
class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="Semantic query to search chunks for"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Filter search by specific document IDs. If empty, searches all documents.",
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")


class SearchChunkResult(BaseModel):
    text: str = Field(..., description="Text content of the retrieved chunk")
    document_id: str = Field(..., description="UUID of the parent document")
    document_name: str = Field(..., description="Filename of the parent document")
    page_number: int = Field(
        ..., description="Page number of the chunk in the original PDF"
    )
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class SearchResponse(BaseModel):
    results: List[SearchChunkResult]


# Question Answering schemas
class QARequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="Question to ask the AI assistant"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Filter sources by specific document IDs. If empty, searches across all documents.",
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional session ID to maintain conversation context/history"
    )


class Citation(BaseModel):
    page_number: int = Field(..., description="Page number where the info was found")
    exact_snippet: str = Field(
        ..., description="The exact verbatim text snippet cited from the document"
    )
    document_id: str = Field(..., description="UUID of the cited document")
    document_name: str = Field(..., description="Filename of the cited document")


class QAResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the assistant")
    confidence_score: float = Field(
        ..., description="Confidence score between 0.0 and 1.0"
    )
    sources: List[Citation] = Field(
        ..., description="Citations supporting the generated answer"
    )
    conversation_id: str = Field(
        ..., description="Unique ID for the conversation thread"
    )
