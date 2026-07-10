from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock
from app.main import app
from app.routers.documents import (
    get_pdf_processor,
    get_embedding_service,
    get_vector_store,
)
from app.models.schemas import QAResponse, Citation


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers


def test_version_endpoint(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Document Q&A Service"
    assert "version" in data
    assert "environment" in data
    assert "X-Request-ID" in response.headers


def test_upload_invalid_mime_type(client: TestClient) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", b"plain text context", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file type" in data["detail"]
    assert data["error_type"] == "InvalidDocumentException"
    assert "request_id" in data


def test_upload_duplicate_detection_flow() -> None:
    # Setup mocks to trigger duplicate detection path
    mock_pdf_processor = MagicMock()
    mock_pdf_processor.compute_sha256.return_value = "duplicate-sha256-hash"

    mock_vector = MagicMock()
    mock_vector.check_duplicate_by_hash.return_value = {
        "document_id": "existing-uuid-1234",
        "filename": "existing.pdf",
        "sha256": "duplicate-sha256-hash",
        "page_count": 5,
        "chunk_count": 10,
    }

    app.dependency_overrides[get_pdf_processor] = lambda: mock_pdf_processor
    app.dependency_overrides[get_vector_store] = lambda: mock_vector

    with TestClient(app) as local_client:
        response = local_client.post(
            "/documents/upload",
            files={"file": ("duplicate.pdf", b"pdf_binary_content", "application/pdf")},
        )
        assert response.status_code == 409
        data = response.json()
        assert "Duplicate document detected" in data["detail"]
        assert "existing-uuid-1234" in data["detail"]
        assert data["error_type"] == "DuplicateDocumentException"

    app.dependency_overrides.clear()


def test_upload_success_mocked() -> None:
    mock_pdf_processor = MagicMock()
    mock_pdf_processor.compute_sha256.return_value = "new-sha256-hash"
    mock_pdf_processor.process_pdf.return_value = (
        [{"text": "Sample text chunk.", "page_number": 1}],
        1,
    )

    mock_emb = AsyncMock()
    mock_emb.get_embeddings.return_value = [[0.1] * 1536]

    mock_vector = MagicMock()
    mock_vector.check_duplicate_by_hash.return_value = None
    mock_vector.add_document_chunks.return_value = None

    app.dependency_overrides[get_pdf_processor] = lambda: mock_pdf_processor
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_vector_store] = lambda: mock_vector

    with TestClient(app) as local_client:
        response = local_client.post(
            "/documents/upload",
            files={"file": ("contract.pdf", b"pdf_binary_content", "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "contract.pdf"
        assert data["sha256"] == "new-sha256-hash"
        assert data["page_count"] == 1
        assert data["chunk_count"] == 1
        assert data["status"] == "processed"
        assert "document_id" in data

    app.dependency_overrides.clear()


def test_qa_endpoint_success_mocked() -> None:
    mock_emb = AsyncMock()
    mock_emb.get_embedding.return_value = [0.1] * 1536

    mock_vector = MagicMock()
    mock_vector.search_similar_chunks.return_value = [
        {
            "text": "The contract duration is 3 years.",
            "document_id": "doc-uuid-1",
            "document_name": "contract.pdf",
            "page_number": 3,
            "score": 0.82,
        }
    ]

    mock_llm = AsyncMock()
    mock_llm.answer_question.return_value = QAResponse(
        answer="The contract duration is 3 years.",
        confidence_score=0.95,
        sources=[
            Citation(
                page_number=3,
                exact_snippet="The contract duration is 3 years.",
                document_id="doc-uuid-1",
                document_name="contract.pdf",
            )
        ],
        conversation_id="conv-session-1",
    )

    from app.routers.qa import (
        get_embedding_service as get_qa_embedding_service,
        get_vector_store as get_qa_vector_store,
        get_llm_service as get_qa_llm_service,
    )

    app.dependency_overrides[get_qa_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_qa_vector_store] = lambda: mock_vector
    app.dependency_overrides[get_qa_llm_service] = lambda: mock_llm

    with TestClient(app) as local_client:
        response = local_client.post(
            "/qa",
            json={
                "query": "How long is the contract?",
                "conversation_id": "conv-session-1",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The contract duration is 3 years."
        assert data["confidence_score"] == 0.95
        assert len(data["sources"]) == 1
        assert data["sources"][0]["page_number"] == 3
        assert data["conversation_id"] == "conv-session-1"

    app.dependency_overrides.clear()
