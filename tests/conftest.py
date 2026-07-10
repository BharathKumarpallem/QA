import pytest
from fastapi.testclient import TestClient
import chromadb
from app.main import app
from app.core.database import get_chroma_client
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from unittest.mock import AsyncMock, MagicMock


# Ephemeral (in-memory) ChromaDB Client for isolated tests
@pytest.fixture(scope="session")
def test_chroma_client():
    return chromadb.EphemeralClient()


# Automatically override the database dependency in all tests
@pytest.fixture(autouse=True)
def override_db(test_chroma_client):
    app.dependency_overrides[get_chroma_client] = lambda: test_chroma_client
    yield
    app.dependency_overrides.clear()


# Standard API TestClient
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# Mock OpenAI Embedding Service
@pytest.fixture
def mock_embedding_service():
    service = MagicMock(spec=EmbeddingService)
    # Return mock 1536-dimensional float vector for single embedding
    service.get_embedding = AsyncMock(return_value=[0.1] * 1536)
    # Return list of mock embeddings matching length of input text list
    service.get_embeddings = AsyncMock(
        side_effect=lambda texts: [[0.1] * 1536 for _ in texts]
    )
    return service


# Mock OpenAI LLM Service
@pytest.fixture
def mock_llm_service():
    service = MagicMock(spec=LLMService)
    return service
