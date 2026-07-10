import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.embedding_service import EmbeddingService
from app.core.exceptions import LLMServiceException


@pytest.mark.asyncio
@patch("app.services.embedding_service.AsyncOpenAI")
async def test_get_embeddings_success(mock_openai_class) -> None:
    # 1. Setup mock structures matching OpenAI client responses
    mock_item = MagicMock()
    mock_item.embedding = [0.15] * 1536

    mock_response = MagicMock()
    mock_response.data = [mock_item]

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(return_value=mock_response)
    mock_openai_class.return_value = mock_client

    # 2. Call service method
    service = EmbeddingService()
    embeddings = await service.get_embeddings(["sample text"])

    # 3. Assertions
    assert len(embeddings) == 1
    assert embeddings[0] == [0.15] * 1536
    mock_client.embeddings.create.assert_called_once_with(
        input=["sample text"], model="text-embedding-3-small"
    )


@pytest.mark.asyncio
@patch("app.services.embedding_service.AsyncOpenAI")
async def test_get_embeddings_empty_input(mock_openai_class) -> None:
    service = EmbeddingService()
    embeddings = await service.get_embeddings([])
    assert embeddings == []


@pytest.mark.asyncio
@patch("app.services.embedding_service.AsyncOpenAI")
async def test_get_embeddings_throws_exception(mock_openai_class) -> None:
    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(
        side_effect=Exception("API connection timeout")
    )
    mock_openai_class.return_value = mock_client

    service = EmbeddingService()
    with pytest.raises(LLMServiceException) as exc_info:
        await service.get_embeddings(["some text"])

    assert "Failed to generate embeddings from OpenAI" in str(exc_info.value)
