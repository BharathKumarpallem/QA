import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.llm_service import LLMService, LLMOutputResponse, LLMOutputCitation


@pytest.mark.asyncio
async def test_answer_preemptive_similarity_cutoff() -> None:
    service = LLMService()
    # Retreived chunks all below default threshold of 0.55
    retrieved_chunks = [
        {
            "text": "Chunk text A",
            "document_id": "doc1",
            "document_name": "doc.pdf",
            "page_number": 1,
            "score": 0.45,
        },
        {
            "text": "Chunk text B",
            "document_id": "doc1",
            "document_name": "doc.pdf",
            "page_number": 2,
            "score": 0.32,
        },
    ]

    response = await service.answer_question(
        query="What is FastAPI?",
        retrieved_chunks=retrieved_chunks,
        chat_history=[],
        conversation_id="session-456",
    )

    # Assert confidence 0.0 and standard fallback response
    assert response.confidence_score == 0.0
    assert "I cannot find the answer" in response.answer
    assert response.sources == []
    assert response.conversation_id == "session-456"


@pytest.mark.asyncio
@patch("app.services.llm_service.AsyncOpenAI")
async def test_answer_question_successful_flow(mock_openai_class) -> None:
    # 1. Setup mock structured completion returns
    mock_citation = LLMOutputCitation(
        page_number=2,
        exact_snippet="ChromaDB is a database for vectors.",
        document_id="doc-99",
        document_name="manual.pdf",
    )
    mock_parsed_response = LLMOutputResponse(
        answer="The vector database used is ChromaDB.",
        confidence_score=0.95,
        citations=[mock_citation],
    )

    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_parsed_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)
    mock_openai_class.return_value = mock_client

    # 2. Call service with matching chunk containing citation snippet
    service = LLMService()
    retrieved_chunks = [
        {
            "text": "ChromaDB is a database for vectors. Use it to perform semantic queries.",
            "document_id": "doc-99",
            "document_name": "manual.pdf",
            "page_number": 2,
            "score": 0.88,
        }
    ]

    response = await service.answer_question(
        query="What is ChromaDB?",
        retrieved_chunks=retrieved_chunks,
        chat_history=[],
        conversation_id="session-999",
    )

    assert response.answer == "The vector database used is ChromaDB."
    assert response.confidence_score == 0.95
    assert len(response.sources) == 1
    assert response.sources[0].exact_snippet == "ChromaDB is a database for vectors."
    assert response.sources[0].page_number == 2


@pytest.mark.asyncio
@patch("app.services.llm_service.AsyncOpenAI")
async def test_citation_hallucination_discarded(mock_openai_class) -> None:
    # LLM outputs a citation snippet that is NOT in the retrieved chunks
    mock_citation = LLMOutputCitation(
        page_number=1,
        exact_snippet="Hallucinated snippet not in source.",
        document_id="doc-99",
        document_name="manual.pdf",
    )
    mock_parsed_response = LLMOutputResponse(
        answer="This answers the question.",
        confidence_score=0.9,
        citations=[mock_citation],
    )

    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_parsed_response
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)
    mock_openai_class.return_value = mock_client

    service = LLMService()
    retrieved_chunks = [
        {
            "text": "This is the actual source chunk text.",
            "document_id": "doc-99",
            "document_name": "manual.pdf",
            "page_number": 1,
            "score": 0.85,
        }
    ]

    response = await service.answer_question(
        query="What is the answer?",
        retrieved_chunks=retrieved_chunks,
        chat_history=[],
        conversation_id="session-999",
    )

    # The citation should have been validated and removed because snippet is not in source
    assert len(response.sources) == 0
    # Confidence score should have been penalized because citations failed validation
    assert response.confidence_score == 0.3
