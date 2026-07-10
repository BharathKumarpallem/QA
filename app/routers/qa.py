import uuid
from fastapi import APIRouter, Depends
from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    SearchChunkResult,
    QARequest,
    QAResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.services.chat_history import ChatHistoryService
from loguru import logger

router = APIRouter(tags=["Retrieval & Question Answering"])


# Dependency Providers
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


def get_llm_service() -> LLMService:
    return LLMService()


def get_chat_history() -> ChatHistoryService:
    return ChatHistoryService()


@router.post(
    "/search", response_model=SearchResponse, summary="Semantic Vector Search Chunks"
)
async def semantic_search(
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> SearchResponse:
    """
    Given a query, generates its vector embedding and retrieves the top_k
    most similar document chunks matching optionally supplied document filters.
    """
    logger.info(f"Received search query: '{request.query}' | top_k: {request.top_k}")

    # 1. Generate Query Vector Embedding
    query_vector = await embedding_service.get_embedding(request.query)

    # 2. Match similar vectors from ChromaDB
    chunks = vector_store.search_similar_chunks(
        query_embedding=query_vector,
        document_ids=request.document_ids,
        top_k=request.top_k,
    )

    results = [
        SearchChunkResult(
            text=c["text"],
            document_id=c["document_id"],
            document_name=c["document_name"],
            page_number=c["page_number"],
            score=c["score"],
        )
        for c in chunks
    ]
    return SearchResponse(results=results)


@router.post("/qa", response_model=QAResponse, summary="Context-Grounded Document Q&A")
async def question_answering(
    request: QARequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    llm_service: LLMService = Depends(get_llm_service),
    chat_history_service: ChatHistoryService = Depends(get_chat_history),
) -> QAResponse:
    """
    Asks a question across uploaded PDFs. Fetches top 8 chunks, pulls context-aware session history,
    forces LLM constraints, filters hallucinated citations, and logs session turns.
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    logger.info(
        f"Processing Q&A query: '{request.query}' | "
        f"Conversation: {conversation_id} | Filters: {request.document_ids}"
    )

    # 1. Embed current query
    query_vector = await embedding_service.get_embedding(request.query)

    # 2. Query top 8 similar chunks to provide rich context
    retrieved_chunks = vector_store.search_similar_chunks(
        query_embedding=query_vector, document_ids=request.document_ids, top_k=8
    )

    # 3. Pull conversation history log
    chat_history = chat_history_service.get_history(conversation_id)

    # 4. Generate structured answers via LLM Service
    qa_response = await llm_service.answer_question(
        query=request.query,
        retrieved_chunks=retrieved_chunks,
        chat_history=chat_history,
        conversation_id=conversation_id,
    )

    # 5. Log chat turn in context log to preserve conversational threads
    # Only save if answer was found to prevent poisoning history with non-answers
    if qa_response.confidence_score > 0.1:
        chat_history_service.add_message(conversation_id, "user", request.query)
        chat_history_service.add_message(
            conversation_id, "assistant", qa_response.answer
        )

    return qa_response
