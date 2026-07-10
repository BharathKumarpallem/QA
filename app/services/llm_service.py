from typing import List, Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.exceptions import LLMServiceException
from app.models.schemas import Citation, QAResponse
from loguru import logger


# Internal Pydantic models for OpenAI Structured Outputs
class LLMOutputCitation(BaseModel):
    page_number: int = Field(
        ..., description="The page number where the information is located"
    )
    exact_snippet: str = Field(
        ...,
        description="The exact verbatim text snippet cited from the document context",
    )
    document_id: str = Field(
        ..., description="The document_id of the document containing this snippet"
    )
    document_name: str = Field(
        ..., description="The name of the document containing this snippet"
    )


class LLMOutputResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The detailed, factually accurate answer based strictly on the provided context",
    )
    confidence_score: float = Field(
        ...,
        description="The confidence score between 0.0 and 1.0 based on availability of context",
    )
    citations: List[LLMOutputCitation] = Field(
        ..., description="List of source citations used to construct the answer"
    )


class LLMService:
    """
    Service responsible for interacting with GPT-4o-mini to generate
    context-grounded, citation-validated Q&A responses.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "gpt-4o-mini"

    async def answer_question(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        conversation_id: str,
    ) -> QAResponse:
        """
        Generates a structured answer using retrieved chunks and conversation history.
        Enforces similarity thresholds and validates returned citations.
        """
        # Confidence Threshold Defense
        # If no chunks were retrieved or the best chunk score is below the threshold, return pre-emptively
        best_similarity = (
            max([chunk["score"] for chunk in retrieved_chunks])
            if retrieved_chunks
            else 0.0
        )

        if best_similarity < settings.SIMILARITY_THRESHOLD:
            logger.info(
                f"Best similarity score ({best_similarity:.4f}) is below threshold "
                f"({settings.SIMILARITY_THRESHOLD}). Returning 'Not Found' to prevent hallucination."
            )
            return QAResponse(
                answer="I cannot find the answer to your question in the uploaded documents.",
                confidence_score=0.0,
                sources=[],
                conversation_id=conversation_id,
            )

        # Build context prompt from retrieved chunks
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            context_str += (
                f"--- CONTEXT BLOCK {idx+1} ---\n"
                f"Document ID: {chunk['document_id']}\n"
                f"Document Name: {chunk['document_name']}\n"
                f"Page Number: {chunk['page_number']}\n"
                f"Text:\n{chunk['text']}\n\n"
            )

        # Setup developer/system system prompts and chat messages
        system_prompt = (
            "You are a professional AI Document Assistant. Your goal is to answer the user's question "
            "strictly using the provided document context blocks. Follow these instructions carefully:\n"
            "1. Answer the question based ONLY on the provided context blocks. Do NOT use any external knowledge.\n"
            "2. For every fact, claim, or quote you extract, you MUST include a citation object in the `citations` list. "
            "A citation must have the exact page number, the exact document ID, document name, and a verbatim snippet "
            "from the context text.\n"
            "3. If the context does not contain the answer, answer: "
            "'I cannot find the answer to your question in the uploaded documents.', set `confidence_score` to 0.0, "
            "and leave the `citations` list empty.\n"
            "4. Assign a `confidence_score` between 0.0 and 1.0. If the context answers the query directly, "
            "confidence should be high (0.8 - 1.0). If it only partially answers or is ambiguous, score lower (0.5 - 0.7).\n"
            "5. The cited `exact_snippet` must be an absolute substring match (case-insensitive and whitespace-normalized) "
            "within the text of the source context block you are citing."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history for context (up to last 10 messages)
        for msg in chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current query
        current_query_with_context = (
            f"Document Context:\n{context_str}\n" f"User Question: {query}"
        )
        messages.append({"role": "user", "content": current_query_with_context})

        try:
            logger.info(
                f"Sending Q&A request to OpenAI ({self.model}) for conversation {conversation_id}."
            )
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=LLMOutputResponse,
                temperature=0.0,  # Greedy decoding to minimize hallucination
            )

            result: LLMOutputResponse = response.choices[0].message.parsed

            # Post-Process & Validate Citations
            validated_citations = self._validate_citations(
                result.citations, retrieved_chunks
            )

            # If citations were returned but none could be validated, adjust confidence
            final_confidence = result.confidence_score
            if result.citations and not validated_citations:
                logger.warning(
                    "All LLM citations failed validation. Reducing confidence score."
                )
                final_confidence = min(final_confidence, 0.3)

            return QAResponse(
                answer=result.answer,
                confidence_score=final_confidence,
                sources=validated_citations,
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.error(f"OpenAI Chat Completion parse failed: {str(e)}")
            raise LLMServiceException(f"Failed to generate answer from LLM: {str(e)}")

    def _validate_citations(
        self,
        llm_citations: List[LLMOutputCitation],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Citation]:
        """
        Validates that each cited exact snippet is actually present in the retrieved chunks
        for that specific document ID and page number.
        """
        validated: List[Citation] = []

        for cit in llm_citations:
            snippet_normalized = " ".join(cit.exact_snippet.lower().split())
            if not snippet_normalized:
                continue

            matched = False
            for chunk in retrieved_chunks:
                # Validate matches document ID and page number
                if (
                    chunk["document_id"] == cit.document_id
                    and chunk["page_number"] == cit.page_number
                ):
                    chunk_text_normalized = " ".join(chunk["text"].lower().split())

                    if snippet_normalized in chunk_text_normalized:
                        matched = True
                        break

            if matched:
                validated.append(
                    Citation(
                        page_number=cit.page_number,
                        exact_snippet=cit.exact_snippet,
                        document_id=cit.document_id,
                        document_name=cit.document_name,
                    )
                )
            else:
                logger.warning(
                    f"Invalid citation snippet discarded | Page: {cit.page_number} "
                    f"| Snippet: '{cit.exact_snippet}'"
                )

        return validated
