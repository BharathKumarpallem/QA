from typing import List
from openai import AsyncOpenAI
from app.core.config import settings
from app.core.exceptions import LLMServiceException
from loguru import logger


class EmbeddingService:
    """
    Service responsible for interacting with the OpenAI Embeddings API
    using the text-embedding-3-small model.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-3-small"

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generates embedding vectors for the provided text chunks.

        Args:
            texts: List of strings to be vectorized.

        Returns:
            List[List[float]]: List of 1536-dimensional float vectors.
        """
        if not texts:
            return []

        logger.info(f"Requesting embeddings from OpenAI for {len(texts)} text chunks.")
        try:
            response = await self.client.embeddings.create(
                input=texts, model=self.model
            )
            # Order matches input order
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI Embeddings API error: {str(e)}")
            raise LLMServiceException(
                f"Failed to generate embeddings from OpenAI: {str(e)}"
            )

    async def get_embedding(self, text: str) -> List[float]:
        """
        Convenience method to generate an embedding for a single string.
        """
        embeddings = await self.get_embeddings([text])
        return embeddings[0]
