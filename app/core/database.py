# pyrefly: ignore [missing-import]
import chromadb

# pyrefly: ignore [missing-import]
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

# pyrefly: ignore [missing-import]
from loguru import logger

# Singleton client reference
_chroma_client = None


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Returns the singleton ChromaDB API client, initializing it if necessary.
    """
    global _chroma_client
    if _chroma_client is None:
        logger.info(
            f"Initializing ChromaDB persistent client at path: {settings.CHROMA_DB_PATH}"
        )
        _chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client
