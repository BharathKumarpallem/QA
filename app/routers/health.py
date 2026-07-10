from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from app.core.config import settings
from app.core.database import get_chroma_client
from app.models.schemas import HealthResponse, VersionResponse
from loguru import logger
import chromadb

router = APIRouter(tags=["System Monitor"])


@router.get(
    "/health", response_model=HealthResponse, summary="Perform System Health Check"
)
async def health_check(
    client: chromadb.ClientAPI = Depends(get_chroma_client),
) -> HealthResponse:
    """
    Checks that the service is running and ChromaDB database connection is responsive.
    """
    db_status = "error"
    try:
        # ChromaDB heartbeat verifies if DB is active
        heartbeat = client.heartbeat()
        if heartbeat is not None:
            db_status = "connected"
    except Exception as e:
        logger.error(f"Health check failed to communicate with ChromaDB: {str(e)}")

    return HealthResponse(
        status="ok", database=db_status, timestamp=datetime.now(timezone.utc)
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Get API Service Version Details",
)
async def get_version() -> VersionResponse:
    """
    Returns application metadata (Version, Environment, and Application Name).
    """
    return VersionResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )
