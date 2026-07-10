from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import get_chroma_client
from app.middleware.request_id import RequestIdMiddleware
from app.middleware.error_handler import setup_exception_handlers
from app.routers import health, documents, qa
from loguru import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager that initializes configurations on startup and performs cleanups on shutdown.
    """
    # 1. Initialize Loguru logger
    setup_logging()
    logger.info("Initializing application lifespan setup...")
    logger.info(
        f"App Configuration loaded: {settings.APP_NAME} | Env: {settings.ENVIRONMENT}"
    )

    try:
        # 2. Touch ChromaDB client to verify database connection on startup
        db_client = get_chroma_client()
        logger.info(f"Database heartbeat: {db_client.heartbeat()}")
    except Exception as e:
        logger.error(
            f"Failed to connect to database during application startup: {str(e)}"
        )

    yield

    logger.info("Tearing down application session...")


# Instantiate FastAPI App with Swagger configurations
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade document Question Answering (RAG) backend API.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Request ID Tracking and logging middleware
app.add_middleware(RequestIdMiddleware)

# Setup centralized exception handling
setup_exception_handlers(app)

# Register Endpoints
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)
