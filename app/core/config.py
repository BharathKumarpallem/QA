from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    APP_NAME: str = "AI Document Q&A Service"
    APP_VERSION: str = "1.0.0"

    OPENAI_API_KEY: str

    CHROMA_DB_PATH: str = "./chroma_db"

    MAX_UPLOAD_SIZE_MB: int = 10

    # Text chunking settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # RAG parameters
    SIMILARITY_THRESHOLD: float = 0.55


# Global settings instance
settings = Settings()
