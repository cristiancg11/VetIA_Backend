"""Application configuration module using Pydantic Settings."""

from functools import lru_cache
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment variable parser."""

    PROJECT_NAME: str = "VetIA / PetEmergency API"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Google Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Supabase Settings
    SUPABASE_URL: str = "https://vdxzoiededgmfysfoozd.supabase.co"
    SUPABASE_KEY: str = ""
    SUPABASE_STORAGE_BUCKET: str = "pet-images"
    SUPABASE_TABLE_NAME: str = "triage_records"

    # Concurrency and Multithreading Settings
    MAX_THREAD_WORKERS: int = 4

    # Network and CORS Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated origin strings into a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Provide a cached singleton instance of application settings."""
    return Settings()
