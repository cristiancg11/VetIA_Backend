"""Application configuration module using Pydantic Settings."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE_PATH = BASE_DIR / ".env"



class Settings(BaseSettings):
    """Application settings and environment variable parser."""

    PROJECT_NAME: str = "VetIA / PetEmergency API"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Google Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Supabase Settings
    SUPABASE_URL: str = "https://xqkqogrfakobdxrfusdg.supabase.co"
    SUPABASE_KEY: str = "sb_publishable_J0B8Xwod53qVb7c2Q-bQgg__nkzi55I"
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
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

    @field_validator("GEMINI_API_KEY", mode="before")
    @classmethod
    def validate_gemini_api_key(cls, v: Optional[str]) -> str:
        """Enforces that GEMINI_API_KEY exists in the environment and is not an empty placeholder."""
        val = (v or "").strip()
        if not val or val == "your_gemini_api_key_here":
            logger.warning("❌ [CONFIG ASSERTION]: GEMINI_API_KEY está vacía o no existe en las variables de entorno.")
        return val

    def assert_gemini_api_key_exists(self) -> bool:
        """Asserts that GEMINI_API_KEY is present and populated before starting API requests."""
        key = (self.GEMINI_API_KEY or "").strip()
        is_valid = bool(key and key != "your_gemini_api_key_here")
        if not is_valid:
            logger.error("❌ [CONFIG ASSERTION FAILED]: GEMINI_API_KEY no existe o no tiene un token válido asignado.")
        return is_valid

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated origin strings into a list and guarantee local dev origins."""
        default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
        parsed: List[str] = []
        if isinstance(v, str):
            parsed = [origin.strip() for origin in v.split(",") if origin.strip()]
        elif isinstance(v, list):
            parsed = [str(origin).strip() for origin in v if str(origin).strip()]

        for default in default_origins:
            if default not in parsed:
                parsed.append(default)
        return parsed

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Provide a cached singleton instance of application settings with Gemini API Key validation."""
    settings = Settings()
    if not settings.assert_gemini_api_key_exists():
        logger.warning(
            "[CONFIG WARNING] La variable 'GEMINI_API_KEY' no está configurada o se encuentra vacía en el archivo .env. "
            "El servicio de triaje operará con el motor clínico de respaldo y contingencia."
        )
    return settings


