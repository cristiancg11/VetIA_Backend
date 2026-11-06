"""Main application entrypoint and router pipeline for VetIA / PetEmergency FastAPI service."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.auth import check_supabase_health
from app.api.v1.router import api_v1_router
from app.api.v1.voice import router as voice_router
from app.core.config import get_settings
from app.core.supabase import get_supabase_client
from app.core.thread_pool import get_thread_pool_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vetia.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling application startup and shutdown."""
    settings = get_settings()
    thread_pool = get_thread_pool_manager()
    thread_pool.initialize(max_workers=settings.MAX_THREAD_WORKERS)
    logger.info("Iniciando %s en ambiente: %s", settings.PROJECT_NAME, settings.ENVIRONMENT)

    # Validate Supabase connection on startup
    try:
        supabase = get_supabase_client(settings=settings)
        logger.info(
            "[STARTUP] Supabase client configurado exitosamente hacia: %s",
            settings.SUPABASE_URL,
        )
    except Exception as exc:
        logger.error(
            "[STARTUP ERROR] Error inicializando Supabase al iniciar la app: %s",
            exc,
            exc_info=True,
        )

    yield

    thread_pool.shutdown(wait=True)
    logger.info("Apagando %s...", settings.PROJECT_NAME)


settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal AI-assisted Veterinary Triage and Emergency Locator API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
# Permit all multipart headers, streaming origins, and HTTP methods for browser voice recording
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount API Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)
app.include_router(voice_router, prefix="/api/v1")
print("🎙️ [VOICE ROUTE] /api/v1/triage/voice-eval montado exitosamente.")
logger.info("🎙️ [VOICE ROUTE] /api/v1/triage/voice-eval montado exitosamente.")


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Service health and liveness probe."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }


# Convenience alias for top-level /health/supabase endpoint
app.add_api_route(
    "/health/supabase",
    check_supabase_health,
    methods=["GET"],
    tags=["Health"],
    summary="Health probe for Supabase connectivity (root alias)",
)

__all__ = ["app", "lifespan", "voice_router"]
