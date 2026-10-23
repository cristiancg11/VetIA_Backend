"""Main application entrypoint for VetIA / PetEmergency FastAPI service."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.thread_pool import get_thread_pool_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager handling application startup and graceful shutdown.
    Initializes the worker ThreadPoolExecutor and shuts it down cleanly upon exit.
    """
    settings = get_settings()
    thread_pool = get_thread_pool_manager()
    thread_pool.initialize(max_workers=settings.MAX_THREAD_WORKERS)
    yield
    thread_pool.shutdown(wait=True)


settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal AI-assisted Veterinary Triage and Emergency Locator API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Service health and liveness probe."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
    )
