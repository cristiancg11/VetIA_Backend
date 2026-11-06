"""Main FastAPI application pipeline mounting voice triage and multimodal services."""

import logging
from app.api.v1.voice import voice_router
from app.core.config import get_settings
from main import app, lifespan

logger = logging.getLogger("vetia.voice")
settings = get_settings()

# Mount the isolated voice triage router in the main FastAPI application pipeline
_has_voice_route = any(
    getattr(route, "path", "").endswith("/triage/voice-eval")
    for route in app.routes
)

if not _has_voice_route:
    app.include_router(voice_router, prefix=settings.API_V1_STR)
    logger.info(
        "[ROUTER] Voice triage router mounted successfully under %s/triage/voice-eval",
        settings.API_V1_STR,
    )

__all__ = ["app", "lifespan"]
