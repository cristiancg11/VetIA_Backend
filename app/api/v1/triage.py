"""FastAPI endpoint for Multimodal Veterinary Triage powered by Google Gemini."""

import logging
from typing import Any, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.schemas.triage import TriageAnalysisResponse
from app.services.triage_ai import MultimodalTriageAIService

logger = logging.getLogger(__name__)

triage_router = APIRouter(prefix="", tags=["Multimodal Veterinary Triage"])

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}


def get_triage_ai_service(
    settings: Settings = Depends(get_settings),
    thread_pool_manager: ThreadPoolManager = Depends(get_thread_pool_manager),
) -> MultimodalTriageAIService:
    """Dependency injection provider for the Multimodal Triage AI Service."""
    return MultimodalTriageAIService(settings=settings, thread_pool_manager=thread_pool_manager)


def _clean_str(val: Any) -> Optional[str]:
    """Safely extract string, avoiding Form object issues if omitted in tests or python calls."""
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _clean_float(val: Any) -> Optional[float]:
    """Safely extract float from numeric or string values."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and val.strip():
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


@triage_router.post(
    "/triage/analyze",
    response_model=TriageAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Análisis clínico multimodal de triaje veterinario con Gemini",
    description=(
        "Recibe una fotografía de la lesión o animal junto con datos clínicos del paciente. "
        "Ejecuta inferencia multimodal con Gemini Flash y retorna la evaluación estructurada "
        "conforme al Manchester Triage System."
    ),
)
async def analyze_veterinary_triage(
    image: UploadFile = File(
        ...,
        description="Fotografía de la lesión, fluido patológico o animal (JPEG, PNG, WebP)",
    ),
    species: Optional[str] = Form(
        None,
        description="Especie del animal (ej. Canino, Felino)",
    ),
    especie: Optional[str] = Form(
        None,
        description="Alias en español para la especie",
    ),
    weight_kg: Optional[float] = Form(
        10.0,
        description="Peso aproximado del animal en kilogramos (default: 10.0)",
    ),
    peso_kg: Optional[float] = Form(
        None,
        description="Alias en español para el peso",
    ),
    description: Optional[str] = Form(
        None,
        description="Descripción detallada de síntomas y comportamiento",
    ),
    descripcion: Optional[str] = Form(
        None,
        description="Alias en español para la descripción",
    ),
    user_id: Optional[str] = Form(
        None,
        description="UUID opcional del usuario autenticado en Supabase",
    ),
    triage_service: MultimodalTriageAIService = Depends(get_triage_ai_service),
) -> TriageAnalysisResponse:
    """Execute complete multimodal veterinary triage workflow with robust parameter handling."""
    # Resolve parameters supporting both English and Spanish aliases
    target_species = _clean_str(species) or _clean_str(especie) or "Mascota"
    weight_val = _clean_float(weight_kg) or _clean_float(peso_kg) or 10.0
    target_weight = weight_val if weight_val > 0 else 10.0
    target_description = _clean_str(description) or _clean_str(descripcion) or "Evaluación de condición general del paciente"

    # Real-time console print for developer visibility in Uvicorn terminal
    print(f"--> [TRIAGE REQUEST] Recibida imagen: {image.filename}, Especie: {target_species}")
    logger.info(
        "--> [TRIAGE REQUEST] Recibida imagen: %s, Especie: %s, Peso: %s kg, User: %s",
        image.filename,
        target_species,
        target_weight,
        user_id or "anon",
    )

    # Read image contents directly as requested
    try:
        contents = await image.read()
    except Exception as read_err:
        logger.error("[TRIAGE ERROR] Error al leer bytes de la imagen: %s", read_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error de lectura en el archivo de imagen subido.",
        )

    # Determine safe MIME type
    mime_type = image.content_type or "image/jpeg"
    if mime_type.lower() not in ALLOWED_IMAGE_TYPES:
        mime_type = "image/jpeg"

    # Execute Multimodal Ingestion and Inference
    try:
        assessment: TriageAnalysisResponse = await triage_service.analyze_triage(
            image_bytes=contents,
            mime_type=mime_type,
            especie=target_species,
            peso_kg=target_weight,
            descripcion=target_description,
        )

        logger.info(
            "[TRIAGE COMPLETED] Evaluación finalizada con éxito. Nivel: %s (Color: %s), Déficit: %s%%",
            assessment.urgency_level,
            assessment.color_code,
            assessment.estimated_fluid_deficit_percent,
        )
        return assessment

    except Exception as exc:
        logger.error(
            "[TRIAGE UNEXPECTED ERROR] Fallo durante la inferencia: %s. Activando motor clínico de contingencia.",
            exc,
            exc_info=True,
        )
        return triage_service._generate_rule_based_fallback(
            especie=target_species,
            peso_kg=target_weight,
            descripcion=target_description,
        )
