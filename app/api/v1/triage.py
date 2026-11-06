"""FastAPI endpoints for Multimodal Veterinary Triage, Confirmation Diagnosis, and Emergency Chat powered strictly by Google Gemini."""

import logging
from typing import Any, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.schemas.triage import (
    ConfirmDiagnosisRequest,
    ConfirmDiagnosisResponse,
    EmergencyChatRequest,
    EmergencyChatResponse,
    SUPPORTED_SPECIES_LIST,
    TriageAnalysisResponse,
    normalize_species_name,
)
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


# =============================================================================
# 1. ANALISIS DE TRIAJE MULTIMODAL ESTRICTO CON GOOGLE GEMINI (SIN MOCKS)
# =============================================================================
@triage_router.post(
    "/triage/analyze",
    response_model=TriageAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Análisis clínico multimodal de triaje veterinario y zootécnico con Gemini",
    description=(
        "Recibe una fotografía de la lesión o animal junto con datos clínicos del paciente. "
        "Soporta especies zootécnicas y rurales (Bovino, Equino, Porcino, Ovino / Caprino, Aves de Corral, Cuyes / Conejos) "
        "así como animales de compañía (Canino, Felino). "
        "Ejecuta inferencia multimodal real con Gemini pasando los bytes de la imagen y retorna "
        "la evaluación estructurada con fisiología y patología específica."
    ),
)
async def analyze_veterinary_triage(
    image: UploadFile = File(
        ...,
        description="Fotografía de la lesión, fluido patológico o animal (JPEG, PNG, WebP)",
    ),
    species: Optional[str] = Form(
        None,
        description="Especie del animal (ej. Bovino, Equino, Porcino, Ovino / Caprino, Aves de Corral, Cuyes / Conejos, Canino, Felino)",
    ),
    especie: Optional[str] = Form(
        None,
        description="Alias en español para la especie",
    ),
    weight_kg: Optional[float] = Form(
        None,
        description="Peso aproximado del animal en kilogramos (si se omite, se asigna el promedio de la especie)",
    ),
    peso_kg: Optional[float] = Form(
        None,
        description="Alias en español para el peso",
    ),
    description: Optional[str] = Form(
        None,
        description="Descripción detallada de síntomas, signos clínicos y comportamiento",
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
    """Execute complete multimodal veterinary triage workflow with real Gemini inference."""
    # Resolve parameters supporting both English and Spanish aliases with zootechnical normalization
    raw_species = _clean_str(species) or _clean_str(especie) or "Bovino"
    target_species = normalize_species_name(raw_species)

    # Species-tailored default weights
    weight_val = _clean_float(weight_kg) or _clean_float(peso_kg)
    if weight_val is None or weight_val <= 0:
        if target_species in ["Aves de Corral", "Cuyes / Conejos"]:
            target_weight = 1.0
        elif target_species in ["Bovino", "Equino"]:
            target_weight = 450.0
        elif target_species in ["Porcino", "Ovino / Caprino"]:
            target_weight = 60.0
        else:
            target_weight = 10.0
    else:
        target_weight = weight_val

    target_description = _clean_str(description) or _clean_str(descripcion) or "Evaluación de condición general del paciente"

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

    # Execute Multimodal Ingestion and Inference strictly via Gemini
    try:
        assessment: TriageAnalysisResponse = await triage_service.analyze_triage(
            image_bytes=contents,
            mime_type=mime_type,
            especie=target_species,
            peso_kg=target_weight,
            descripcion=target_description,
        )

        logger.info(
            "[TRIAGE COMPLETED] Evaluación finalizada con éxito. ID: %s, Nivel: %s (Color: %s), Preguntas: %d",
            assessment.triage_id,
            assessment.urgency_level,
            assessment.color_code,
            len(assessment.diagnostic_questions),
        )
        return assessment

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[TRIAGE ERROR] Error durante la inferencia de Gemini: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en inferencia multimodal de Gemini: {str(exc)}",
        )


# =============================================================================
# 2. ENDPOINT DE CONFIRMACION Y AJUSTE DE PRONOSTICO (TEST DE DESCARTE)
# =============================================================================
@triage_router.post(
    "/triage/confirm-diagnosis",
    response_model=ConfirmDiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Confirmación diagnóstica y ajuste de pronóstico clínico",
    description=(
        "Recibe el triage_id, los síntomas iniciales y las respuestas del test de descarte. "
        "Gemini re-evalúa el nivel de urgencia definitivo considerando las respuestas (por ejemplo, "
        "si el abdomen está duro o las encías blancas, escala de inmediato a CRÍTICO / ROJO)."
    ),
)
async def confirm_veterinary_diagnosis(
    payload: ConfirmDiagnosisRequest,
    triage_service: MultimodalTriageAIService = Depends(get_triage_ai_service),
) -> ConfirmDiagnosisResponse:
    """Execute clinical re-evaluation of triage urgency via Gemini based on rule-out responses."""
    target_species = normalize_species_name(payload.species) if payload.species else None
    print(f"--> [CONFIRM DIAGNOSIS REQUEST] Triage ID: {payload.triage_id}, Especie: {target_species}, Respuestas: {payload.answers}")
    logger.info(
        "--> [CONFIRM DIAGNOSIS REQUEST] Triage ID: %s, Especie: %s, Respuestas: %s",
        payload.triage_id,
        target_species or "N/A",
        payload.answers,
    )

    try:
        confirmation: ConfirmDiagnosisResponse = await triage_service.confirm_diagnosis(
            triage_id=payload.triage_id,
            answers=payload.answers,
            initial_symptoms=payload.initial_symptoms,
            species=target_species,
            initial_urgency_level=payload.initial_urgency_level,
            questions=payload.questions,
        )

        logger.info(
            "[CONFIRM DIAGNOSIS COMPLETED] Triage ID: %s reevaluado a: %s (Escalado: %s, Color: %s)",
            confirmation.triage_id,
            confirmation.urgency_level,
            confirmation.is_escalated,
            confirmation.color_code,
        )
        return confirmation

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[CONFIRM DIAGNOSIS ERROR] Error durante la confirmación: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en confirmación diagnóstica de Gemini: {str(exc)}",
        )


# =============================================================================
# 3. ENDPOINT PARA EL CHAT ASISTENTE DE URGENCIAS Y PRIMEROS AUXILIOS
# =============================================================================
@triage_router.post(
    "/triage/chat",
    response_model=EmergencyChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat Asistente de Urgencias y Primeros Auxilios",
    description=(
        "Recibe triage_id, user_message (o message) del tutor y el contexto del triaje actual. "
        "Responde con instrucciones precisas y empáticas de primeros auxilios sin emitir "
        "recetas farmacológicas no autorizadas."
    ),
)
async def triage_emergency_chat(
    payload: EmergencyChatRequest,
    triage_service: MultimodalTriageAIService = Depends(get_triage_ai_service),
) -> EmergencyChatResponse:
    """Process contextual emergency first-aid questions grounded on clinical triage context via Gemini."""
    query_text = payload.get_query()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El campo 'user_message' o 'message' es obligatorio.",
        )

    print(f"--> [EMERGENCY CHAT REQUEST] Triage ID: {payload.triage_id}, Mensaje: {query_text[:60]}")
    logger.info(
        "--> [EMERGENCY CHAT REQUEST] Triage ID: %s, Mensaje: %s",
        payload.triage_id,
        query_text[:80],
    )

    try:
        reply: EmergencyChatResponse = await triage_service.emergency_chat(
            triage_id=payload.triage_id,
            message=query_text,
            context=payload.context,
            conversation_history=payload.conversation_history,
        )
        return reply

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[EMERGENCY CHAT ERROR] Error en chat de emergencias: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en chat de emergencias de Gemini: {str(exc)}",
        )
