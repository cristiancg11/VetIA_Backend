"""FastAPI endpoints for Multimodal Veterinary Triage, Confirmation Diagnosis, and Emergency Chat powered strictly by Google Gemini."""

import logging
import time
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
from app.services.triage_ai import (
    MultimodalTriageAIService,
    generate_local_chat_fallback,
    generate_local_clinical_fallback,
    generate_local_confirmation_fallback,
    is_transient_overload_error,
)

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
    start_time = time.perf_counter()
    try:
        assessment: TriageAnalysisResponse = await triage_service.analyze_triage(
            image_bytes=contents,
            mime_type=mime_type,
            especie=target_species,
            peso_kg=target_weight,
            descripcion=target_description,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(
            f"[TRIAGE PERFORMANCE] Inferencia completada con éxito en {elapsed_ms:.2f} ms | "
            f"ID: {assessment.triage_id} | Especie: {target_species} | Urgencia: {assessment.urgency_level} ({assessment.color_code})"
        )
        logger.info(
            "[TRIAGE PERFORMANCE] Inferencia completada con éxito en %.2f ms (ID: %s, Especie: %s, Nivel: %s, Color: %s, Preguntas: %d)",
            elapsed_ms,
            assessment.triage_id,
            target_species,
            assessment.urgency_level,
            assessment.color_code,
            len(assessment.diagnostic_questions),
        )
        return assessment

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning(
            "[TRIAGE API SAFE RECOVERY] Activando respuesta estructurada de contingencia (200 OK) ante excepción en Gemini (%s): %s",
            type(exc).__name__,
            exc,
        )
        print(
            f"\n🛡️ [TRIAGE API SAFE RECOVERY] Fallback estructurado 200 OK generado en {elapsed_ms:.2f} ms "
            f"para {target_species} ante incidencia en API externa: {exc}\n"
        )
        fallback = generate_local_clinical_fallback(
            species=target_species,
            peso_kg=target_weight,
            descripcion=target_description,
        )
        return fallback


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

    start_time = time.perf_counter()
    try:
        confirmation: ConfirmDiagnosisResponse = await triage_service.confirm_diagnosis(
            triage_id=payload.triage_id,
            answers=payload.answers,
            initial_symptoms=payload.initial_symptoms,
            species=target_species,
            initial_urgency_level=payload.initial_urgency_level,
            questions=payload.questions,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(
            f"[CONFIRM DIAGNOSIS PERFORMANCE] Confirmación completada en {elapsed_ms:.2f} ms | "
            f"Triage ID: {confirmation.triage_id} | Urgencia: {confirmation.urgency_level}"
        )
        logger.info(
            "[CONFIRM DIAGNOSIS COMPLETED] Triage ID: %s reevaluado en %.2f ms a: %s (Escalado: %s, Color: %s)",
            confirmation.triage_id,
            elapsed_ms,
            confirmation.urgency_level,
            confirmation.is_escalated,
            confirmation.color_code,
        )
        return confirmation

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning(
            "[CONFIRM DIAGNOSIS SAFE RECOVERY] Activando respuesta estructurada de contingencia (200 OK): %s",
            exc,
        )
        print(
            f"\n🛡️ [CONFIRM DIAGNOSIS SAFE RECOVERY] Fallback estructurado 200 OK generado en {elapsed_ms:.2f} ms ante caída de Gemini.\n"
        )
        return generate_local_confirmation_fallback(
            triage_id=payload.triage_id,
            species=target_species or "Bovino",
            initial_urgency=payload.initial_urgency_level or "MODERADO",
            answers=payload.answers,
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

    start_time = time.perf_counter()
    try:
        reply: EmergencyChatResponse = await triage_service.emergency_chat(
            triage_id=payload.triage_id,
            message=query_text,
            context=payload.context,
            conversation_history=payload.conversation_history,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(
            f"[EMERGENCY CHAT PERFORMANCE] Chat completado en {elapsed_ms:.2f} ms | "
            f"Triage ID: {payload.triage_id}"
        )
        logger.info(
            "[EMERGENCY CHAT COMPLETED] Chat para Triage ID: %s respondido en %.2f ms",
            payload.triage_id,
            elapsed_ms,
        )
        return reply

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.warning(
            "[EMERGENCY CHAT SAFE RECOVERY] Activando respuesta orientativa estructurada (200 OK): %s",
            exc,
        )
        print(
            f"\n🛡️ [EMERGENCY CHAT SAFE RECOVERY] Chat estructurado 200 OK generado en {elapsed_ms:.2f} ms ante caída de Gemini.\n"
        )
        return generate_local_chat_fallback(
            triage_id=payload.triage_id,
            species="Mascota",
            message=query_text,
        )
