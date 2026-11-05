"""FastAPI endpoint for Multimodal Veterinary Triage powered by Google Gemini."""

import logging
from typing import Optional
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


@triage_router.post(
    "/triage/analyze",
    response_model=TriageAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Análisis clínico multimodal de triaje veterinario con Gemini",
    description=(
        "Recibe una fotografía clínica (lesión cutánea, vómito, deposición o estado general) "
        "junto con los datos del paciente (especie, peso en kg y descripción de síntomas). "
        "Ejecuta inferencia multimodal con Google Gemini Flash y retorna la clasificación "
        "estructurada de triaje, riesgos vitales y pautas seguras de primeros auxilios."
    ),
)
async def analyze_veterinary_triage(
    image: UploadFile = File(
        ...,
        description="Fotografía de la lesión, fluido patológico o animal (JPEG, PNG, WebP)",
    ),
    especie: str = Form(
        ...,
        description="Especie del animal (ej. Canino, Felino, Ave, Roedor)",
    ),
    peso_kg: float = Form(
        ...,
        description="Peso corporal aproximado del animal en kilogramos",
    ),
    descripcion: str = Form(
        ...,
        description="Descripción detallada de síntomas, comportamiento y tiempo de evolución",
    ),
    raza: Optional[str] = Form(
        None,
        description="Raza del paciente veterinario (opcional)",
    ),
    edad_aprox: Optional[str] = Form(
        None,
        description="Edad aproximada del animal (ej: 3 meses, 4 años) (opcional)",
    ),
    triage_service: MultimodalTriageAIService = Depends(get_triage_ai_service),
    thread_pool_manager: ThreadPoolManager = Depends(get_thread_pool_manager),
) -> TriageAnalysisResponse:
    """Execute complete multimodal veterinary triage workflow."""
    logger.info(
        "[TRIAGE REQUEST] Recibida solicitud de triaje para paciente %s (%s kg). Imagen: %s (%s)",
        especie,
        peso_kg,
        image.filename,
        image.content_type,
    )

    # 1. Validate Gemini API Key configuration
    if not triage_service.is_configured:
        logger.error("[TRIAGE ERROR] Inferencia bloqueada: GEMINI_API_KEY no configurada en el servidor.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El servicio de inteligencia artificial de triaje no está disponible: "
                "la variable de entorno GEMINI_API_KEY no está configurada o es inválida."
            ),
        )

    # 2. Validate Image Content Type
    if not image.content_type or image.content_type.lower() not in ALLOWED_IMAGE_TYPES:
        logger.warning(
            "[TRIAGE REJECTED] Formato de imagen no permitido: %s",
            image.content_type,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato de imagen inválido o no compatible ({image.content_type}). "
                "Los formatos válidos permitidos son: JPEG, PNG, WebP, HEIC."
            ),
        )

    # 3. Validate Clinical Fields
    if not especie or not especie.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El campo 'especie' es obligatorio.",
        )

    if peso_kg <= 0 or peso_kg > 600:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El peso corporal (peso_kg) debe ser un número positivo realista en kilogramos.",
        )

    if not descripcion or not descripcion.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La descripción clínica de síntomas no puede estar vacía.",
        )

    # 4. Offload file reading to thread pool to prevent async loop blocking
    try:
        image_bytes = await thread_pool_manager.run_in_thread(image.file.read)
    except Exception as read_err:
        logger.error("[TRIAGE ERROR] Error leyendo los bytes del archivo de imagen: %s", read_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se pudo procesar el archivo de imagen subido.",
        )

    if not image_bytes or len(image_bytes) == 0:
        logger.warning("[TRIAGE REJECTED] Archivo de imagen vacío subido por el cliente.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo de imagen proporcionado está vacío (0 bytes).",
        )

    # 5. Execute Multimodal Ingestion and Inference
    try:
        assessment: TriageAnalysisResponse = await triage_service.analyze_triage(
            image_bytes=image_bytes,
            mime_type=image.content_type,
            especie=especie.strip(),
            peso_kg=peso_kg,
            descripcion=descripcion.strip(),
            raza=raza.strip() if raza else None,
            edad_aprox=edad_aprox.strip() if edad_aprox else None,
        )

        logger.info(
            "[TRIAGE COMPLETED] Evaluación finalizada con éxito. Nivel: %s (Color: %s)",
            assessment.urgency_level,
            assessment.color_code,
        )
        return assessment

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[TRIAGE UNEXPECTED ERROR] Excepción al procesar el triaje con Gemini: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error inesperado al procesar la evaluación clínica: {str(exc)}",
        )

