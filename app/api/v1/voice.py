"""FastAPI endpoint for Voice-based Clinical Triage and Field Audio Parsing."""

import logging
import os
from typing import Optional, Set
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.schemas.voice_triage import VoiceTriageResponse
from app.services.voice_service import process_voice_triage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/triage", tags=["Voice Triage"])
voice_router = router  # Alias for backward compatibility

ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".aac"}
ALLOWED_AUDIO_MIME_TYPES: Set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp3",
    "audio/mpeg",
    "audio/m4a",
    "audio/x-m4a",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
    "audio/aac",
}

EXTENSION_TO_MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".m4a": "audio/m4a",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
}


def validate_and_resolve_audio_mime(filename: str, content_type: str) -> str:
    """Validates audio file extension and content type, returning resolved MIME type."""
    ext = os.path.splitext(filename or "")[1].lower()
    ct = (content_type or "").lower().strip()

    is_valid_ext = ext in ALLOWED_AUDIO_EXTENSIONS
    is_valid_mime = ct in ALLOWED_AUDIO_MIME_TYPES

    if not is_valid_ext and not is_valid_mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Formato de audio no compatible ({ct or ext or 'desconocido'}). "
                "Formatos permitidos: .wav, .mp3, .m4a, .webm"
            ),
        )

    # Return valid standard MIME
    if is_valid_mime and ct not in ("application/octet-stream", ""):
        return ct
    return EXTENSION_TO_MIME.get(ext, "audio/wav")


@router.post(
    "/voice-eval",
    response_model=VoiceTriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcriptor y Evaluador por Voz de Campo para Triaje Veterinario",
    description=(
        "Recibe un archivo de audio (.wav, .mp3, .m4a o .webm) grabado en campo por el usuario o productor rural, "
        "junto con una transcripción en vivo opcional provista por el navegador. "
        "Utiliza Google Gemini para transcribir fielmente el relato, identificar la especie zootécnica o de compañía, "
        "aislar los signos clínicos relevantes, extraer el tiempo de evolución y clasificar la urgencia preliminar."
    ),
)
async def evaluate_voice_triage(
    audio_file: UploadFile = File(
        ...,
        description="Grabación de audio en campo (.wav, .mp3, .m4a o .webm) con el relato del paciente",
    ),
    live_transcript: Optional[str] = Form(
        None,
        description="Transcripción en vivo opcional capturada en el navegador del usuario",
    ),
) -> VoiceTriageResponse:
    """Evaluates field voice note and returns structured clinical triage response."""
    mime_type = validate_and_resolve_audio_mime(
        filename=audio_file.filename or "",
        content_type=audio_file.content_type or "",
    )

    try:
        audio_bytes = await audio_file.read()
    except Exception as exc:
        logger.error("[VOICE API ERROR] Error al leer los bytes del archivo de audio: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No fue posible leer el archivo de audio: {str(exc)}",
        )

    if not audio_bytes or len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo de audio proporcionado está vacío (0 bytes).",
        )

    result_dict = await process_voice_triage(
        audio_bytes=audio_bytes,
        mime_type=mime_type,
        live_transcript=live_transcript,
    )

    return VoiceTriageResponse(
        transcription=str(result_dict.get("transcription", "")),
        detected_species=result_dict.get("detected_species"),
        extracted_symptoms=str(result_dict.get("extracted_symptoms", "")),
        estimated_duration=result_dict.get("estimated_duration"),
        preliminary_urgency=str(result_dict.get("preliminary_urgency", "MODERADO")),
    )
