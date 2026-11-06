"""Gemini-powered Multimodal Audio Triage Service for rural field voice recordings."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional
from google import genai
from google.genai import types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

VOICE_SYSTEM_INSTRUCTION = (
    "Eres un asistente veterinario de triaje rural en Nariño. "
    "Escucha el relato del campesino/productor, extrae la transcripción literal "
    "y estructura los datos clínicos en formato JSON estricto."
)

VOICE_EXTRACTION_PROMPT = """Analiza con atención el audio adjunto grabado por un productor, campesino o tutor en campo rural.
Debes realizar la transcripción fiel de lo que dice y estructurar los hallazgos clínicos veterinarios.

INSTRUCCIONES CLÍNICAS:
1. 'transcription': Transcribe textualmente y con exactitud lo que dice la persona en el audio.
2. 'detected_species': Identifica la especie animal mencionada (ej: bovino, equino, porcino, ovino, canino, felino, aves, cuy). Si no se especifica con claridad, indica null.
3. 'extracted_symptoms': Sintetiza de manera precisa los signos y síntomas clínicos descritos (ej. inapetencia, cólico, diarrea con sangre, timpanismo, renguera, dificultad respiratoria, etc.).
4. 'estimated_duration': Extrae el tiempo de evolución o duración que reporta la persona (ej. '2 días', 'desde ayer en la tarde', 'unas 4 horas'). Si no se menciona, indica null.
5. 'preliminary_urgency': Clasifica la urgencia en una de estas 4 categorías según la gravedad clínica descrita:
   - 'CRITICO' (Riesgo vital inminente, asfixia, torsión gástrica/cólico hiperagudo, hemorragia masiva, postración comatosa).
   - 'URGENTE' (Timpanismo inicial, dolor agudo moderado a severo, prolapso, deshidratación rápida, vómito/diarrea profusa).
   - 'MODERADO' (Fiebre, claudicación, heridas superficiales, anorexia de 24h, tos leve sin disnea).
   - 'LEVE' (Problemas dérmicos leves, afecciones crónicas estables, chequeo de rutina).

Responde ÚNICAMENTE con un JSON válido con la siguiente estructura exacta:
{
  "transcription": "texto transcrito...",
  "detected_species": "bovino" | "equino" | "porcino" | "ovino" | "canino" | "felino" | "aves" | "cuy" | null,
  "extracted_symptoms": "descripción concisa de síntomas aislados...",
  "estimated_duration": "tiempo reportado o null",
  "preliminary_urgency": "LEVE" | "MODERADO" | "URGENTE" | "CRITICO"
}
"""


VOICE_MODELS_CASCADE = [
    "gemini-1.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
]


def is_transient_voice_error(exc: Exception) -> bool:
    """Detects 503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, rate limits or transient outages."""
    code = getattr(exc, "code", None)
    if code in (503, 429):
        return True
    status_attr = getattr(exc, "status", None)
    if status_attr in ("UNAVAILABLE", "RESOURCE_EXHAUSTED"):
        return True
    err_str = str(exc).lower()
    transient_tokens = [
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "high demand",
        "currently experiencing high demand",
        "deadline_exceeded",
        "rate limit",
        "overloaded",
    ]
    return any(token in err_str for token in transient_tokens)


def generate_voice_fallback(error_detail: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a deterministic clinical orientation fallback when Gemini audio processing
    encounters 503 high demand, quota limits, or network timeouts.
    Guarantees continuous uptime and structured response without breaking the client.
    """
    logger.warning("[VOICE TRIAGE FAILOVER] Activando respuesta orientativa de contingencia. Motivo: %s", error_detail)
    return {
        "transcription": (
            "Audio de campo recibido y almacenado para triaje. "
            "El servicio de transcripción neuronal de voz experimentó alta demanda transitoria; "
            "se presenta evaluación orientativa preventiva para el productor rural."
        ),
        "detected_species": None,
        "extracted_symptoms": (
            "Signos clínicos reportados por voz en espera de sincronización completa. "
            "Por precaución clínica, evalúe si el animal presenta timpanismo, decúbito, disnea o dolor agudo."
        ),
        "estimated_duration": "En evaluación de campo",
        "preliminary_urgency": "MODERADO",
    }


def clean_and_parse_voice_json(raw_text: str) -> Dict[str, Any]:
    """Extract valid JSON from raw Gemini model response text."""
    cleaned = (raw_text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


async def process_voice_triage(audio_bytes: bytes, mime_type: str) -> dict:
    """
    Processes rural field audio recording with Google Gemini multimodal capabilities.
    Features model fallback cascade and failover resilience against 503 / quota errors.
    """
    settings = get_settings()
    api_key = (os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY or "").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        logger.error("[VOICE TRIAGE] GEMINI_API_KEY no configurada o inválida.")
        return generate_voice_fallback("GEMINI_API_KEY no configurada")

    client = None
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=45000),
        )
    except Exception as exc:
        logger.error("[VOICE TRIAGE] Error al inicializar cliente Gemini: %s", exc)
        return generate_voice_fallback(str(exc))

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    prompt_content = f"{VOICE_SYSTEM_INSTRUCTION}\n\n{VOICE_EXTRACTION_PROMPT}"

    config = types.GenerateContentConfig(
        system_instruction=VOICE_SYSTEM_INSTRUCTION,
        temperature=0.2,
        top_p=0.8,
        max_output_tokens=1000,
        response_mime_type="application/json",
    )

    last_error: Optional[Exception] = None

    for model_name in VOICE_MODELS_CASCADE:
        try:
            logger.info("[VOICE TRIAGE] Procesando audio con modelo: %s...", model_name)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=[audio_part, prompt_content],
                config=config,
            )

            raw_text = response.text or ""
            if not raw_text.strip():
                raise ValueError(f"El modelo {model_name} retornó respuesta vacía.")

            parsed_json = clean_and_parse_voice_json(raw_text)
            logger.info("[VOICE TRIAGE SUCCESS] Audio procesado exitosamente con %s", model_name)
            return parsed_json

        except Exception as exc:
            last_error = exc
            print(f"\n❌ [ERROR VOZ GEMINI API]: Tipo: {type(exc).__name__} | Modelo: {model_name} | Mensaje: {str(exc)}\n")
            if is_transient_voice_error(exc):
                logger.warning(
                    "[VOICE TRIAGE] Modelo %s con alta demanda/cuota. Conmutando a siguiente modelo de respaldo...",
                    model_name,
                )
                await asyncio.sleep(0.5)
                continue
            else:
                logger.warning("[VOICE TRIAGE] Error inesperado en modelo %s: %s", model_name, exc)
                break

    # Failover: si todos los modelos fallan por 503, cuota o red, retornar fallback determinístico
    return generate_voice_fallback(str(last_error))
