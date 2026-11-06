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
    Transcribes speaker audio and parses clinical entities into a structured dictionary.
    """
    settings = get_settings()
    api_key = (os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY or "").strip()

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=45000),
    )

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

    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-1.5-flash",
        contents=[audio_part, prompt_content],
        config=config,
    )

    raw_text = response.text or ""
    parsed_json = clean_and_parse_voice_json(raw_text)
    return parsed_json
