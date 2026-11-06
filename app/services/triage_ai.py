"""Pure Google Gemini Multimodal Veterinary Triage Service.
Zero static mocks or hardcoded template fallbacks:
Enforces real image byte ingestion, species-specific clinical reasoning,
and returns raw Gemini AI responses or raises explicit 500 errors on failure.
"""

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from app.core.config import Settings, get_settings
from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.schemas.triage import (
    ChatMessage,
    ConfirmDiagnosisRequest,
    ConfirmDiagnosisResponse,
    DiagnosticQuestion,
    EmergencyChatRequest,
    EmergencyChatResponse,
    TriageAnalysisResponse,
)

logger = logging.getLogger(__name__)

# Global in-memory session cache for clinical triage context across requests
_GLOBAL_TRIAGE_SESSION_STORE: Dict[str, Dict[str, Any]] = {}


def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Extract valid JSON from raw Gemini model text response,
    stripping markdown ```json code blocks and leading/trailing chatter.
    """
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


def build_triage_prompt(species: str, weight_kg: float, description: str) -> str:
    """Builds clinical multimodal veterinary triage prompt."""
    return f"""Actúa como Médico Veterinario Especialista en Urgencias.
Analiza detalladamente esta imagen de un {species} de peso {weight_kg} kg.
Descripción del tutor: "{description}".

INSTRUCCIONES CLÍNICAS OBLIGATORIAS:
1. Evalúa según la especie indicada ({species}). NUNCA hables de canino si es un felino.
2. En visual_findings: describe minuciosamente lo que se observa en la foto real (color exacto, textura, presencia de sangre/moco, postura, signos evidentes). Sé detallado y exhaustivo.
3. En presumptive_diagnosis: ofrece un diagnóstico presuntivo clínico fundamentado y profundo (al menos 2 hipótesis explicadas).
4. En vital_risks: explica qué órganos o funciones biológicas están comprometidas de forma inminente.
5. En first_aid_actions: lista pasos de soporte inmediato y seguro antes de llegar a la clínica.
6. En diagnostic_questions: formula 3 preguntas clave de descarte sobre el estado actual del animal.

Responde ÚNICAMENTE en JSON válido con este formato:
{{
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
  "visual_findings": "...",
  "presumptive_diagnosis": ["...", "..."],
  "vital_risks": ["..."],
  "first_aid_actions": ["..."],
  "warnings_contraindications": ["..."],
  "diagnostic_questions": [
    {{"id": "q1", "question": "..."}},
    {{"id": "q2", "question": "..."}},
    {{"id": "q3", "question": "..."}}
  ]
}}
"""


class MultimodalTriageAIService:
    """
    Pure Multimodal Veterinary Triage AI Service powered strictly by Google Gemini.
    No hardcoded mocks or canned templates.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        thread_pool_manager: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._thread_pool_manager = thread_pool_manager or get_thread_pool_manager()
        self._api_key = (os.getenv("GEMINI_API_KEY") or self._settings.GEMINI_API_KEY or "").strip()
        self._primary_model = (os.getenv("GEMINI_MODEL") or self._settings.GEMINI_MODEL or "gemini-2.5-flash").strip()

        # Candidate models prioritized: primary model first, followed by active Gemini Flash models
        self._candidate_models = [
            self._primary_model,
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
        ]
        # Deduplicate preserving order
        self._candidate_models = list(dict.fromkeys(self._candidate_models))

        self._client: Optional[genai.Client] = None
        if self._api_key and self._api_key != "your_gemini_api_key_here":
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(timeout=15000),
                )
                logger.info(
                    "[TRIAGE AI] Google GenAI Client inicializado. Modelos candidatos: %s",
                    self._candidate_models,
                )
            except Exception as exc:
                logger.error("[TRIAGE AI] Error al inicializar Google GenAI Client: %s", exc)
                self._client = None

    def _sync_analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        especie: str,
        peso_kg: float,
        descripcion: str,
    ) -> TriageAnalysisResponse:
        """
        Executes genuine multimodal inference with Google Gemini.
        Passes the image bytes part and prompt content. Raises on failure (NO mocks).
        """
        print(f"[TRIAGE AI] Analizando imagen de {len(image_bytes)} bytes para especie: {especie}")
        logger.info("[TRIAGE AI] Analizando imagen de %d bytes para especie: %s", len(image_bytes), especie)

        if not self._client or not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY no está configurada en las variables de entorno (.env) o el cliente de Gemini no pudo inicializarse."
            )

        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("No se recibieron bytes de imagen válidos para el análisis de triaje.")

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type or "image/jpeg",
        )

        prompt_text = build_triage_prompt(
            species=especie,
            weight_kg=peso_kg,
            description=descripcion,
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )

        last_error: Optional[Exception] = None

        for model_name in self._candidate_models:
            try:
                logger.info("[TRIAGE AI] Ejecutando inferencia multimodal en Gemini con modelo: %s", model_name)
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=[image_part, prompt_text],
                    config=config,
                )

                raw_text = response.text or ""
                if not raw_text.strip():
                    raise ValueError(f"El modelo {model_name} retornó una respuesta de texto vacía.")

                result_dict = clean_and_parse_json(raw_text)
                if not isinstance(result_dict, dict):
                    raise ValueError(f"La respuesta de Gemini no es un objeto JSON válido: {raw_text[:100]}")

                # Ensure triage_id and valid questions
                if not result_dict.get("triage_id"):
                    result_dict["triage_id"] = str(uuid.uuid4())

                print(f"[TRIAGE AI] Respuesta generada: {result_dict.get('urgency_level')} - {str(result_dict.get('visual_findings', ''))[:80]}...")
                logger.info(
                    "[TRIAGE AI SUCCESS] Inferencia exitosa con modelo %s. Urgencia: %s",
                    model_name,
                    result_dict.get("urgency_level"),
                )

                assessment = TriageAnalysisResponse.model_validate(result_dict)
                self._save_session(assessment.triage_id, assessment, especie, peso_kg, descripcion)
                return assessment

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[TRIAGE AI WARNING] Fallo con modelo %s: %s. Probando siguiente modelo.",
                    model_name,
                    exc,
                )

        # If all candidate models fail, raise the exact Gemini error
        logger.error("[TRIAGE AI FATAL] Todos los modelos de Gemini fallaron: %s", last_error)
        raise RuntimeError(f"Error en inferencia multimodal de Gemini: {str(last_error)}")

    def _save_session(
        self,
        triage_id: str,
        assessment: TriageAnalysisResponse,
        especie: str,
        peso_kg: float,
        descripcion: str,
    ) -> None:
        """Caches active triage session context for chat and confirmation endpoints."""
        _GLOBAL_TRIAGE_SESSION_STORE[triage_id] = {
            "triage_id": triage_id,
            "species": especie,
            "weight_kg": peso_kg,
            "initial_symptoms": descripcion,
            "assessment": assessment.model_dump(),
            "questions": [q.model_dump() for q in assessment.diagnostic_questions],
        }

    async def analyze_triage(
        self,
        image_bytes: bytes,
        mime_type: str,
        especie: str,
        peso_kg: float,
        descripcion: str,
    ) -> TriageAnalysisResponse:
        """Asynchronously dispatches triage analysis to thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_analyze,
            image_bytes,
            mime_type,
            especie,
            peso_kg,
            descripcion,
        )

    # -------------------------------------------------------------------------
    # 2. CONFIRMATION DIAGNOSIS (REEVALUATION VIA GEMINI)
    # -------------------------------------------------------------------------
    def _sync_confirm_diagnosis(
        self,
        triage_id: str,
        answers: Dict[str, Any],
        initial_symptoms: Optional[str] = None,
        species: Optional[str] = None,
        initial_urgency_level: Optional[str] = None,
        questions: Optional[List[DiagnosticQuestion]] = None,
    ) -> ConfirmDiagnosisResponse:
        """Executes clinical re-evaluation strictly with Gemini (NO mocks)."""
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        target_species = species or session.get("species") or "Mascota"
        target_symptoms = initial_symptoms or session.get("initial_symptoms") or "Paciente en urgencias"
        target_urgency = initial_urgency_level or (
            session.get("assessment", {}).get("urgency_level") if session.get("assessment") else "MODERADO"
        )

        if not self._client or not self._api_key:
            raise RuntimeError("GEMINI_API_KEY no está configurada.")

        ans_lines = [f"- Pregunta ID {k}: {'SÍ (Confirmado)' if v else 'NO'}" for k, v in answers.items()]
        answers_str = "\n".join(ans_lines)

        prompt = f"""Actúa como Médico Veterinario Especialista en Urgencias y Cuidados Críticos.
Reevalúa de forma definitiva este caso de triaje clínico veterinario:
- Triage ID: {triage_id}
- Especie: {target_species}
- Síntomas iniciales: "{target_symptoms}"
- Urgencia preliminar: {target_urgency}

Respuestas del tutor al test de descarte clínico:
{answers_str}

REGLAS OBLIGATORIAS:
1. Reevalúa la urgencia considerando la fisiopatología estricta de {target_species}.
2. Si el tutor confirma signos de alarma vital (encías blancas/moradas, abdomen duro/doloroso, ingesta de veneno o sangrado activo), ESCALA INMEDIATAMENTE a CRÍTICO / RED.
3. Responde ÚNICAMENTE en JSON válido con este formato:
{{
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
  "is_escalated": true | false,
  "clinical_reassessment": "Explicación detallada del ajuste o confirmación",
  "presumptive_diagnosis": ["...", "..."],
  "vital_risks": ["..."],
  "first_aid_actions": ["..."],
  "warnings_contraindications": ["..."],
  "estimated_fluid_deficit_percent": 5.0,
  "recommended_facility_type": "Clínica 24 horas con quirófano" | "Consultorio general"
}}
"""

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )

        last_error: Optional[Exception] = None
        for model_name in self._candidate_models:
            try:
                logger.info("[TRIAGE AI] Reevaluando con Gemini modelo: %s", model_name)
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                raw_text = response.text or ""
                result_dict = clean_and_parse_json(raw_text)
                result_dict["triage_id"] = triage_id
                return ConfirmDiagnosisResponse.model_validate(result_dict)
            except Exception as exc:
                last_error = exc
                logger.warning("[TRIAGE AI WARNING] Fallo reevaluación con %s: %s", model_name, exc)

        raise RuntimeError(f"Error en reevaluación confirmatoria de Gemini: {str(last_error)}")

    async def confirm_diagnosis(
        self,
        triage_id: str,
        answers: Dict[str, Any],
        initial_symptoms: Optional[str] = None,
        species: Optional[str] = None,
        initial_urgency_level: Optional[str] = None,
        questions: Optional[List[DiagnosticQuestion]] = None,
    ) -> ConfirmDiagnosisResponse:
        """Asynchronously dispatches diagnosis confirmation to thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_confirm_diagnosis,
            triage_id,
            answers,
            initial_symptoms,
            species,
            initial_urgency_level,
            questions,
        )

    # -------------------------------------------------------------------------
    # 3. EMERGENCY CHAT ASSISTANT (PURE GEMINI INFERENCE)
    # -------------------------------------------------------------------------
    def _sync_emergency_chat(
        self,
        triage_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> EmergencyChatResponse:
        """Processes emergency chat queries strictly with Gemini (NO mocks)."""
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        full_context = dict(session)
        if context:
            full_context.update(context)

        species = full_context.get("species") or "Mascota"
        assessment = full_context.get("assessment", {})
        urgency = assessment.get("urgency_level") or full_context.get("urgency_level") or "MODERADO"

        if not self._client or not self._api_key:
            raise RuntimeError("GEMINI_API_KEY no está configurada para el chat de emergencias.")

        hist_str = ""
        if conversation_history:
            hist_str = "\n".join([f"{m.role.upper()}: {m.content}" for m in conversation_history[-4:]])

        prompt = f"""Actúa como Asistente Veterinario de Urgencias de VetIA.
Estás orientando al tutor de una mascota en situación de urgencia.

CONTEXTO DEL CASO:
- Triage ID: {triage_id}
- Especie: {species}
- Nivel de Urgencia: {urgency}
- Diagnóstico presuntivo: {', '.join(assessment.get('presumptive_diagnosis', [])) if assessment else 'En evaluación'}
Historial previo de conversación:
{hist_str}

NUEVA CONSULTA DEL TUTOR:
"{message}"

DIRECTRICES MÉDICAS:
1. Responde de forma empática, clara y directa con pautas de soporte y primeros auxilios para {species}.
2. ESTRICTAMENTE PROHIBIDO emitir recetas farmacológicas no autorizadas ni dosificar medicamentos veterinarios.
3. ADVIERTE contundentemente que los medicamentos humanos (paracetamol, acetaminofén, ibuprofeno, aspirina) son MORTALES para perros y gatos.
4. Si el cuadro denota riesgo vital, insiste en el traslado inmediato a urgencias veterinarias.
"""

        last_error: Optional[Exception] = None
        for model_name in self._candidate_models:
            try:
                logger.info("[TRIAGE AI CHAT] Consultando chat con Gemini modelo: %s", model_name)
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2),
                )
                if response.text and response.text.strip():
                    import datetime
                    return EmergencyChatResponse(
                        triage_id=triage_id,
                        reply=response.text.strip(),
                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    )
            except Exception as exc:
                last_error = exc
                logger.warning("[TRIAGE AI CHAT WARNING] Fallo chat con %s: %s", model_name, exc)

        raise RuntimeError(f"Error en chat de emergencias de Gemini: {str(last_error)}")

    async def emergency_chat(
        self,
        triage_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> EmergencyChatResponse:
        """Asynchronously dispatches emergency chat to thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_emergency_chat,
            triage_id,
            message,
            context,
            conversation_history,
        )
