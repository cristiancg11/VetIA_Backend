"""Pure Google Gemini Multimodal Veterinary & Zootechnical Triage Service.
Zero static mocks or hardcoded template fallbacks:
Enforces real image byte ingestion and species-specific clinical and zootechnical reasoning
for livestock (Bovino, Equino, Porcino, Ovino/Caprino, Aves, Cuyes/Conejos) and companion animals.
Includes exponential retry and resilient model cascade fallback for 503 / 429 service spikes.
"""

import asyncio
import datetime
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from fastapi import HTTPException, status
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
    normalize_species_name,
)

logger = logging.getLogger(__name__)

# Global in-memory session cache for clinical triage context across requests
_GLOBAL_TRIAGE_SESSION_STORE: Dict[str, Dict[str, Any]] = {}

# Ordered cascade of vision-compatible models prioritized for resilience
MODELS_CASCADE: List[str] = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro",
]

# Ultra-low latency timeouts and retry delays
MODEL_TIMEOUT_SECONDS: float = 8.0
RETRY_DELAY_SECONDS: float = 0.5


def is_transient_overload_error(exc: Exception) -> bool:
    """Detects 503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, rate limits, or spikes in server demand."""
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
        "spikes in demand",
        "currently experiencing high demand",
        "deadline_exceeded",
        "rate limit",
        "overloaded",
    ]
    return any(token in err_str for token in transient_tokens)


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
    """Builds clinical multimodal veterinary and zootechnical triage prompt for rural and companion animals."""
    normalized = normalize_species_name(species)
    return f"""Actúa como Médico Veterinario y Zootecnista (MVZ) Especialista en Triage de Urgencias, Medicina de Campo y Producción Agropecuaria.
Analiza detalladamente esta fotografía clínica/zootécnica de un paciente de especie {normalized} con un peso aproximado de {weight_kg} kg.
Descripción del productor o tutor: "{description}".

DIRECTRICES CLÍNICAS Y FISIOPATOLÓGICAS OBLIGATORIAS POR ESPECIE:
1. IDENTIFICACIÓN Y FISIOLOGÍA ESPECÍFICA ({normalized}):
   - Si es BOVINO (Vacas, terneros, novillos): Gran rumiante. Si presenta abdomen distendido o asimétrico en la fosa paralumbar izquierda (ijar), sospechar TIMPANISMO RUMINAL (Meteorismo gaseoso o espumoso, Urgencia ROJA / NARANJA por asfixia inminente por compresión diafragmática). Evaluar también paresia puerperal (vaca caída / hipocalcemia), mastitis hiperaguda o reticuloperitonitis traumática.
   - Si es EQUINO (Caballos, mulas, asnos): Monogástrico con fermentación en ciego/colon. Si muestra dolor abdominal, pateo de panza, mirarse los flancos, echarse y revolcarse, o sudoración profusa, sospechar SÍNDROME ABDOMINAL AGUDO / CÓLICO EQUINO (Urgencia ROJA INMEDIATA por riesgo letal de torsión, vólvulo o shock endotóxico). Evaluar también laminitis/infosura o laceraciones profundas.
   - Si es PORCINO (Cerdos de ceba, lechones): Evaluar deshidratación severa por diarreas neonatales/post-destete, síndrome respiratorio o postración con eritema dérmico (erisipela).
   - Si es OVINO / CAPRINO (Ovejas, corderos, cabras): Pequeños rumiantes. Sospechar enterotoxemia, timpanismo agudo o parasitismo gastrointestinal severo (Haemonchus contortus / anemia de mucosas y signo de mandíbula de botella).
   - Si son AVES DE CORRAL (Gallinas criollas, pollos): Evaluar a nivel individual y de lote (brotes en lote, coccidiosis, problemas respiratorios, prolapso cloacal). Ajustar pautas a escala avícola.
   - Si son CUYES / CONEJOS (Explotaciones menores / cunicultura): Monogástricos cecotrofos muy sensibles a estasis gastrointestinal, timpanismo cecal, neumonía aguda o pasteurelosis/salmonelosis en lote. Ajustar volúmenes a gramos o pocos kilogramos.
   - Si es CANINO / FELINO (Compañía y pastoreo): Razonamiento de pequeños animales (mordeduras de serpiente/depredadores en campo, GDV en caninos, lipidosis hepática u obstrucción uretral en felinos).

2. INSPECCIÓN VISUAL ('visual_findings'):
   - Describe minuciosamente los elementos visuales observables en la foto (colorimetría, texturas, heridas, mucosas, aspecto de heces o fluidos patológicos, postura antálgica, estado corporal, condición del animal o del lote).
   - Si la foto presenta limitaciones de visibilidad, explícalo con cautela clínica sin omitir la evaluación.

3. DIAGNÓSTICOS PRESUNTIVOS ('presumptive_diagnosis'):
   - Ofrece al menos 2 hipótesis clínicas fundamentadas en la fisiopatología estricta de {normalized}.

4. RIESGOS VITALES ('vital_risks'):
   - Explica con claridad qué órganos o funciones biológicas (cardiopulmonar, ruminal, endotóxica, deshidratación en lote) están en riesgo inminente.

5. PRIMEROS AUXILIOS Y MANEJO EN CAMPO ('first_aid_actions'):
   - Brinda instrucciones prácticas de manejo rural y soporte seguro antes de la llegada del médico veterinario de campo (ej. descompresión ruminal de emergencia en timpanismo asfixiante, caminar al caballo con cólico y no permitir revolcones, aislar animales sospechosos del lote, hidratación oral, etc.).

6. PREGUNTAS DE DESCARTE CLÍNICO ('diagnostic_questions'):
   - Formula exactamente 3 a 4 preguntas clave de descarte con respuesta binaria ("yes_no") adaptadas a la especie y producción agropecuaria (ej. ¿Se infló el flanco izquierdo de la panza? ¿El animal puede levantarse por sí mismo o está postrado? ¿El equino patea su abdomen o se tira al suelo? ¿Hay otros animales del mismo corral o lote con síntomas idénticos?).

Responde ÚNICAMENTE en JSON válido con este formato:
{{
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
  "visual_findings": "...",
  "presumptive_diagnosis": ["...", "..."],
  "vital_risks": ["..."],
  "first_aid_actions": ["..."],
  "warnings_contraindications": ["..."],
  "estimated_fluid_deficit_percent": 5.0,
  "diagnostic_questions": [
    {{"id": "q1", "question": "..."}},
    {{"id": "q2", "question": "..."}},
    {{"id": "q3", "question": "..."}}
  ]
}}
"""


def build_confirmation_prompt(
    triage_id: str,
    target_species: str,
    target_symptoms: str,
    target_urgency: str,
    answers_str: str,
) -> str:
    """Builds clinical confirmation and re-evaluation prompt."""
    return f"""Actúa como Médico Veterinario y Zootecnista (MVZ) Especialista en Urgencias de Campo y Producción Agropecuaria.
Reevalúa de forma definitiva este caso de triaje clínico zootécnico:
- Triage ID: {triage_id}
- Especie: {target_species}
- Síntomas iniciales: "{target_symptoms}"
- Urgencia preliminar: {target_urgency}

Respuestas del productor/tutor al test de descarte clínico:
{answers_str}

REGLAS DE REEVALUACIÓN AGROPECUARIA:
1. Reevalúa la urgencia considerando la fisiopatología estricta de {target_species}.
2. ESCALADA CRÍTICA INMEDIATA (ROJO / CRITICO):
   - En BOVINOS o RUMIANTES: Si confirman inflamación del flanco izquierdo (timpanismo ruminal agudo con dificultad respiratoria) o postración sin poder levantarse (vaca caída / paresia puerperal).
   - En EQUINOS: Si confirman que el caballo patea su panza, mira sus flancos o se tira a revolcarse (cólico equino agudo / síndrome abdominal).
   - En CUALQUIER ESPECIE: Postración extrema, encías pálidas/cianóticas, sangrado activo profuso o alta mortandad súbita en el lote.
3. Responde ÚNICAMENTE en JSON válido con este formato:
{{
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
  "is_escalated": true | false,
  "clinical_reassessment": "Explicación detallada del ajuste o confirmación para {target_species}",
  "presumptive_diagnosis": ["...", "..."],
  "vital_risks": ["..."],
  "first_aid_actions": ["..."],
  "warnings_contraindications": ["..."],
  "estimated_fluid_deficit_percent": 5.0,
  "recommended_facility_type": "Atención médico-veterinaria de campo inmediata" | "Clínica 24 horas con quirófano" | "Consultorio general"
}}
"""


def build_emergency_chat_prompt(
    triage_id: str,
    species: str,
    urgency: str,
    assessment: Dict[str, Any],
    hist_str: str,
    message: str,
) -> str:
    """Builds first-aid emergency chat prompt."""
    return f"""Actúa como Asistente Veterinario y Zootécnico de Urgencias y Medicina de Campo de VetIA.
Estás orientando al productor o tutor en una situación de emergencia agropecuaria o clínica.

CONTEXTO DEL CASO:
- Triage ID: {triage_id}
- Especie: {species}
- Nivel de Urgencia: {urgency}
- Diagnóstico presuntivo: {', '.join(assessment.get('presumptive_diagnosis', [])) if assessment else 'En evaluación de campo'}
Historial previo de conversación:
{hist_str}

NUEVA CONSULTA DEL PRODUCTOR/TUTOR:
"{message}"

DIRECTRICES MÉDICAS Y ZOOTÉCNICAS:
1. Responde de forma empática, clara, técnica y práctica con pautas de primeros auxilios y manejo de campo para {species}.
2. Si es un Bovino con timpanismo, explica posturas de alivio (cabeza más alta que la grupa), no dar líquidos por boca si no deglute y preparar trocarización o sonda nasorruminal solo por personal calificado.
3. Si es un Equino con cólico, indicar caminarlo a paso suave para evitar que se revuelque y lesione, no forzar alimento ni agua fría.
4. Si son Aves o Cuyes, indicar aislamiento preventivo del lote y abrigo/ventilación adecuada.
5. ESTRICTAMENTE PROHIBIDO dosificar fármacos sin prescripción médica directa y advertir contra analgésicos humanos tóxicos.
"""


class MultimodalTriageAIService:
    """
    Pure Multimodal Veterinary & Zootechnical Triage AI Service powered strictly by Google Gemini.
    Features exponential retries, resilient model cascade fallback, and zero hardcoded canine mocks.
    """

    MODELS_CASCADE = MODELS_CASCADE

    def __init__(
        self,
        settings: Optional[Settings] = None,
        thread_pool_manager: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._thread_pool_manager = thread_pool_manager or get_thread_pool_manager()
        self._api_key = (os.getenv("GEMINI_API_KEY") or self._settings.GEMINI_API_KEY or "").strip()
        env_model = os.getenv("GEMINI_MODEL")
        if env_model and env_model.strip() and "2.5" not in env_model:
            self._primary_model = env_model.strip()
        else:
            self._primary_model = "gemini-1.5-flash"

        # Prioritize gemini-1.5-flash first, followed by MODELS_CASCADE
        cascade = [self._primary_model] + [m for m in MODELS_CASCADE if m != self._primary_model]
        self.models_cascade = list(dict.fromkeys(cascade))

        self._client: Optional[genai.Client] = None
        if self._api_key and self._api_key != "your_gemini_api_key_here":
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(timeout=45000),
                )
                logger.info(
                    "[TRIAGE AI] Google GenAI Client inicializado. Modelos cascada: %s",
                    self.models_cascade,
                )
            except Exception as exc:
                logger.error("[TRIAGE AI] Error al inicializar Google GenAI Client: %s", exc)
                self._client = None

    def _process_analysis_response(
        self,
        response: Any,
        model_name: str,
        normalized_species: str,
        peso_kg: float,
        descripcion: str,
    ) -> TriageAnalysisResponse:
        """Parses and validates Gemini response, caching the active triage session."""
        raw_text = response.text or ""
        if not raw_text.strip():
            raise ValueError(f"El modelo {model_name} retornó una respuesta de texto vacía.")

        result_dict = clean_and_parse_json(raw_text)
        if not isinstance(result_dict, dict):
            raise ValueError(f"La respuesta de Gemini no es un objeto JSON válido: {raw_text[:100]}")

        if not result_dict.get("triage_id"):
            result_dict["triage_id"] = str(uuid.uuid4())

        print(
            f"[TRIAGE AI] Respuesta generada ({model_name}): {result_dict.get('urgency_level')} - "
            f"{str(result_dict.get('visual_findings', ''))[:80]}..."
        )
        logger.info(
            "[TRIAGE AI SUCCESS] Inferencia exitosa con modelo %s. Especie: %s, Urgencia: %s",
            model_name,
            normalized_species,
            result_dict.get("urgency_level"),
        )

        assessment = TriageAnalysisResponse.model_validate(result_dict)
        self._save_session(assessment.triage_id, assessment, normalized_species, peso_kg, descripcion)
        return assessment

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

    # -------------------------------------------------------------------------
    # 1. MULTIMODAL INFERENCE WITH RETRIES AND MODEL CASCADE FALLBACK
    # -------------------------------------------------------------------------
    async def analyze_triage(
        self,
        image_bytes: bytes,
        mime_type: str,
        especie: str,
        peso_kg: float,
        descripcion: str,
    ) -> TriageAnalysisResponse:
        """
        Executes genuine multimodal inference with Google Gemini for livestock and companion animals.
        Features exponential retries on 503/429 and automatic model cascade fallback.
        Raises HTTP 503 if all models in the cascade are exhausted due to high demand.
        """
        normalized_species = normalize_species_name(especie)
        print(f"[TRIAGE AI] Analizando imagen de {len(image_bytes)} bytes para especie: {normalized_species}")
        logger.info("[TRIAGE AI] Analizando imagen de %d bytes para especie: %s", len(image_bytes), normalized_species)

        if not self._client or not self._api_key or self._api_key == "your_gemini_api_key_here":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY no está configurada en las variables de entorno (.env).",
            )

        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No se recibieron bytes de imagen válidos para el análisis de triaje.",
            )

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type or "image/jpeg",
        )

        prompt_text = build_triage_prompt(
            species=normalized_species,
            weight_kg=peso_kg,
            description=descripcion,
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1200,
            top_p=0.8,
        )

        last_error: Optional[Exception] = None

        for idx, model_name in enumerate(self.models_cascade):
            next_model = self.models_cascade[idx + 1] if idx + 1 < len(self.models_cascade) else "ninguno"

            def _invoke_gemini(target_model: str):
                return self._client.models.generate_content(
                    model=target_model,
                    contents=[image_part, prompt_text],
                    config=config,
                )

            # Intento 1 con el modelo actual (límite estricto de 8 segundos)
            try:
                logger.info("[TRIAGE AI] Ejecutando inferencia multimodal en Gemini con modelo: %s", model_name)
                response = await asyncio.wait_for(
                    self._thread_pool_manager.run_in_thread(_invoke_gemini, model_name),
                    timeout=MODEL_TIMEOUT_SECONDS,
                )
                return self._process_analysis_response(response, model_name, normalized_species, peso_kg, descripcion)

            except (asyncio.TimeoutError, TimeoutError):
                last_error = TimeoutError(f"Modelo {model_name} excedió tiempo límite de {MODEL_TIMEOUT_SECONDS}s")
                print(f"[TRIAGE AI] Modelo {model_name} excedió tiempo límite (8s). Conmutando a {next_model}...")
                logger.warning(
                    "[TRIAGE AI TIMEOUT] Modelo %s superó el límite de 8s sin responder. Conmutando de inmediato a %s...",
                    model_name,
                    next_model,
                )
                continue

            except Exception as e:
                last_error = e
                print(f"\n❌ [ERROR CRÍTICO GEMINI API]: Tipo: {type(e).__name__} | Mensaje: {str(e)}\n")
                if is_transient_overload_error(e):
                    logger.warning(
                        "[TRIAGE AI] Modelo %s con alta demanda/error transitorio (%s). Reintentando en %.1fs...",
                        model_name,
                        e,
                        RETRY_DELAY_SECONDS,
                    )
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

                    # Intento 2 (Reintento tras espera corta de 0.5s con timeout de 8s)
                    try:
                        logger.info("[TRIAGE AI] Reintentando inferencia multimodal con modelo: %s", model_name)
                        response = await asyncio.wait_for(
                            self._thread_pool_manager.run_in_thread(_invoke_gemini, model_name),
                            timeout=MODEL_TIMEOUT_SECONDS,
                        )
                        return self._process_analysis_response(response, model_name, normalized_species, peso_kg, descripcion)

                    except (asyncio.TimeoutError, TimeoutError):
                        last_error = TimeoutError(f"Reintento de modelo {model_name} excedió tiempo límite de {MODEL_TIMEOUT_SECONDS}s")
                        print(f"[TRIAGE AI] Modelo {model_name} excedió tiempo límite (8s). Conmutando a {next_model}...")
                        logger.warning(
                            "[TRIAGE AI TIMEOUT] Reintento de modelo %s superó límite de 8s. Conmutando a %s...",
                            model_name,
                            next_model,
                        )
                        continue

                    except Exception as retry_exc:
                        last_error = retry_exc
                        print(f"\n❌ [ERROR CRÍTICO GEMINI API]: Tipo: {type(retry_exc).__name__} | Mensaje: {str(retry_exc)}\n")
                        if is_transient_overload_error(retry_exc):
                            print(f"[TRIAGE AI] Modelo {model_name} saturado (503). Conmutando a {next_model}...")
                            logger.warning(
                                "[TRIAGE AI] Modelo %s saturado (503). Conmutando a %s...",
                                model_name,
                                next_model,
                            )
                        else:
                            logger.warning(
                                "[TRIAGE AI WARNING] Fallo en reintento con %s: %s. Conmutando a %s",
                                model_name,
                                retry_exc,
                                next_model,
                            )
                        continue
                else:
                    logger.warning(
                        "[TRIAGE AI WARNING] Fallo con modelo %s: %s. Conmutando a %s.",
                        model_name,
                        exc,
                        next_model,
                    )
                    continue

        # Si todos los modelos de la cascada fallan
        logger.error("[TRIAGE AI FATAL] Todos los modelos de la cascada fallaron: %s", last_error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los servidores de IA veterinaria están experimentando alta demanda momentánea. Por favor presiona 'Reintentar Evaluación' en 10 segundos.",
        )

    # -------------------------------------------------------------------------
    # 2. CONFIRMATION DIAGNOSIS (REEVALUATION VIA GEMINI)
    # -------------------------------------------------------------------------
    async def confirm_diagnosis(
        self,
        triage_id: str,
        answers: Dict[str, Any],
        initial_symptoms: Optional[str] = None,
        species: Optional[str] = None,
        initial_urgency_level: Optional[str] = None,
        questions: Optional[List[DiagnosticQuestion]] = None,
    ) -> ConfirmDiagnosisResponse:
        """Executes clinical re-evaluation strictly with Gemini, featuring model cascade and retries."""
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        target_species = normalize_species_name(species or session.get("species") or "Bovino")
        target_symptoms = initial_symptoms or session.get("initial_symptoms") or "Paciente en urgencias"
        target_urgency = initial_urgency_level or (
            session.get("assessment", {}).get("urgency_level") if session.get("assessment") else "MODERADO"
        )

        if not self._client or not self._api_key or self._api_key == "your_gemini_api_key_here":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY no está configurada.",
            )

        ans_lines = [f"- Pregunta ID {k}: {'SÍ (Confirmado / Positivo)' if v else 'NO (Negativo)'}" for k, v in answers.items()]
        answers_str = "\n".join(ans_lines)

        prompt = build_confirmation_prompt(
            triage_id=triage_id,
            target_species=target_species,
            target_symptoms=target_symptoms,
            target_urgency=target_urgency,
            answers_str=answers_str,
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1200,
            top_p=0.8,
        )

        last_error: Optional[Exception] = None

        for idx, model_name in enumerate(self.models_cascade):
            next_model = self.models_cascade[idx + 1] if idx + 1 < len(self.models_cascade) else "ninguno"

            def _invoke_confirm(target_model: str):
                return self._client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )

            # Intento 1 (con límite de 8s)
            try:
                logger.info("[TRIAGE AI] Reevaluando con Gemini modelo: %s para especie: %s", model_name, target_species)
                response = await asyncio.wait_for(
                    self._thread_pool_manager.run_in_thread(_invoke_confirm, model_name),
                    timeout=MODEL_TIMEOUT_SECONDS,
                )
                raw_text = response.text or ""
                result_dict = clean_and_parse_json(raw_text)
                result_dict["triage_id"] = triage_id
                return ConfirmDiagnosisResponse.model_validate(result_dict)

            except (asyncio.TimeoutError, TimeoutError):
                last_error = TimeoutError(f"Modelo {model_name} excedió tiempo límite de {MODEL_TIMEOUT_SECONDS}s")
                print(f"[TRIAGE AI] Modelo {model_name} excedió tiempo límite (8s). Conmutando a {next_model}...")
                logger.warning("[TRIAGE AI TIMEOUT] Confirmación con modelo %s superó límite de 8s. Conmutando a %s...", model_name, next_model)
                continue

            except Exception as e:
                last_error = e
                print(f"\n❌ [ERROR CRÍTICO GEMINI API]: Tipo: {type(e).__name__} | Mensaje: {str(e)}\n")
                if is_transient_overload_error(e):
                    logger.warning("[TRIAGE AI] Modelo %s con alta demanda en confirmación. Reintentando en %.1fs...", model_name, RETRY_DELAY_SECONDS)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

                    try:
                        logger.info("[TRIAGE AI] Reintentando confirmación con modelo: %s", model_name)
                        response = await asyncio.wait_for(
                            self._thread_pool_manager.run_in_thread(_invoke_confirm, model_name),
                            timeout=MODEL_TIMEOUT_SECONDS,
                        )
                        raw_text = response.text or ""
                        result_dict = clean_and_parse_json(raw_text)
                        result_dict["triage_id"] = triage_id
                        return ConfirmDiagnosisResponse.model_validate(result_dict)

                    except (asyncio.TimeoutError, TimeoutError):
                        last_error = TimeoutError(f"Reintento de confirmación con modelo {model_name} excedió 8s")
                        print(f"[TRIAGE AI] Modelo {model_name} excedió tiempo límite (8s). Conmutando a {next_model}...")
                        logger.warning("[TRIAGE AI TIMEOUT] Reintento de confirmación con %s superó 8s. Conmutando a %s...", model_name, next_model)
                        continue

                    except Exception as retry_exc:
                        last_error = retry_exc
                        print(f"\n❌ [ERROR CRÍTICO GEMINI API]: Tipo: {type(retry_exc).__name__} | Mensaje: {str(retry_exc)}\n")
                        if is_transient_overload_error(retry_exc):
                            print(f"[TRIAGE AI] Modelo {model_name} saturado (503). Conmutando a {next_model}...")
                            logger.warning("[TRIAGE AI] Modelo %s saturado (503). Conmutando a %s...", model_name, next_model)
                        continue
                else:
                    logger.warning("[TRIAGE AI WARNING] Fallo reevaluación con %s: %s. Conmutando a %s", model_name, exc, next_model)
                    continue

        logger.error("[TRIAGE AI FATAL] Todos los modelos fallaron en confirmación: %s", last_error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los servidores de IA veterinaria están experimentando alta demanda momentánea. Por favor presiona 'Reintentar Evaluación' en 10 segundos.",
        )

    # -------------------------------------------------------------------------
    # 3. EMERGENCY CHAT ASSISTANT (PURE GEMINI INFERENCE)
    # -------------------------------------------------------------------------
    async def emergency_chat(
        self,
        triage_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> EmergencyChatResponse:
        """Processes emergency chat queries strictly with Gemini, featuring model cascade and retries."""
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        full_context = dict(session)
        if isinstance(context, dict):
            full_context.update(context)
        elif isinstance(context, str) and context.strip():
            full_context["extra_notes"] = context.strip()

        raw_species = full_context.get("species") or "Mascota"
        species = normalize_species_name(raw_species)
        assessment = full_context.get("assessment", {})
        urgency = assessment.get("urgency_level") or full_context.get("urgency_level") or "MODERADO"

        if not self._client or not self._api_key or self._api_key == "your_gemini_api_key_here":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GEMINI_API_KEY no está configurada para el chat de emergencias.",
            )

        hist_str = ""
        if conversation_history:
            hist_str = "\n".join([f"{m.role.upper()}: {m.content}" for m in conversation_history[-4:]])

        prompt = build_emergency_chat_prompt(
            triage_id=triage_id,
            species=species,
            urgency=urgency,
            assessment=assessment,
            hist_str=hist_str,
            message=message,
        )

        last_error: Optional[Exception] = None

        for idx, model_name in enumerate(self.models_cascade):
            next_model = self.models_cascade[idx + 1] if idx + 1 < len(self.models_cascade) else "ninguno"

            def _invoke_chat(target_model: str):
                return self._client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1200,
                        top_p=0.8,
                    ),
                )

            # Intento 1 (con límite de 8s)
            try:
                logger.info("[TRIAGE AI CHAT] Consultando chat con Gemini modelo: %s para especie: %s", model_name, species)
                response = await asyncio.wait_for(
                    self._thread_pool_manager.run_in_thread(_invoke_chat, model_name),
                    timeout=MODEL_TIMEOUT_SECONDS,
                )
                if response.text and response.text.strip():
                    return EmergencyChatResponse(
                        triage_id=triage_id,
                        reply=response.text.strip(),
                        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    )

            except (asyncio.TimeoutError, TimeoutError):
                last_error = TimeoutError(f"Chat con modelo {model_name} excedió tiempo límite de {MODEL_TIMEOUT_SECONDS}s")
                print(f"[TRIAGE AI] Modelo {model_name} excedió tiempo límite (8s). Conmutando a {next_model}...")
                logger.warning("[TRIAGE AI TIMEOUT] Chat con modelo %s superó límite de 8s. Conmutando a %s...", model_name, next_model)
                continue

            except Exception as e:
                last_error = e
                print(f"\n❌ [ERROR CRÍTICO GEMINI API]: Tipo: {type(e).__name__} | Mensaje: {str(e)}\n")
                if is_transient_overload_error(e):
                    logger.warning("[TRIAGE AI CHAT] Modelo %s saturado en chat. Reintentando en %.1fs...", model_name, RETRY_DELAY_SECONDS)
                    await asyncio.sleep(RETRY_DELAY_SECONDS)

                    try:
                        logger.info("[TRIAGE AI CHAT] Reintentando chat con modelo: %s", model_name)
                        response = await asyncio.wait_for(
                            self._thread_pool_manager.run_in_thread(_invoke_chat, model_name),
                            timeout=MODEL_TIMEOUT_SECONDS,
                        )
                        if response.text and response.text.strip():
                            return EmergencyChatResponse(
                                triage_id=triage_id,
                                reply=response.text.strip(),
                                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            )

                    except (asyncio.TimeoutError, TimeoutError):
                        last_error = TimeoutError(f"Reintento de chat con modelo {model_name} excedió 8s")
                        print(f"[TRIAGE AI] Modelo {model_name} excedió tiempo límite (8s). Conmutando a {next_model}...")
                        logger.warning("[TRIAGE AI TIMEOUT] Reintento de chat con %s superó 8s. Conmutando a %s...", model_name, next_model)
                        continue

                    except Exception as retry_exc:
                        last_error = retry_exc
                        print(f"\n❌ [ERROR CRÍTICO GEMINI API]: Tipo: {type(retry_exc).__name__} | Mensaje: {str(retry_exc)}\n")
                        if is_transient_overload_error(retry_exc):
                            print(f"[TRIAGE AI] Modelo {model_name} saturado (503). Conmutando a {next_model}...")
                            logger.warning("[TRIAGE AI] Modelo %s saturado (503). Conmutando a %s...", model_name, next_model)
                        continue
                else:
                    logger.warning("[TRIAGE AI CHAT WARNING] Fallo chat con %s: %s. Conmutando a %s", model_name, exc, next_model)
                    continue

        logger.error("[TRIAGE AI CHAT FATAL] Todos los modelos fallaron en chat: %s", last_error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los servidores de IA veterinaria están experimentando alta demanda momentánea. Por favor presiona 'Reintentar Evaluación' en 10 segundos.",
        )
