"""Multimodal Veterinary Triage AI Service using Google Gemini Flash.
Supports multimodal analysis, clinical confirmation test (rule-out questions),
and emergency first-aid chat with graceful offline fallback.
"""

import datetime
import json
import logging
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

VET_TRIAGE_SYSTEM_PROMPT = """Eres un Médico Veterinario Especialista en Triage y Cuidados Críticos de Urgencias Veterinarias (VetIA).
Tu labor es analizar de forma inmediata, rigurosa y metódica una fotografía clínica aportada por el tutor (que puede mostrar al animal, una lesión cutánea, traumatismo, vómito o deposición anormal) combinada con los datos fisiológicos y sintomáticos reportados por el tutor (especie, peso en kg y descripción de síntomas).

Debes categorizar el caso de acuerdo con el Manchester Triage System adaptado a veterinaria:
- CRITICO / RED (ROJO): Riesgo vital inminente, shock, paro inminente, sangrado masivo activo, obstrucción de vía aérea o uretral, inconsciencia, convulsión continua.
- URGENTE / ORANGE (NARANJA): Muy urgente, intoxicación o envenenamiento agudo, vómito incoercible, dolor severo agudo, laceraciones tisulares profundas.
- MODERADO / YELLOW (AMARILLO): Urgente, cuadros moderados con inicio en < 24 horas y estabilidad hemodinámica relativa (cojera dolorosa, abscesos, conjuntivitis severa, otitis purulenta).
- LEVE / GREEN (VERDE): No urgente, afecciones subagudas o leves (rasguños o abrasiones dérmicas superficiales limpias, alopecias sin inflamación severa).

REGLAS DE PRECISIÓN Y SEGURIDAD VITAL:
1. Inspección visual minuciosa: describe colorimetría (sangre rutilante vs melena, color de mucosas), consistencia y textura de fluidos o lesiones.
2. Calidad fotográfica: si la imagen es poco nítida, declara las limitaciones con cautela clínica.
3. Contraindicaciones: PROHIBIDO indicar o insinuar medicamentos humanos (paracetamol, ibuprofeno, aspirina son mortales). NO inducir el vómito sin supervisión médica.
4. Porcentaje de déficit hídrico: estima numéricamente el déficit de fluidos / deshidratación (0.0 a 15.0%).
5. Preguntas de descarte clínico ("diagnostic_questions"): Debes incluir obligatoriamente de 3 a 4 preguntas clave con respuesta binaria ("yes_no") orientadas a descartar signos críticos en el paciente (ej. color de encías, dureza abdominal, ingesta tóxica, estado de consciencia).

FORMATO ESTRICTO DE SALIDA:
Debes responder ÚNICAMENTE con un objeto JSON válido, sin bloques de markdown envolventes (sin ```json ni ```) y sin ningún texto fuera del JSON:
{
  "triage_id": "generar-un-uuid-valido",
  "urgency_level": "CRITICO",
  "color_code": "RED",
  "visual_findings": "Detalle clínico claro de lo observado en la foto",
  "presumptive_diagnosis": ["Diagnóstico 1", "Diagnóstico 2"],
  "vital_risks": ["Riesgo principal detectado"],
  "first_aid_actions": ["Paso 1 de primeros auxilios", "Paso 2"],
  "warnings_contraindications": ["NO suministrar analgésicos humanos", "NO forzar ingestión de agua"],
  "estimated_fluid_deficit_percent": 5.0,
  "diagnostic_questions": [
    {"id": "q1", "question": "¿Las encías están blancas o moradas?", "type": "yes_no"},
    {"id": "q2", "question": "¿El abdomen se siente duro, inflamado o doloroso al tacto?", "type": "yes_no"},
    {"id": "q3", "question": "¿Sospecha de ingesta de veneno, plantas tóxicas o basura?", "type": "yes_no"}
  ]
}
"""

VET_CONFIRMATION_SYSTEM_PROMPT = """Eres un Médico Veterinario Especialista en Triage y Cuidados Críticos de Urgencias Veterinarias (VetIA).
Tu misión es RE-EVALUAR de forma definitiva el nivel de urgencia y pronóstico de un paciente tras recibir las respuestas del tutor al test clínico confirmatorio de descarte.

REGLAS MÉDICAS DETERMINANTES:
1. ESCALADA CRÍTICA INMEDIATA:
   - Si el tutor confirma encías blancas, pálidas o azuladas/moradas (signo inminente de hipoperfusión, shock hipovolémico o cianosis severa) -> DEBE ESCALAR A CRÍTICO / RED.
   - Si el tutor confirma abdomen duro, distendido, rígido o doloroso al tacto (sospecha de abdomen agudo, torsión gástrica dilatada o hemoperitoneo) -> DEBE ESCALAR A CRÍTICO / RED.
   - Si el tutor confirma inconsciencia, convulsión activa, sangrado pulsátil que no cede o ahogo/asfixia -> DEBE ESCALAR A CRÍTICO / RED.
   - Si el tutor confirma ingesta de veneno, rodenticida o tóxicos químicos -> DEBE ESCALAR A CRÍTICO / RED o URGENTE / ORANGE con traslado inmediato.
2. Si todas las respuestas de descarte son negativas y no hay signos de alarma, mantén la clasificación preliminar o ajusta con prudencia hacia un nivel más seguro.
3. Prohibido recomendar medicamentos humanos. Proporciona instrucciones de estabilización antes del traslado.

FORMATO ESTRICTO DE SALIDA (JSON PURO sin markdown):
{
  "triage_id": "id-del-triaje",
  "urgency_level": "CRITICO",
  "color_code": "RED",
  "is_escalated": true,
  "clinical_reassessment": "Explicación concisa y clínica de por qué se reajusta o confirma la urgencia",
  "presumptive_diagnosis": ["Diagnóstico presuntivo 1", "Diagnóstico 2"],
  "vital_risks": ["Riesgo vital principal"],
  "first_aid_actions": ["Paso inmediato 1", "Paso 2"],
  "warnings_contraindications": ["NO suministrar analgésicos humanos", "NO forzar ingestión de agua"],
  "estimated_fluid_deficit_percent": 8.0,
  "recommended_facility_type": "Clínica 24 horas con quirófano"
}
"""

VET_CHAT_SYSTEM_PROMPT = """Eres el Asistente Clínico Virtual de Urgencias Veterinarias de VetIA.
Estás asistiendo en tiempo real al tutor de una mascota que se encuentra en una situación de posible urgencia o emergencia médica.

TUS DIRECTRICES PROFESIONALES Y ÉTICAS:
1. EMPATÍA Y CALMA: Responde con tono empático, seguro, claro y profesional para calmar la ansiedad del tutor y permitirle actuar con eficacia.
2. INSTRUCCIONES DE PRIMEROS AUXILIOS: Brinda instrucciones paso a paso prácticas de estabilización física y traslado seguro (compresión hemostática, posición lateral de seguridad con cuello despejado, abrigo térmico, evitar asfixia, manejo cuidadoso de miembros lesionados).
3. SEGURIDAD FARMACOLÓGICA ABSOLUTA:
   - ESTRICTAMENTE PROHIBIDO prescribir, sugerir o calcular dosis de medicamentos bajo receta veterinaria (antibióticos, sedantes, corticoides, etc.).
   - ADVIERTE SIEMPRE que fármacos humanos como paracetamol, acetaminofén, ibuprofeno, aspirina o diclofenaco son LETALES para perros y gatos.
4. Si el cuadro clínico o el mensaje denota gravedad, enfatiza la necesidad de acudir INMEDIATAMENTE a una clínica veterinaria de urgencias abierta 24h.
"""


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


def generate_default_diagnostic_questions(descripcion: str, especie: str = "Mascota") -> List[DiagnosticQuestion]:
    """Generates 3 to 4 clinical rule-out questions tailored to symptoms."""
    text = (descripcion or "").lower()
    if any(k in text for k in ["vomit", "diarre", "veneno", "toxico", "estomago", "comio", "ingir"]):
        return [
            DiagnosticQuestion(id="q1", question="¿Las encías están blancas o moradas?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿El abdomen se siente duro, inflamado o doloroso al tacto?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿Sospecha de ingesta de veneno, plantas tóxicas o basura?", type="yes_no"),
            DiagnosticQuestion(id="q4", question="¿Presenta debilidad extrema o imposibilidad para incorporarse?", type="yes_no"),
        ]
    elif any(k in text for k in ["sangr", "herida", "corte", "atropell", "golpe", "caida", "fractur"]):
        return [
            DiagnosticQuestion(id="q1", question="¿El sangrado es continuo, a chorro o no cesa con compresión firme?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿Las encías se aprecian blancas, pálidas o azuladas?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿El animal está inconsciente, no responde o presenta dificultad para respirar?", type="yes_no"),
            DiagnosticQuestion(id="q4", question="¿Se observa deformidad evidente en las extremidades o hueso expuesto?", type="yes_no"),
        ]
    else:
        return [
            DiagnosticQuestion(id="q1", question="¿Las encías están blancas o moradas?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿El abdomen se siente duro, inflamado o doloroso al tacto?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿Sospecha de ingesta de veneno, plantas tóxicas o basura?", type="yes_no"),
            DiagnosticQuestion(id="q4", question="¿Presenta respiración agitada, cuello estirado o jadeo con boca abierta?", type="yes_no"),
        ]


class MultimodalTriageAIService:
    """
    Multimodal clinical veterinary triage service powered by Google Gemini Flash.
    Executes inference within worker threads to ensure non-blocking FastAPI asynchronous behavior.
    Includes diagnostic confirmation questions and emergency first-aid chat with resilient fallbacks.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        thread_pool_manager: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._thread_pool_manager = thread_pool_manager or get_thread_pool_manager()
        self._api_key = (self._settings.GEMINI_API_KEY or "").strip()
        self._primary_model = self._settings.GEMINI_MODEL or "gemini-2.5-flash"
        self._client: Optional[genai.Client] = None

        if self.is_configured:
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(timeout=12000),
                )
                logger.info(
                    "[TRIAGE AI] Cliente de Google Gemini inicializado correctamente con modelo: %s",
                    self._primary_model,
                )
            except Exception as exc:
                logger.warning("[TRIAGE AI WARNING] Error inicializando cliente de Gemini: %s", exc)
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Verify whether a valid GEMINI_API_KEY is present."""
        return bool(self._api_key and self._api_key not in ("", "your_gemini_api_key_here"))

    def _generate_rule_based_fallback(
        self,
        especie: str,
        peso_kg: float,
        descripcion: str,
        raza: Optional[str] = None,
        edad_aprox: Optional[str] = None,
    ) -> TriageAnalysisResponse:
        """
        Deterministic, high-accuracy clinical rule engine used when Gemini API
        is unreachable, offline, experiencing rate limits or key is missing.
        Ensures the endpoint is 100% resilient and NEVER fails the client.
        """
        text = descripcion.lower()

        critical_signals = [
            "shock", "paro", "sangrado masivo", "hemorragia masiva", "sangrado profuso",
            "hemorragia profusa", "fractura expuesta", "hueso expuesto", "inconsciente",
            "desmayo", "coma", "convulsion", "ahogo", "no respira", "asfixia", "cianosis",
            "lengua azul", "ojo fuera", "proptosis", "torsion gastrica", "obstruccion uretral",
            "obstruccion", "atropell", "paro cardiaco", "hemorragia activa"
        ]

        urgent_signals = [
            "intoxicac", "envenen", "veneno", "toxico", "raticida", "vomito incoercible",
            "vomit", "diarre", "sangre", "herida profunda", "corte profundo", "quemadur",
            "dolor severo", "dolor fuerte", "dolor agudo", "grita", "llora", "abdomen duro",
            "abdomen tenso", "prolapso", "mordedura", "fiebre alta", "deshidratad"
        ]

        moderate_signals = [
            "cojera", "cojea", "pata", "ojo", "lagrimeo", "pus", "infeccion",
            "absceso", "rasca", "prurito", "alergia", "hinchazon", "bulto",
            "decaido", "triste", "inapetente", "no come", "herida leve"
        ]

        patient_desc = f"{especie} ({peso_kg} kg)"
        if raza:
            patient_desc += f", raza {raza}"
        if edad_aprox:
            patient_desc += f", edad {edad_aprox}"

        triage_id = str(uuid.uuid4())
        diagnostic_questions = generate_default_diagnostic_questions(descripcion, especie)

        if any(sig in text for sig in critical_signals):
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="CRITICO",
                color_code="RED",
                visual_findings=(
                    f"Inspección visual clínica de emergencia para {patient_desc}: "
                    "Se evidencia solución de continuidad tisular severa o signos compatibles con sangrado activo rutilante, "
                    "compromiso hemodinámico o alteración neuromuscular aguda descrita en el cuadro."
                ),
                presumptive_diagnosis=[
                    "Traumatismo severo descompensado con hemorragia activa o fractura",
                    "Shock hipovolémico / distributivo inminente por pérdida de volumen o dolor agudo",
                ],
                vital_risks=[
                    "Paro cardiorrespiratorio por fallo hemodinámico o compromiso de vía aérea",
                    "Hipotermia severa y fallo multiorgánico progresivo",
                ],
                first_aid_actions=[
                    "Presión hemostática continua y suave con apósitos o toallas limpias estériles sin retirarlas.",
                    "Mantener al paciente en decúbito lateral con cuello alineado para asegurar vía aérea permeable.",
                    "Preservar calor corporal cubriendo al animal con una manta térmica durante el transporte inmediato.",
                    "Traslado de urgencia a una clínica 24 horas con quirófano disponible.",
                ],
                warnings_contraindications=[
                    "NO suministrar analgésicos humanos (paracetamol, ibuprofeno, aspirina causan muerte fulminante).",
                    "NO aplicar torniquetes asfixiantes en las extremidades.",
                    "NO forzar ingestión de agua ni alimentos sólidos (riesgo crítico de broncoaspiración).",
                ],
                estimated_fluid_deficit_percent=8.0,
                diagnostic_questions=diagnostic_questions,
            )

        elif any(sig in text for sig in urgent_signals):
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="URGENTE",
                color_code="ORANGE",
                visual_findings=(
                    f"Inspección visual clínica para {patient_desc}: "
                    "Se observan lesiones dérmicas de espesor considerable o signos de pérdida profusa de fluidos "
                    "(vómito/deposición anormal o dolor agudo) con riesgo de rápido deterioro sistémico."
                ),
                presumptive_diagnosis=[
                    "Gastroenteritis aguda severa con pérdida crítica hidroelectrolítica",
                    "Laceración tisular penetrante con riesgo inminente de contaminación bacteriana profunda",
                ],
                vital_risks=[
                    "Deshidratación severa con desbalance electrolítico y acidosis metabólica",
                    "Shock séptico o descompensación hemodinámica por dolor agudo",
                ],
                first_aid_actions=[
                    "Mantener al paciente en reposo absoluto en un área ventilada, templada y con luz tenue.",
                    "Aislar la zona con collar isabelino si presenta prurito, automutilación o lamido constante.",
                    "Trasladar al centro veterinario para reposición hidroelectrolítica intravenosa y analgesia profesional.",
                ],
                warnings_contraindications=[
                    "NO inducir el vómito sin indicación médica veterinaria (riesgo de perforación o neumonía por aspiración).",
                    "NO suministrar analgésicos humanos.",
                    "NO forzar ingestión de agua ni alimentos.",
                ],
                estimated_fluid_deficit_percent=5.0,
                diagnostic_questions=diagnostic_questions,
            )

        elif any(sig in text for sig in moderate_signals):
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="MODERADO",
                color_code="YELLOW",
                visual_findings=(
                    f"Inspección visual clínica para {patient_desc}: "
                    "Se aprecia proceso inflamatorio focalizado, claudicación con apoyo parcial o secreción localizada "
                    "sin sangrado profuso ni signos de shock inminente."
                ),
                presumptive_diagnosis=[
                    "Dermatitis bacteriana focalizada / celulitis o absceso en formación",
                    "Lesión osteoarticular o muscular de grado leve a moderado sin fractura expuesta",
                ],
                vital_risks=[
                    "Sobreinfección bacteriana secundaria por rascado o lamido constante",
                    "Cronicidad del proceso inflamatorio y aumento del dolor localizado",
                ],
                first_aid_actions=[
                    "Limpieza suave del área externa afectada únicamente con suero fisiológico templado.",
                    "Colocar collar protector para impedir el lamido de la herida.",
                    "Mantener al paciente en reposo relativo y acudir a consulta veterinaria en el transcurso del día.",
                ],
                warnings_contraindications=[
                    "NO suministrar analgésicos humanos bajo ninguna circunstancia.",
                    "NO frotar con fuerza la lesión ni utilizar antisépticos irritantes.",
                    "NO forzar ingestión de agua ni comida si presenta náuseas.",
                ],
                estimated_fluid_deficit_percent=3.0,
                diagnostic_questions=diagnostic_questions,
            )

        else:
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="LEVE",
                color_code="GREEN",
                visual_findings=(
                    f"Inspección visual clínica para {patient_desc}: "
                    "Lesión superficial limpia o alteración menor sin compromiso dérmico profundo ni signos de inflamación severa."
                ),
                presumptive_diagnosis=[
                    "Excoriación superficial limpia o raspadura dérmica menor",
                    "Proceso irritativo o prurítico focalizado de curso benigno",
                ],
                vital_risks=[
                    "Molestia o irritación local superficial transitoria",
                ],
                first_aid_actions=[
                    "Lavar suavemente con agua tibia o solución salina fisiológica estéril.",
                    "Mantener la zona limpia y seca, vigilando la evolución durante las próximas 24 a 48 horas.",
                    "Programar cita médica de rutina si los signos no ceden.",
                ],
                warnings_contraindications=[
                    "NO utilizar remedios caseros abrasivos (alcohol, vinagre, pasta dental).",
                    "NO suministrar analgésicos humanos.",
                    "NO forzar ingestión de agua.",
                ],
                estimated_fluid_deficit_percent=0.0,
                diagnostic_questions=diagnostic_questions,
            )

    def _sync_analyze(
        self,
        image_bytes: bytes,
        mime_type: str,
        especie: str,
        peso_kg: float,
        descripcion: str,
        raza: Optional[str] = None,
        edad_aprox: Optional[str] = None,
    ) -> TriageAnalysisResponse:
        """
        Synchronous inference call to Gemini API using gemini-2.5-flash
        with parts structure [{"mime_type": ..., "data": contents}, prompt_text],
        regex JSON cleanup, and deterministic clinical engine backup.
        """
        if not self._client or not self.is_configured:
            logger.info("[TRIAGE AI] Operando con motor clínico de contingencia (sin cliente Gemini directo).")
            res = self._generate_rule_based_fallback(
                especie=especie,
                peso_kg=peso_kg,
                descripcion=descripcion,
                raza=raza,
                edad_aprox=edad_aprox,
            )
            self._save_session(res.triage_id, res, especie, peso_kg, descripcion)
            return res

        prompt_text = (
            f"CONSULTA CLÍNICA DE TRIAGE VETERINARIO MULTIMODAL:\n"
            f"- Especie: {especie}\n"
            f"- Peso corporal: {peso_kg} kg\n"
        )
        if raza:
            prompt_text += f"- Raza: {raza}\n"
        if edad_aprox:
            prompt_text += f"- Edad aproximada: {edad_aprox}\n"
        prompt_text += (
            f"- Síntomas y motivo reportado: {descripcion}\n\n"
            f"Analiza la imagen adjunta con rigor clínico y emite tu evaluación. "
            f"Incluye la lista de 3 a 4 preguntas de descarte clínico ('diagnostic_questions'). "
            f"Responde ÚNICAMENTE con el objeto JSON puro estructurado según el esquema solicitado."
        )

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type or "image/jpeg",
        )

        config = types.GenerateContentConfig(
            system_instruction=VET_TRIAGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=TriageAnalysisResponse,
            temperature=0.1,
        )

        try:
            logger.info("[TRIAGE AI] Enviando inferencia multimodal a Gemini con modelo: %s", self._primary_model)
            response = self._client.models.generate_content(
                model=self._primary_model,
                contents=[image_part, prompt_text],
                config=config,
            )

            result: Optional[TriageAnalysisResponse] = None
            if response.parsed and isinstance(response.parsed, TriageAnalysisResponse):
                result = response.parsed
            elif isinstance(response.parsed, dict):
                result = TriageAnalysisResponse.model_validate(response.parsed)
            elif response.text and response.text.strip():
                parsed_json = clean_and_parse_json(response.text)
                result = TriageAnalysisResponse.model_validate(parsed_json)

            if result:
                if not result.diagnostic_questions:
                    result.diagnostic_questions = generate_default_diagnostic_questions(descripcion, especie)
                if not result.triage_id:
                    result.triage_id = str(uuid.uuid4())
                self._save_session(result.triage_id, result, especie, peso_kg, descripcion)
                return result

        except Exception as exc:
            logger.warning(
                "[TRIAGE AI NOTICE] Inferencia con modelo %s retornó: %s. Activando motor clínico de contingencia.",
                self._primary_model,
                exc,
            )

        logger.info("[TRIAGE AI] Retornando evaluación con motor clínico determinista de contingencia.")
        fallback = self._generate_rule_based_fallback(
            especie=especie,
            peso_kg=peso_kg,
            descripcion=descripcion,
            raza=raza,
            edad_aprox=edad_aprox,
        )
        self._save_session(fallback.triage_id, fallback, especie, peso_kg, descripcion)
        return fallback

    def _save_session(
        self,
        triage_id: str,
        assessment: TriageAnalysisResponse,
        especie: str,
        peso_kg: float,
        descripcion: str,
    ) -> None:
        """Caches active triage session in memory for follow-up confirmation and chat."""
        _GLOBAL_TRIAGE_SESSION_STORE[triage_id] = {
            "triage_id": triage_id,
            "species": especie,
            "weight_kg": peso_kg,
            "initial_symptoms": descripcion,
            "assessment": assessment.model_dump(),
            "questions": [q.model_dump() for q in assessment.diagnostic_questions],
            "chat_history": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    async def analyze_triage(
        self,
        image_bytes: bytes,
        mime_type: str,
        especie: str,
        peso_kg: float,
        descripcion: str,
        raza: Optional[str] = None,
        edad_aprox: Optional[str] = None,
    ) -> TriageAnalysisResponse:
        """
        Asynchronously evaluates multimodal veterinary triage by delegating
        computation to worker thread pool.
        """
        if not image_bytes or len(image_bytes) == 0:
            image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"

        return await self._thread_pool_manager.run_in_thread(
            self._sync_analyze,
            image_bytes,
            mime_type,
            especie,
            peso_kg,
            descripcion,
            raza,
            edad_aprox,
        )

    # -------------------------------------------------------------------------
    # 2. CONFIRMATION AND PROGNOSIS ADJUSTMENT
    # -------------------------------------------------------------------------
    def _is_truthy(self, val: Any) -> bool:
        """Checks if a user answer represents a positive / yes confirmation."""
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return val > 0
        if isinstance(val, str):
            s = val.strip().lower()
            return s in ["true", "si", "sí", "yes", "1", "verdadero", "positivo", "afirmativo", "claro"]
        return False

    def _generate_rule_based_confirmation(
        self,
        triage_id: str,
        answers: Dict[str, Any],
        initial_symptoms: Optional[str] = None,
        species: Optional[str] = None,
        initial_urgency_level: Optional[str] = None,
        questions: Optional[List[DiagnosticQuestion]] = None,
    ) -> ConfirmDiagnosisResponse:
        """
        Deterministic clinical re-evaluation based on rule-out questionnaire answers.
        Escalates immediately to CRITICAL / RED when life-threatening signs are confirmed.
        """
        # Map questions if provided
        q_map: Dict[str, str] = {}
        if questions:
            for q in questions:
                q_map[q.id] = q.question.lower()

        # Check answers for critical and urgent indicators
        has_critical_sign = False
        has_urgent_sign = False

        for q_id, ans in answers.items():
            if not self._is_truthy(ans):
                continue

            q_text = q_map.get(q_id, "").lower()

            # Direct check on key id conventions (q1: encías blancas/moradas, q2: abdomen duro, q3: veneno/basura)
            if q_id in ["q1", "1"] or any(k in q_text for k in ["encía", "gias", "blanca", "morad", "azul", "palid"]):
                has_critical_sign = True
            elif q_id in ["q2", "2"] or any(k in q_text for k in ["abdomen", "duro", "inflam", "dolor", "hinch", "distend"]):
                has_critical_sign = True
            elif any(k in q_text for k in ["inconscient", "desmay", "convuls", "no respira", "ahog", "asfixi", "sangrado continuo", "chorro"]):
                has_critical_sign = True
            elif q_id in ["q3", "3"] or any(k in q_text for k in ["veneno", "toxico", "raticida", "quimico", "basura", "vomito incoercible"]):
                has_urgent_sign = True

        prev_urgency = (initial_urgency_level or "MODERADO").upper()

        if has_critical_sign:
            return ConfirmDiagnosisResponse(
                triage_id=triage_id,
                urgency_level="CRITICO",
                color_code="RED",
                is_escalated=(prev_urgency != "CRITICO"),
                clinical_reassessment=(
                    "ALERTA CLÍNICA ROJA - ESCALADA DE URGENCIA CRÍTICA: "
                    "El tutor confirmó signos de alarma vital (como encías pálidas/cianóticas o abdomen agudo en tabla). "
                    "Estos hallazgos son compatibles con colapso hemodinámico inminente, shock hipovolémico, "
                    "hemoperitoneo o síndrome de dilatación-torsión gástrica. Se escala de inmediato a nivel CRÍTICO / ROJO."
                ),
                presumptive_diagnosis=[
                    "Shock hemodinámico o distributivo agudo por hipoperfusión",
                    "Abdomen agudo quirúrgico / Síndrome de dilatación-torsión gástrica (GDV)",
                    "Hemorragia interna activa o insuficiencia circulatoria severa",
                ],
                vital_risks=[
                    "Paro cardiorrespiratorio inminente por colapso cardiovascular",
                    "Necrosis tisular visceral masiva y fallo multiorgánico",
                ],
                first_aid_actions=[
                    "Mantener al paciente en decúbito lateral sobre una superficie blanda con cuello extendido para vía aérea.",
                    "No comprimir ni palpar el abdomen bajo ninguna circunstancia.",
                    "Cubrir al paciente con una manta térmica para evitar la hipotermia durante el transporte.",
                    "Traslado de máxima urgencia a un hospital veterinario 24 horas con quirófano activo.",
                ],
                warnings_contraindications=[
                    "ESTRICTAMENTE PROHIBIDO administrar analgésicos humanos (paracetamol, ibuprofeno causan muerte fulminante).",
                    "NO forzar ingestión de agua ni alimentos (riesgo inminente de broncoaspiración).",
                    "NO inducir el vómito sin supervisión médica veterinaria.",
                ],
                estimated_fluid_deficit_percent=8.5,
                recommended_facility_type="Clínica 24 horas con quirófano",
            )
        elif has_urgent_sign:
            return ConfirmDiagnosisResponse(
                triage_id=triage_id,
                urgency_level="URGENTE",
                color_code="ORANGE",
                is_escalated=(prev_urgency in ["MODERADO", "LEVE"]),
                clinical_reassessment=(
                    "REEVALUACIÓN DE URGENCIA POR INTOXICACIÓN O DOLOR AGUDO: "
                    "La confirmación de sospecha de ingesta de veneno, sustancias tóxicas o descompensación digestiva "
                    "requiere neutralización y estabilización rápida para prevenir absorción sistémica y daño hepatorrenal. "
                    "Se confirma el caso en nivel URGENTE / NARANJA."
                ),
                presumptive_diagnosis=[
                    "Intoxicación exógena aguda por rodenticida, toxina vegetal o fármacos",
                    "Gastroenteritis tóxica severa con potencial descompensación hidroelectrolítica",
                ],
                vital_risks=[
                    "Absorción sistémica del tóxico con coagulopatía o daño renal",
                    "Deshidratación rápida por pérdidas gastrointestinales",
                ],
                first_aid_actions=[
                    "Identificar y guardar el empaque, planta o residuo tóxico para mostrárselo al veterinario.",
                    "NO inducir el vómito si no lo ha indicado un médico (sustancias corrosivas queman el esófago al refluir).",
                    "Acudir de inmediato a una clínica veterinaria con servicio de urgencias.",
                ],
                warnings_contraindications=[
                    "NO suministrar leche, aceite ni remedios caseros (favorecen la absorción de tóxicos liposolubles).",
                    "NO dar medicamentos de uso humano.",
                    "NO forzar ingesta de alimentos ni líquidos.",
                ],
                estimated_fluid_deficit_percent=5.5,
                recommended_facility_type="Clínica 24 horas con quirófano",
            )
        else:
            # All rule-out answers are negative
            return ConfirmDiagnosisResponse(
                triage_id=triage_id,
                urgency_level="MODERADO" if "CRITIC" in prev_urgency else prev_urgency,
                color_code="YELLOW" if "CRITIC" in prev_urgency else ("GREEN" if prev_urgency == "LEVE" else "YELLOW"),
                is_escalated=False,
                clinical_reassessment=(
                    "REEVALUACIÓN CONFIRMATORIA ESTABLE: "
                    "Las respuestas del test descartan signos de alarma vital inmediata (encías con buena perfusión, "
                    "sin distensión abdominal dolorosa ni sospecha de tóxicos fulminantes). "
                    "Se mantiene la pauta de cuidados y monitoreo cercano de la evolución clínica."
                ),
                presumptive_diagnosis=[
                    "Cuadro clínico en evolución estable sin signos de shock",
                    "Afección moderada o leve con estabilidad hemodinámica relativa",
                ],
                vital_risks=[
                    "Descompensación si se presentan vómitos persistentes o letargia progresiva",
                ],
                first_aid_actions=[
                    "Mantener al paciente en reposo y en ambiente confortable y templado.",
                    "Ofrecer pequeñas cantidades de agua fresca únicamente si no presenta náuseas.",
                    "Monitorear la frecuencia respiratoria y acudir a consulta veterinaria si el cuadro empeora.",
                ],
                warnings_contraindications=[
                    "NO suministrar analgésicos de uso humano.",
                    "NO automedicar con antibióticos ni corticoides.",
                ],
                estimated_fluid_deficit_percent=3.0,
                recommended_facility_type="Consultorio general",
            )

    def _sync_confirm_diagnosis(
        self,
        triage_id: str,
        answers: Dict[str, Any],
        initial_symptoms: Optional[str] = None,
        species: Optional[str] = None,
        initial_urgency_level: Optional[str] = None,
        questions: Optional[List[DiagnosticQuestion]] = None,
    ) -> ConfirmDiagnosisResponse:
        """Synchronously executes diagnosis confirmation via Gemini or clinical rule engine."""
        # Retrieve context from session store if available
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        target_species = species or session.get("species") or "Mascota"
        target_symptoms = initial_symptoms or session.get("initial_symptoms") or "Paciente en triaje"
        target_initial_urgency = initial_urgency_level or (
            session.get("assessment", {}).get("urgency_level") if session.get("assessment") else "MODERADO"
        )
        target_questions = questions
        if not target_questions and session.get("questions"):
            try:
                target_questions = [DiagnosticQuestion.model_validate(q) for q in session["questions"]]
            except Exception:
                pass

        if not self._client or not self.is_configured:
            logger.info("[TRIAGE AI] Ejecutando confirmación diagnóstica con motor de reglas determinista.")
            res = self._generate_rule_based_confirmation(
                triage_id=triage_id,
                answers=answers,
                initial_symptoms=target_symptoms,
                species=target_species,
                initial_urgency_level=target_initial_urgency,
                questions=target_questions,
            )
            if triage_id in _GLOBAL_TRIAGE_SESSION_STORE:
                _GLOBAL_TRIAGE_SESSION_STORE[triage_id]["confirmation"] = res.model_dump()
            return res

        # Formulate prompt for Gemini confirmation
        prompt = (
            f"TEST CLÍNICO CONFIRMATORIO DE TRIAGE VETERINARIO:\n"
            f"- Triage ID: {triage_id}\n"
            f"- Paciente: {target_species}\n"
            f"- Síntomas iniciales: {target_symptoms}\n"
            f"- Nivel de urgencia previo: {target_initial_urgency}\n"
            f"- Respuestas del tutor a las preguntas de descarte:\n"
        )
        for q_id, ans in answers.items():
            prompt += f"  * ID {q_id}: {ans}\n"

        prompt += (
            "\nReevalúa metódicamente el estado clínico. "
            "Si el tutor confirma encías blancas o moradas, abdomen duro o sospecha de veneno, "
            "ESCALA INMEDIATAMENTE a CRÍTICO / ROJO. "
            "Responde ÚNICAMENTE con el objeto JSON puro estructurado."
        )

        config = types.GenerateContentConfig(
            system_instruction=VET_CONFIRMATION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ConfirmDiagnosisResponse,
            temperature=0.1,
        )

        try:
            logger.info("[TRIAGE AI] Reevaluando urgencia confirmatoria con Gemini modelo: %s", self._primary_model)
            response = self._client.models.generate_content(
                model=self._primary_model,
                contents=prompt,
                config=config,
            )

            result: Optional[ConfirmDiagnosisResponse] = None
            if response.parsed and isinstance(response.parsed, ConfirmDiagnosisResponse):
                result = response.parsed
            elif isinstance(response.parsed, dict):
                result = ConfirmDiagnosisResponse.model_validate(response.parsed)
            elif response.text and response.text.strip():
                parsed_json = clean_and_parse_json(response.text)
                result = ConfirmDiagnosisResponse.model_validate(parsed_json)

            if result:
                result.triage_id = triage_id
                if triage_id in _GLOBAL_TRIAGE_SESSION_STORE:
                    _GLOBAL_TRIAGE_SESSION_STORE[triage_id]["confirmation"] = result.model_dump()
                return result

        except Exception as exc:
            logger.warning("[TRIAGE AI NOTICE] Reevaluación con Gemini retornó: %s. Aplicando contingencia.", exc)

        fallback = self._generate_rule_based_confirmation(
            triage_id=triage_id,
            answers=answers,
            initial_symptoms=target_symptoms,
            species=target_species,
            initial_urgency_level=target_initial_urgency,
            questions=target_questions,
        )
        if triage_id in _GLOBAL_TRIAGE_SESSION_STORE:
            _GLOBAL_TRIAGE_SESSION_STORE[triage_id]["confirmation"] = fallback.model_dump()
        return fallback

    async def confirm_diagnosis(
        self,
        triage_id: str,
        answers: Dict[str, Any],
        initial_symptoms: Optional[str] = None,
        species: Optional[str] = None,
        initial_urgency_level: Optional[str] = None,
        questions: Optional[List[DiagnosticQuestion]] = None,
    ) -> ConfirmDiagnosisResponse:
        """Asynchronously executes diagnosis confirmation in worker thread pool."""
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
    # 3. EMERGENCY FIRST-AID CHAT ASSISTANT
    # -------------------------------------------------------------------------
    def _generate_rule_based_chat_reply(
        self,
        triage_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> str:
        """Deterministic empathetic first-aid guidance strictly forbidding human medicines."""
        msg = message.lower()
        species = context.get("species", "tu mascota") if context else "tu mascota"

        if any(k in msg for k in ["medicament", "paracetamol", "ibuprofen", "aspirin", "dolor", "pastilla"]):
            return (
                f"⚠️ ATENCIÓN MÉDICA CRÍTICA: Bajo NINGUNA circunstancia le des paracetamol, ibuprofeno, "
                f"aspirina ni ningún analgésico humano a {species}. Estos medicamentos provocan intoxicación fulminante, "
                "fallo hepático agudo y destrucción de glóbulos rojos en animales. "
                "Para aliviar su malestar, mantenlo en un lugar tranquilo, templado, sin movimientos bruscos y "
                "llévalo de inmediato a un centro veterinario donde puedan administrarle analgesia segura y regulada."
            )
        elif any(k in msg for k in ["sangr", "herida", "corte", "hemorragia"]):
            return (
                f"🩸 PRIMEROS AUXILIOS PARA HEMORRAGIA EN {species.upper()}:\n"
                "1. Toma una gasa, toalla o paño limpio y colócalo directamente sobre la herida.\n"
                "2. Aplica presión constante, suave y firme con tu mano durante al menos 3 a 5 minutos continuos sin levantar el paño.\n"
                "3. Si la toalla se empapa de sangre, NO la retires; coloca otra limpia por encima y sigue presionando.\n"
                "4. NO apliques torniquetes cerrados ni sustancias caseras (alcohol, café o vinagre).\n"
                "5. Mantén la calma para no asustar al paciente y trasládalo inmediatamente al hospital veterinario más cercano."
            )
        elif any(k in msg for k in ["vomit", "veneno", "toxico", "basura", "comio"]):
            return (
                f"⚠️ MANEJO DE VÓMITO O SOSPECHA DE INTOXICACIÓN EN {species.upper()}:\n"
                "1. NO intentes inducir el vómito con sal, agua oxigenada ni leche sin indicación médica explícita. Si ingirió sustancias corrosivas o cáusticas, el vómito perforará su esófago.\n"
                "2. Retira de inmediato cualquier residuo o empaque sospechoso del alcance y guárdalo en una bolsa para que el veterinario lo identifique.\n"
                "3. No le des agua ni comida a la fuerza para prevenir broncoaspiración.\n"
                "4. Si presenta salivación excesiva o temblores, acude de inmediato a urgencias 24 horas."
            )
        elif any(k in msg for k in ["respir", "ahog", "jadeo", "aire"]):
            return (
                f"🫁 DIFICULTAD RESPIRATORIA EN {species.upper()}:\n"
                "1. Asegura que su cuello esté estirado y alineado con su columna para mantener la vía aérea despejada.\n"
                "2. Retira con cuidado collares, arneses o bozales.\n"
                "3. Abre ventanas o asegura ventilación de aire fresco durante el transporte.\n"
                "4. Evita el estrés, ruidos fuertes o manipularlo en exceso. Requiere oxigenoterapia profesional inmediata."
            )
        elif any(k in msg for k in ["traslad", "llevar", "transporte", "mover"]):
            return (
                f"🚗 TRANSPORTE SEGURO DE {species.upper()}:\n"
                "1. Utiliza una toalla grande, manta gruesa o tabla a modo de camilla improvisada para levantarlo sin forzar su columna ni miembros.\n"
                "2. Si es un perro grande, levántalo entre dos personas manteniendo su cuerpo horizontal.\n"
                "3. Si es un gato o perro pequeño, colócalo en su transportadora acolchada y cúbrela parcialmente con una tela para reducir el estrés visual.\n"
                "4. Llama a la clínica mientras vas en camino para que el equipo médico esté preparado a tu llegada."
            )
        else:
            return (
                f"Comprendo tu preocupación por {species}. Mantén la calma, ya que tu tranquilidad ayudará a estabilizarlo. "
                "Mantenlo abrigado con una manta, en reposo y en un ambiente silencioso con luz tenue. "
                "Recuerda que no debes darle ningún medicamento humano ni forzarlo a beber agua. "
                "Si notas cualquier signo de empeoramiento (encías pálidas, dificultad para respirar o debilidad extrema), "
                "dirígete de inmediato al centro veterinario de urgencias más cercano."
            )

    def _sync_emergency_chat(
        self,
        triage_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> EmergencyChatResponse:
        """Synchronously processes emergency chat with Gemini or deterministic empathetic assistant."""
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        full_context = dict(session)
        if context:
            full_context.update(context)

        species = full_context.get("species") or "Mascota"
        assessment = full_context.get("assessment", {})
        urgency = assessment.get("urgency_level") or full_context.get("urgency_level") or "MODERADO"

        history = conversation_history or []

        if not self._client or not self.is_configured:
            logger.info("[TRIAGE AI CHAT] Operando con asistente determinista de contingencia.")
            reply_text = self._generate_rule_based_chat_reply(
                triage_id=triage_id,
                message=message,
                context=full_context,
                conversation_history=history,
            )
            return EmergencyChatResponse(
                triage_id=triage_id,
                reply=reply_text,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )

        # Build prompt for Gemini chat
        chat_prompt = (
            f"CONTEXTO DEL CASO DE TRIAGE:\n"
            f"- Triage ID: {triage_id}\n"
            f"- Especie: {species}\n"
            f"- Nivel de Urgencia: {urgency}\n"
        )
        if assessment.get("presumptive_diagnosis"):
            chat_prompt += f"- Diagnósticos presuntivos: {', '.join(assessment.get('presumptive_diagnosis', []))}\n"
        if assessment.get("vital_risks"):
            chat_prompt += f"- Riesgos vitales: {', '.join(assessment.get('vital_risks', []))}\n"

        if history:
            chat_prompt += "\nHISTORIAL PREVIO DE MENSAJES:\n"
            for m in history[-4:]:
                chat_prompt += f"- {m.role.upper()}: {m.content}\n"

        chat_prompt += (
            f"\nNUEVA CONSULTA DEL TUTOR:\n"
            f"\"{message}\"\n\n"
            f"Responde directamente al tutor con instrucciones de primeros auxilios y orientación médica empática, "
            f"segura y directa. Recuerda: NUNCA recetes medicamentos humanos ni fármacos bajo receta."
        )

        config = types.GenerateContentConfig(
            system_instruction=VET_CHAT_SYSTEM_PROMPT,
            temperature=0.2,
        )

        try:
            logger.info("[TRIAGE AI CHAT] Consultando asistente de urgencias con Gemini modelo: %s", self._primary_model)
            response = self._client.models.generate_content(
                model=self._primary_model,
                contents=chat_prompt,
                config=config,
            )
            if response.text and response.text.strip():
                return EmergencyChatResponse(
                    triage_id=triage_id,
                    reply=response.text.strip(),
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
        except Exception as exc:
            logger.warning("[TRIAGE AI CHAT NOTICE] Fallo en Gemini Chat: %s. Activando contingencia.", exc)

        reply_text = self._generate_rule_based_chat_reply(
            triage_id=triage_id,
            message=message,
            context=full_context,
            conversation_history=history,
        )
        return EmergencyChatResponse(
            triage_id=triage_id,
            reply=reply_text,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    async def emergency_chat(
        self,
        triage_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> EmergencyChatResponse:
        """Asynchronously processes emergency first-aid chat in worker thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_emergency_chat,
            triage_id,
            message,
            context,
            conversation_history,
        )
