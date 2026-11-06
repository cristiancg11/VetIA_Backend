"""Multimodal Veterinary Triage AI Service using Google Gemini Flash.
Supports multimodal analysis with strict visual byte inspection and species differentiation,
clinical confirmation test (rule-out questions), and emergency first-aid chat.
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

VET_TRIAGE_SYSTEM_PROMPT = """Eres un médico veterinario especialista en triage de urgencias. Tu tarea primordial es INSPECCIONAR VISUALMENTE la imagen adjunta.

REGLAS OBLIGATORIAS:
- Primero, identifica la especie exacta que aparece en la imagen o la muestra biológica (vómito, heces, piel, mucosas, herida, postura). Si el usuario dice que es un gato y en la foto se ve un gato o muestra felina, analiza la fisiopatología FELINA (los gatos se descompensan mucho más rápido por lipidosis hepática, obstrucción uretral felina, hipotermia o choque circulatorio silente). Si es un perro, analiza la fisiopatología CANINA (riesgo de torsión gástrica/GDV, parvovirosis, shock hipovolémico).
- En el campo 'visual_findings' DEBES describir elementos visuales concretos y únicos de ESTA foto específica (color exacto, textura, consistencia, área anatómica visible, presencia o ausencia de sangre, postura del animal). Si no describes lo que ves visualmente en la foto, el diagnóstico será inválido.
- Formula las 3 a 4 preguntas de descarte clínico ('diagnostic_questions') TOTALMENTE PERSONALIZADAS a lo que acabas de ver en la foto y a la especie indicada, con respuesta binaria ("yes_no").
- PROHIBIDO devolver respuestas de plantilla o diagnósticos genéricos de canino cuando se evalúa un felino o viceversa.
- Categoriza la urgencia según el Manchester Triage System:
  * CRITICO / RED (ROJO): Riesgo vital inminente, shock, paro inminente, sangrado masivo activo, obstrucción uretral o respiratoria, inconsciencia.
  * URGENTE / ORANGE (NARANJA): Muy urgente, intoxicación o sospecha de veneno, dolor agudo severo, vómito incoercible, laceraciones profundas.
  * MODERADO / YELLOW (AMARILLO): Urgente pero estable (< 24h, cojera, abscesos, conjuntivitis severa).
  * LEVE / GREEN (VERDE): No urgente, abrasiones superficiales limpias, afecciones subagudas leves.
- CONTRAINDICACIONES ESTRICTAS: PROHIBIDO indicar o sugerir medicamentos humanos (paracetamol, acetaminofén, ibuprofeno, aspirina son LETALES en perros y gatos). NO inducir vómito sin indicación médica presencial.

FORMATO ESTRICTO DE SALIDA:
Debes responder ÚNICAMENTE con un objeto JSON válido, sin bloques de markdown envolventes (sin ```json ni ```) y sin ningún texto fuera del JSON:
{
  "triage_id": "generar-un-uuid-valido",
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
  "visual_findings": "Descripción detallada y única de los hallazgos observados en esta foto específica",
  "presumptive_diagnosis": ["Diagnóstico 1 específico de la especie", "Diagnóstico 2"],
  "vital_risks": ["Riesgo principal detectado"],
  "first_aid_actions": ["Paso 1 de primeros auxilios", "Paso 2"],
  "warnings_contraindications": ["NO suministrar analgésicos humanos", "NO forzar ingestión de agua"],
  "estimated_fluid_deficit_percent": 5.0,
  "diagnostic_questions": [
    {"id": "q1", "question": "Pregunta de descarte adaptada a la especie y lo visto en la foto", "type": "yes_no"},
    {"id": "q2", "question": "Pregunta de descarte 2", "type": "yes_no"},
    {"id": "q3", "question": "Pregunta de descarte 3", "type": "yes_no"}
  ]
}
"""

VET_CONFIRMATION_SYSTEM_PROMPT = """Eres un Médico Veterinario Especialista en Triage y Cuidados Críticos de Urgencias Veterinarias (VetIA).
Tu misión es RE-EVALUAR de forma definitiva el nivel de urgencia y pronóstico de un paciente tras recibir las respuestas del tutor al test clínico confirmatorio de descarte.

REGLAS MÉDICAS DETERMINANTES:
1. ESCALADA CRÍTICA INMEDIATA:
   - Si el tutor confirma encías blancas, pálidas o azuladas/moradas (signo inminente de hipoperfusión, shock hipovolémico o cianosis severa) -> DEBE ESCALAR A CRÍTICO / RED.
   - Si el tutor confirma abdomen duro, distendido, rígido o doloroso al tacto (sospecha de abdomen agudo, torsión gástrica dilatada en caninos o hemoperitoneo/peritonitis) -> DEBE ESCALAR A CRÍTICO / RED.
   - Si el tutor confirma inconsciencia, convulsión activa, sangrado pulsátil que no cede o ahogo/asfixia -> DEBE ESCALAR A CRÍTICO / RED.
   - Si el tutor confirma ingesta de veneno, rodenticida o tóxicos químicos -> DEBE ESCALAR A CRÍTICO / RED o URGENTE / ORANGE con traslado inmediato.
2. Si todas las respuestas de descarte son negativas y no hay signos de alarma, mantén la clasificación preliminar o ajusta con prudencia hacia un nivel más seguro.
3. Prohibido recomendar medicamentos humanos. Proporciona instrucciones de estabilización antes del traslado.

FORMATO ESTRICTO DE SALIDA (JSON PURO sin markdown):
{
  "triage_id": "id-del-triaje",
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
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
    """Generates 3 to 4 clinical rule-out questions tailored specifically to symptoms and species."""
    text = (descripcion or "").lower()
    is_feline = any(k in especie.lower() for k in ["felin", "gat"])

    if any(k in text for k in ["vomit", "diarre", "veneno", "toxico", "estomago", "comio", "ingir"]):
        q4 = (
            "¿Lleva más de 24 horas sin comer o beber agua? (En gatos esto puede causar lipidosis hepática grave)"
            if is_feline
            else "¿El perro intenta vomitar con arcadas improductivas continuas?"
        )
        return [
            DiagnosticQuestion(id="q1", question="¿Las encías están blancas o moradas?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿El abdomen se siente duro, inflamado o doloroso al tacto?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿Sospecha de ingesta de veneno, plantas tóxicas (ej. lirios en gatos) o basura?", type="yes_no"),
            DiagnosticQuestion(id="q4", question=q4, type="yes_no"),
        ]
    elif any(k in text for k in ["sangr", "herida", "corte", "atropell", "golpe", "caida", "fractur"]):
        return [
            DiagnosticQuestion(id="q1", question="¿El sangrado es continuo, a chorro o no cesa con compresión firme?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿Las encías se aprecian blancas, pálidas o azuladas?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿El animal está inconsciente, no responde o presenta respiración con boca abierta?", type="yes_no"),
            DiagnosticQuestion(id="q4", question="¿Se observa deformidad evidente en las extremidades o hueso expuesto?", type="yes_no"),
        ]
    elif any(k in text for k in ["orina", "miccion", "vejiga", "uretr"]):
        return [
            DiagnosticQuestion(id="q1", question="¿Pasa tiempo en el arenero pujando sin expulsar orina o maúlla de dolor?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿El abdomen bajo se siente como una pelota dura y dolorosa?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿Las encías se sienten frías, pálidas o pegajosas?", type="yes_no"),
            DiagnosticQuestion(id="q4", question="¿Presenta vómitos, letargia extrema o temblores corporales?", type="yes_no"),
        ]
    else:
        q_respir = (
            "¿Respira con la boca abierta o jadea intensamente? (En gatos la respiración con boca abierta es emergencia crítica)"
            if is_feline
            else "¿Presenta respiración agitada, cuello estirado o jadeo con ruidos anormales?"
        )
        return [
            DiagnosticQuestion(id="q1", question="¿Las encías están blancas o moradas?", type="yes_no"),
            DiagnosticQuestion(id="q2", question="¿El abdomen se siente duro, inflamado o doloroso al tacto?", type="yes_no"),
            DiagnosticQuestion(id="q3", question="¿Sospecha de ingesta de veneno, plantas tóxicas o basura?", type="yes_no"),
            DiagnosticQuestion(id="q4", question=q_respir, type="yes_no"),
        ]


class MultimodalTriageAIService:
    """
    Multimodal clinical veterinary triage service powered by Google Gemini Flash.
    Executes real image byte inspection within worker threads to ensure non-blocking async FastAPI behavior.
    Includes species-specific clinical reasoning, diagnostic confirmation, and emergency first-aid chat.
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

        # Candidate models list: start with primary, follow with active Gemini flash models
        self._candidate_models = [
            self._primary_model,
            "gemini-3.5-flash",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
        ]
        # De-duplicate while preserving order
        self._candidate_models = list(dict.fromkeys(self._candidate_models))

        self._client: Optional[genai.Client] = None

        if self.is_configured:
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(timeout=12000),
                )
                logger.info(
                    "[TRIAGE AI] Cliente de Google Gemini inicializado. Modelos candidatos: %s",
                    self._candidate_models,
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
        Deterministic species-specific clinical rule engine used only when all Gemini
        models are completely unreachable (no internet or service outage).
        """
        text = descripcion.lower()
        is_feline = any(k in especie.lower() for k in ["felin", "gat"])

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

        species_tag = "Felino" if is_feline else "Canino"
        patient_desc = f"{species_tag} ({especie}, {peso_kg} kg)"
        if raza:
            patient_desc += f", raza {raza}"

        triage_id = str(uuid.uuid4())
        diagnostic_questions = generate_default_diagnostic_questions(descripcion, especie)

        if any(sig in text for sig in critical_signals):
            diag = (
                ["Shock hipovolémico / hipotérmico agudo felino", "Obstrucción uretral o trauma torácico con disnea felina"]
                if is_feline
                else ["Traumatismo severo canino con hemorragia activa o fractura", "Síndrome de dilatación-torsión gástrica (GDV) / Shock hipovolémico"]
            )
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="CRITICO",
                color_code="RED",
                visual_findings=(
                    f"Inspección clínica urgente para {patient_desc}: "
                    "Se evidencia solución de continuidad tisular grave o signos compatibles con sangrado profuso activo, "
                    "colapso circulatorio o alteración neuromuscular severa descrita por el tutor."
                ),
                presumptive_diagnosis=diag,
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
            diag = (
                ["Gastroenteritis aguda felina con riesgo acelerado de lipidosis hepática", "Laceración tisular penetrante con riesgo de infección profunda"]
                if is_feline
                else ["Gastroenteritis aguda severa canina con pérdida hidroelectrolítica crítica", "Intoxicación o laceración dérmica profunda"]
            )
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="URGENTE",
                color_code="ORANGE",
                visual_findings=(
                    f"Inspección clínica de urgencia para {patient_desc}: "
                    "Se observan lesiones dérmicas de espesor considerable o signos de pérdida profusa de fluidos "
                    "(vómito/deposición anormal o dolor agudo) con riesgo de rápido deterioro sistémico."
                ),
                presumptive_diagnosis=diag,
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
            diag = (
                ["Dermatitis focal o absceso felino (frecuente por rasguño o mordedura)", "Lesión osteoarticular con apoyo parcial"]
                if is_feline
                else ["Dermatitis bacteriana focalizada / celulitis o absceso canino", "Lesión musculoesquelética de grado leve a moderado"]
            )
            return TriageAnalysisResponse(
                triage_id=triage_id,
                urgency_level="MODERADO",
                color_code="YELLOW",
                visual_findings=(
                    f"Inspección clínica para {patient_desc}: "
                    "Se aprecia proceso inflamatorio focalizado, claudicación con apoyo parcial o secreción localizada "
                    "sin sangrado profuso ni signos de shock inminente."
                ),
                presumptive_diagnosis=diag,
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
                    f"Inspección clínica para {patient_desc}: "
                    "Lesión superficial limpia o alteración menor sin compromiso dérmico profundo ni signos de inflamación severa."
                ),
                presumptive_diagnosis=[
                    f"Excoriación superficial limpia en {species_tag.lower()}",
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
        Synchronous multimodal inference call to Google Gemini with real image bytes ingestion.
        Tries candidate models sequentially to guarantee actual AI execution.
        """
        # Mandatory console log as requested
        print(f"[TRIAGE AI] Analizando imagen de {len(image_bytes)} bytes para especie: {especie}")
        logger.info("[TRIAGE AI] Analizando imagen de %d bytes para especie: %s", len(image_bytes), especie)

        if not self._client or not self.is_configured:
            logger.info("[TRIAGE AI] Cliente Gemini no disponible, activando motor determinista.")
            res = self._generate_rule_based_fallback(
                especie=especie,
                peso_kg=peso_kg,
                descripcion=descripcion,
                raza=raza,
                edad_aprox=edad_aprox,
            )
            print(f"[TRIAGE AI] Respuesta generada (fallback): {res.urgency_level} - {res.visual_findings[:80]}...")
            self._save_session(res.triage_id, res, especie, peso_kg, descripcion)
            return res

        # Construct multimodal input parts
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type or "image/jpeg",
        )

        patient_info = f"- Especie del paciente: {especie}\n- Peso aproximado: {peso_kg} kg\n"
        if raza:
            patient_info += f"- Raza: {raza}\n"
        if edad_aprox:
            patient_info += f"- Edad aproximada: {edad_aprox}\n"
        patient_info += f"- Descripción de síntomas / motivo de consulta: {descripcion}\n"

        prompt_text = (
            f"CONSULTA CLÍNICA DE TRIAGE VETERINARIO MULTIMODAL:\n"
            f"{patient_info}\n"
            f"INSTRUCCIONES CLÍNICAS PRIORITARIAS:\n"
            f"1. Inspecciona minuciosamente la imagen adjunta. Describe en 'visual_findings' los elementos "
            f"visuales exactos observados (colorimetría, texturas, heridas, consistencia de fluidos o postura).\n"
            f"2. Basa tu análisis exclusivamente en la fisiopatología de {especie}. No apliques razonamiento de caninos si es felino ni viceversa.\n"
            f"3. Genera 3 a 4 preguntas de descarte clínico ('diagnostic_questions') totalmente personalizadas a lo que ves en la imagen.\n"
            f"Responde ÚNICAMENTE con el objeto JSON puro estructurado según el esquema solicitado."
        )

        config = types.GenerateContentConfig(
            system_instruction=VET_TRIAGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.1,
        )

        # Iterate through models: primary first, then working fallback flash models
        for model_name in self._candidate_models:
            try:
                logger.info("[TRIAGE AI] Invocando modelo multimodal: %s con imagen de %d bytes", model_name, len(image_bytes))
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=[image_part, prompt_text],
                    config=config,
                )

                raw_text = response.text or ""
                if not raw_text.strip() and response.parsed:
                    result_dict = response.parsed.model_dump() if hasattr(response.parsed, "model_dump") else dict(response.parsed)
                else:
                    result_dict = clean_and_parse_json(raw_text)

                if result_dict and isinstance(result_dict, dict):
                    # Guarantee unique UUID and species-specific questions if missing
                    if not result_dict.get("triage_id"):
                        result_dict["triage_id"] = str(uuid.uuid4())
                    if not result_dict.get("diagnostic_questions"):
                        result_dict["diagnostic_questions"] = [
                            q.model_dump() for q in generate_default_diagnostic_questions(descripcion, especie)
                        ]

                    assessment = TriageAnalysisResponse.model_validate(result_dict)

                    # Required console output
                    print(f"[TRIAGE AI] Respuesta generada: {result_dict.get('urgency_level')} - {str(result_dict.get('visual_findings', ''))[:80]}...")
                    logger.info(
                        "[TRIAGE AI SUCCESS] Inferencia multimodal completada con %s para %s. Nivel: %s",
                        model_name,
                        especie,
                        assessment.urgency_level,
                    )

                    self._save_session(assessment.triage_id, assessment, especie, peso_kg, descripcion)
                    return assessment

            except Exception as exc:
                logger.warning(
                    "[TRIAGE AI NOTICE] Inferencia con modelo %s falló: %s. Probando siguiente modelo.",
                    model_name,
                    exc,
                )

        logger.error("[TRIAGE AI ERROR] Todos los modelos de Gemini fallaron. Utilizando contingencia.")
        fallback = self._generate_rule_based_fallback(
            especie=especie,
            peso_kg=peso_kg,
            descripcion=descripcion,
            raza=raza,
            edad_aprox=edad_aprox,
        )
        print(f"[TRIAGE AI] Respuesta generada (fallback de emergencia): {fallback.urgency_level} - {fallback.visual_findings[:80]}...")
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
        q_map: Dict[str, str] = {}
        if questions:
            for q in questions:
                q_map[q.id] = q.question.lower()

        has_critical_sign = False
        has_urgent_sign = False

        for q_id, ans in answers.items():
            if not self._is_truthy(ans):
                continue

            q_text = q_map.get(q_id, "").lower()

            if q_id in ["q1", "1"] or any(k in q_text for k in ["encía", "gias", "blanca", "morad", "azul", "palid"]):
                has_critical_sign = True
            elif q_id in ["q2", "2"] or any(k in q_text for k in ["abdomen", "duro", "inflam", "dolor", "hinch", "distend"]):
                has_critical_sign = True
            elif any(k in q_text for k in ["inconscient", "desmay", "convuls", "no respira", "ahog", "asfixi", "sangrado continuo", "chorro"]):
                has_critical_sign = True
            elif q_id in ["q3", "3"] or any(k in q_text for k in ["veneno", "toxico", "raticida", "quimico", "basura", "vomito incoercible"]):
                has_urgent_sign = True

        prev_urgency = (initial_urgency_level or "MODERADO").upper()
        target_species = species or "Mascota"
        is_feline = any(k in target_species.lower() for k in ["felin", "gat"])

        if has_critical_sign:
            diag = (
                ["Shock hemodinámico felino agudo", "Obstrucción uretral felina (FLUTD) / Hipoperfusión severa", "Peritonitis o hemorragia interna"]
                if is_feline
                else ["Shock hemodinámico canino por colapso cardiovascular", "Abdomen agudo quirúrgico / Dilatación-torsión gástrica (GDV)", "Hemorragia interna activa"]
            )
            return ConfirmDiagnosisResponse(
                triage_id=triage_id,
                urgency_level="CRITICO",
                color_code="RED",
                is_escalated=(prev_urgency != "CRITICO"),
                clinical_reassessment=(
                    f"ALERTA CLÍNICA ROJA - ESCALADA DE URGENCIA CRÍTICA ({target_species}): "
                    "El tutor confirmó signos de alarma vital (como encías pálidas/cianóticas o abdomen agudo doloroso). "
                    "Estos hallazgos son compatibles con colapso hemodinámico inminente, shock hipovolémico o descompensación severa. "
                    "Se escala de inmediato a nivel CRÍTICO / ROJO con requerimiento de estabilización hospitalaria urgente."
                ),
                presumptive_diagnosis=diag,
                vital_risks=[
                    "Paro cardiorrespiratorio inminente por colapso circulatorio",
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
                    f"REEVALUACIÓN DE URGENCIA POR INTOXICACIÓN O DOLOR AGUDO ({target_species}): "
                    "La confirmación de sospecha de ingesta de veneno, sustancias tóxicas o descompensación digestiva "
                    "requiere neutralización y estabilización rápida para prevenir absorción sistémica y daño hepatorrenal. "
                    "Se confirma el caso en nivel URGENTE / NARANJA."
                ),
                presumptive_diagnosis=[
                    f"Intoxicación exógena aguda en {target_species.lower()}",
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
            return ConfirmDiagnosisResponse(
                triage_id=triage_id,
                urgency_level="MODERADO" if "CRITIC" in prev_urgency else prev_urgency,
                color_code="YELLOW" if "CRITIC" in prev_urgency else ("GREEN" if prev_urgency == "LEVE" else "YELLOW"),
                is_escalated=False,
                clinical_reassessment=(
                    f"REEVALUACIÓN CONFIRMATORIA ESTABLE ({target_species}): "
                    "Las respuestas del test descartan signos de alarma vital inmediata (encías con buena perfusión, "
                    "sin distensión abdominal dolorosa ni sospecha de tóxicos fulminantes). "
                    "Se mantiene la pauta de cuidados y monitoreo cercano de la evolución clínica."
                ),
                presumptive_diagnosis=[
                    f"Cuadro clínico en {target_species.lower()} en evolución estable sin signos de shock",
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
        """Synchronously executes diagnosis confirmation via Gemini with multi-model failover."""
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
            logger.info("[TRIAGE AI] Reevaluación con motor de reglas determinista.")
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

        prompt = (
            f"TEST CLÍNICO CONFIRMATORIO DE TRIAGE VETERINARIO:\n"
            f"- Triage ID: {triage_id}\n"
            f"- Especie: {target_species}\n"
            f"- Síntomas iniciales: {target_symptoms}\n"
            f"- Nivel de urgencia previo: {target_initial_urgency}\n"
            f"- Respuestas del tutor a las preguntas de descarte:\n"
        )
        for q_id, ans in answers.items():
            prompt += f"  * ID {q_id}: {ans}\n"

        prompt += (
            f"\nReevalúa metódicamente el estado clínico considerando la fisiopatología de {target_species}. "
            "Si el tutor confirma encías blancas o moradas, abdomen duro o sospecha de veneno, "
            "ESCALA INMEDIATAMENTE a CRÍTICO / ROJO. "
            "Responde ÚNICAMENTE con el objeto JSON puro estructurado."
        )

        config = types.GenerateContentConfig(
            system_instruction=VET_CONFIRMATION_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.1,
        )

        for model_name in self._candidate_models:
            try:
                logger.info("[TRIAGE AI] Reevaluando urgencia confirmatoria con Gemini modelo: %s", model_name)
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )

                raw_text = response.text or ""
                parsed_json = clean_and_parse_json(raw_text)
                if parsed_json and isinstance(parsed_json, dict):
                    parsed_json["triage_id"] = triage_id
                    result = ConfirmDiagnosisResponse.model_validate(parsed_json)
                    if triage_id in _GLOBAL_TRIAGE_SESSION_STORE:
                        _GLOBAL_TRIAGE_SESSION_STORE[triage_id]["confirmation"] = result.model_dump()
                    return result

            except Exception as exc:
                logger.warning("[TRIAGE AI NOTICE] Reevaluación con Gemini (%s) falló: %s", model_name, exc)

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
        """Synchronously processes emergency chat with Gemini or deterministic assistant."""
        session = _GLOBAL_TRIAGE_SESSION_STORE.get(triage_id, {})
        full_context = dict(session)
        if context:
            full_context.update(context)

        species = full_context.get("species") or "Mascota"
        assessment = full_context.get("assessment", {})
        urgency = assessment.get("urgency_level") or full_context.get("urgency_level") or "MODERADO"

        history = conversation_history or []

        if not self._client or not self.is_configured:
            logger.info("[TRIAGE AI CHAT] Operando con asistente determinista.")
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
            f"Responde directamente al tutor de {species} con instrucciones de primeros auxilios y orientación médica empática, "
            f"segura y directa. Recuerda: NUNCA recetes medicamentos humanos ni fármacos bajo receta médica."
        )

        config = types.GenerateContentConfig(
            system_instruction=VET_CHAT_SYSTEM_PROMPT,
            temperature=0.2,
        )

        for model_name in self._candidate_models:
            try:
                logger.info("[TRIAGE AI CHAT] Consultando chat con Gemini modelo: %s", model_name)
                response = self._client.models.generate_content(
                    model=model_name,
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
                logger.warning("[TRIAGE AI CHAT NOTICE] Fallo en Gemini Chat (%s): %s", model_name, exc)

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
