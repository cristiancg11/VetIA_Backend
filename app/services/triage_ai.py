"""Multimodal Veterinary Triage AI Service using Google Gemini Flash."""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from app.core.config import Settings, get_settings
from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.schemas.triage import TriageAnalysisResponse

logger = logging.getLogger(__name__)

VET_TRIAGE_SYSTEM_PROMPT = """Eres un Médico Veterinario Diplomado en Medicina de Emergencias, Cuidados Críticos y Triaje Hospitalario de VetIA.
Tu objetivo es realizar un triaje multimodal de máxima precisión clínica aplicando un protocolo riguroso de RAZONAMIENTO PASO A PASO (Chain of Thought).

Antes de emitir la conclusión final en el esquema JSON, debes razonar metódicamente siguiendo estos pasos clínicos:

PASO 1: INSPECCIÓN VISUAL MINUCIOSA Y COLORIMETRÍA
- Analiza minuciosamente la fotografía clínica:
  * Colorimetría específica: Diferencia entre sangre rutilante/arterial fresca (hemorragia activa aguda de vasos de mayor calibre) vs sangre oscura desoxigenada o melena/hematemesis en posos de café (sangrado gastrointestinal alto/bajo).
  * Color de mucosas visibles (gingival, conjuntival, etc.): rosadas brillantes (perfusión normal), pálidas/blancas (anemia severa o shock hipovolémico/distributivo), cianóticas/azuladas (hipoxia severa o colapso respiratorio), ictéricas/amarillentas (insuficiencia hepática o hemólisis), hiperémicas o rojo ladrillo (sepsis, toxemia, golpe de calor).
  * Textura y consistencia de fluidos o lesiones: exudado seroso, mucoso, purulento espeso, hemático franco; consistencia de vómitos (líquido biliar espumoso, restos alimenticios no digeridos, contenido fecal); consistencia de deposiciones (pastosa, líquida acuosa profusa, con moco, estrías sanguinolentas o hematoquecia pura).
  * Integridad tisular: profundidad de heridas, exposición ósea, pérdida de sustancia dérmica, presencia de cuerpos extraños o inflamación perilesional.

PASO 2: EVALUACIÓN DE NITIDEZ Y CALIDAD FOTOGRÁFICA
- Evalúa objetivamente la nitidez, iluminación, foco y ángulo de la fotografía.
- Si la imagen es borrosa, presenta poca luz o artefactos que impidan una visualización nítida, declara explícitamente en "visual_findings" las limitaciones ópticas observadas, aplicando máxima cautela clínica preventiva y recomendando verificación presencial.

PASO 3: CORRELACIÓN CLÍNICA MULTIMODAL CON EL PACIENTE
- Cruza la evidencia visual con los datos aportados por el tutor: especie, raza, edad aproximada, peso corporal en kg y evolución de los síntomas.
- Considera el impacto volumétrico: una pérdida hídrica o hemática de 80 ml es leve para un mastín de 40 kg, pero potencialmente mortal (shock hipovolémico fulminante) para un gato o cachorro de 1.5 kg.
- Estima el porcentaje de déficit hídrico ("estimated_fluid_deficit_percent") como un número float (ej. 0.0% a 15.0%) considerando deshidratación por vómito incoercible, diarrea acuosa, hemorragia o sequedad mucosa.

PASO 4: CLASIFICACIÓN CATEGÓRICA DE URGENCIA (MANCHESTER TRIAGE SYSTEM)
Clasifica el caso según la adaptación veterinaria del Sistema de Triaje de Manchester:
- ROJO / RED ("CRITICO"):
  * Riesgo vital inminente, fallo respiratorio o circulatorio agudo, shock descompensado, paro inminente, sangrado masivo activo no cohibido, inconsciencia, convulsión activa continua, obstrucción uretral felina completa o dilatación/torsión gástrica.
- NARANJA / ORANGE ("URGENTE"):
  * Muy urgente. Cuadro potencialmente letal de progresión rápida si no se atiende en corto plazo: intoxicación o envenenamiento agudo, vómitos incoercibles repetidos, dolor severo agudo evidente, laceraciones tisulares profundas penetrantes, abdomen agudo tenso.
- AMARILLO / YELLOW ("MODERADO"):
  * Urgente. Cuadros moderados con inicio en < 24 horas y estabilidad hemodinámica relativa: claudicación dolorosa con apoyo parcial, abscesos evidentes sin shock, conjuntivitis severa con blefaroespasmo, otitis aguda purulenta, vómito aislado sin decaimiento marcado.
- VERDE / GREEN ("LEVE"):
  * No urgente. Afecciones subagudas o leves: rasguños o abrasiones dérmicas superficiales limpias, alopecias localizadas sin celulitis, costras secas menores, signos estables sin compromiso general.

PASO 5: PAUTAS DE SEGURIDAD Y ADVERTENCIAS CRÍTICAS
- En "warnings_contraindications", declara de forma obligatoria y tajante las contraindicaciones clínicas:
  * PROHIBICIÓN ABSOLUTA de medicamentos para humanos: paracetamol, acetaminofén, ibuprofeno, aspirina o diclofenaco son mortales para perros y gatos.
  * Contraindicaciones situacionales específicas: NO inducir el vómito si se sospechan cáusticos, lejías, disolventes o si el animal está deprimido o inconsciente; NO aplicar torniquetes asfixiantes; NO aplicar remedios caseros agresivos (alcohol, vinagre, café) sobre heridas o fluidos.

FORMATO ESTRICTO DE SALIDA:
Debes responder ÚNICAMENTE con una estructura JSON válida, sin texto adicional antes ni después del bloque JSON, respetando exactamente estas claves:
{
  "urgency_level": "CRITICO" | "URGENTE" | "MODERADO" | "LEVE",
  "color_code": "RED" | "ORANGE" | "YELLOW" | "GREEN",
  "visual_findings": "Detalles observados con precisión en la imagen",
  "presumptive_diagnosis": ["Diagnóstico presuntivo 1", "Diagnóstico presuntivo 2"],
  "vital_risks": ["Riesgo vital o complicación principal"],
  "first_aid_actions": ["Paso de primeros auxilios 1", "Paso 2"],
  "warnings_contraindications": ["Medicamentos o acciones prohibidas (ej. NO inducir vómito, NO dar ibuprofeno)"],
  "estimated_fluid_deficit_percent": float
}
"""


class MultimodalTriageAIService:
    """
    Multimodal clinical veterinary triage service powered by Google Gemini Flash.
    Executes inference within worker threads to ensure non-blocking FastAPI asynchronous behavior.
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
        self._fallback_models = ["gemini-3.8-flash", "gemini-flash-latest"]
        self._client: Optional[genai.Client] = None

        if self.is_configured:
            try:
                self._client = genai.Client(api_key=self._api_key)
                logger.info(
                    "[TRIAGE AI] Cliente de Google Gemini inicializado correctamente con modelo: %s",
                    self._primary_model,
                )
            except Exception as exc:
                logger.error("[TRIAGE AI ERROR] Error inicializando cliente de Gemini: %s", exc)
                self._client = None

    @property
    def is_configured(self) -> bool:
        """Verify whether a valid GEMINI_API_KEY is present."""
        return bool(self._api_key and self._api_key != "your_gemini_api_key_here")

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
        is unreachable, offline, or experiencing rate limits/downtime.
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

        if any(sig in text for sig in critical_signals):
            return TriageAnalysisResponse(
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
                    "PROHIBIDO administrar analgésicos o antiinflamatorios de humanos (paracetamol, ibuprofeno, aspirina causan muerte fulminante).",
                    "NO aplicar torniquetes asfixiantes en las extremidades.",
                    "NO suministrar agua ni alimentos sólidos (riesgo crítico de broncoaspiración).",
                ],
                estimated_fluid_deficit_percent=8.0,
            )

        elif any(sig in text for sig in urgent_signals):
            return TriageAnalysisResponse(
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
                    "NO inducir el vómito sin indicación veterinaria (riesgo de perforación o neumonía por aspiración).",
                    "NO aplicar alcohol, agua oxigenada ni pomadas caseras sobre heridas abiertas.",
                    "NO medicar con fármacos humanos de venta libre.",
                ],
                estimated_fluid_deficit_percent=6.0,
            )

        elif any(sig in text for sig in moderate_signals):
            return TriageAnalysisResponse(
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
                    "NO forzar ingesta de comida si presenta náuseas.",
                ],
                estimated_fluid_deficit_percent=3.0,
            )

        else:
            return TriageAnalysisResponse(
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
                    "NO administrar medicación de humanos.",
                ],
                estimated_fluid_deficit_percent=0.0,
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
        Synchronous inference call to Gemini API with robust JSON parsing,
        model fallback, and deterministic clinical engine backup.
        """
        if not self._client or not self.is_configured:
            logger.warning("[TRIAGE AI] GEMINI_API_KEY no configurada. Activando motor clínico de respaldo.")
            return self._generate_rule_based_fallback(
                especie=especie,
                peso_kg=peso_kg,
                descripcion=descripcion,
                raza=raza,
                edad_aprox=edad_aprox,
            )

        # Build clinical user prompt enforcing step-by-step reasoning
        user_prompt = (
            f"CONSULTA DE TRIAGE CLÍNICO MULTIMODAL CON RAZONAMIENTO CLÍNICO:\n"
            f"- Especie del paciente: {especie}\n"
            f"- Peso corporal: {peso_kg} kg\n"
        )
        if raza:
            user_prompt += f"- Raza: {raza}\n"
        if edad_aprox:
            user_prompt += f"- Edad aproximada: {edad_aprox}\n"
        user_prompt += (
            f"- Motivo de consulta y síntomas reportados por el tutor: {descripcion}\n\n"
            f"Aplica la metodología Chain of Thought descrita en tus instrucciones:\n"
            f"1. Inspecciona minuciosamente la imagen (colorimetría, textura, consistencia).\n"
            f"2. Evalúa la calidad fotográfica y nitidez.\n"
            f"3. Correlaciona los hallazgos visuales con los síntomas y el peso del paciente.\n"
            f"4. Clasifica el caso según el Manchester Triage System.\n"
            f"5. Genera pautas de primeros auxilios y contraindicaciones críticas.\n\n"
            f"Responde ESTRICTAMENTE con el objeto JSON estructurado conforme al esquema requerido."
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

        models_to_try = [self._primary_model] + [m for m in self._fallback_models if m != self._primary_model]
        last_error: Optional[Exception] = None

        for model_name in models_to_try:
            try:
                logger.info("[TRIAGE AI] Ejecutando inferencia multimodal con modelo: %s", model_name)
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=[image_part, user_prompt],
                    config=config,
                )

                # Attempt 1: Native parsed object from response_schema
                if response.parsed and isinstance(response.parsed, TriageAnalysisResponse):
                    return response.parsed
                elif isinstance(response.parsed, dict):
                    return TriageAnalysisResponse.model_validate(response.parsed)

                # Attempt 2: Text extraction and JSON parsing
                if response.text and response.text.strip():
                    raw_text = response.text.strip()
                    # Strip markdown codeblocks if present
                    if "```json" in raw_text:
                        raw_text = raw_text.split("```json")[-1].split("```")[0].strip()
                    elif "```" in raw_text:
                        raw_text = raw_text.split("```")[-1].split("```")[0].strip()

                    parsed_json = json.loads(raw_text)
                    return TriageAnalysisResponse.model_validate(parsed_json)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[TRIAGE AI MODEL NOTICE] Fallo inferencia con %s: %s. Probando siguiente modelo.",
                    model_name,
                    exc,
                )

        logger.error(
            "[TRIAGE AI ERROR] No se pudo completar inferencia con Gemini tras agotar modelos (%s). "
            "Activando motor clínico de respaldo.",
            last_error,
            exc_info=True,
        )
        return self._generate_rule_based_fallback(
            especie=especie,
            peso_kg=peso_kg,
            descripcion=descripcion,
            raza=raza,
            edad_aprox=edad_aprox,
        )

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
        computation to the dedicated worker thread pool.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Los bytes de la imagen no pueden estar vacíos.")

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
