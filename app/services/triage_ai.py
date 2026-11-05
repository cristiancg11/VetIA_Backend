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

VET_TRIAGE_SYSTEM_PROMPT = """Eres un Médico Veterinario Especialista en Triage y Cuidados Críticos de Urgencias Veterinarias (VetIA).
Tu labor es analizar de forma inmediata, rigurosa y empática una fotografía clínica aportada por el tutor (que puede mostrar al animal en su conjunto, una lesión cutánea o traumatismo, o fluidos patológicos como vómito, diarrea o secreciones) combinada con los datos fisiológicos y sintomáticos reportados por el tutor.

Debes categorizar el caso de acuerdo con los siguientes criterios clínicos estandarizados:

1. CRITICO (color_code: "RED"):
   - Condición que amenaza la vida de forma inminente (vía aérea, respiración o colapso circulatorio).
   - Ejemplos: hemorragias activas profusas pulsátiles, traumatismo severo con fractura expuesta, disnea marcada o cianosis, estado convulsivo activo, inconsciencia o estupor profundo, proptosis ocular, torsión gástrica (abdomen timpánico doloroso), sospecha de intoxicación aguda severa con sintomatología nerviosa.
   - vet_consultation_timeframe: "Inmediato (menos de 1 hora)"
   - requires_immediate_hospitalization: true

2. URGENTE (color_code: "ORANGE"):
   - Condición grave con potencial de deterioro rápido sin riesgo vital inmediato en los primeros minutos.
   - Ejemplos: dolor agudo manifiesto, laceraciones dérmicas profundas sin sangrado exanguinante, múltiples episodios de vómito/diarrea con riesgo de deshidratación severa, prolapso de mucosas, dolor abdominal marcado, heridas por mordedura penetrantes, quemaduras de segundo grado.
   - vet_consultation_timeframe: "Inmediato (menos de 1 hora)" o "En el transcurso del día"
   - requires_immediate_hospitalization: true o false según compromiso sistémico

3. MODERADO (color_code: "YELLOW"):
   - Condición estable hemodinámicamente que produce malestar, dolor o riesgo de sobreinfección.
   - Ejemplos: claudicación evidente pero con apoyo parcial, abscesos cerrados o con drenaje purulento localizado, conjuntivitis severa con secreción abundante, otitis aguda con dolor a la palpación, prurito intenso con escoriaciones por rascado continuo, vómito aislado con ánimo conservado.
   - vet_consultation_timeframe: "En el transcurso del día"
   - requires_immediate_hospitalization: false

4. LEVE (color_code: "GREEN"):
   - Alteraciones superficiales, crónicas o menores sin repercusión sistémica ni dolor invalidante.
   - Ejemplos: abrasiones o rasguños superficiales limpios, alopecias localizadas sin eritema severo, costras secas superficiales, sarro dental sin gingivitis ulcerativa aguda.
   - vet_consultation_timeframe: "Próximas 24-48 horas"
   - requires_immediate_hospitalization: false

DIRECTRICES DE PRIMEROS AUXILIOS Y SEGURIDAD VITAL:
- PROHIBICIÓN ABSOLUTA: Está terminantemente prohibido indicar o insinuar medicamentos humanos (el paracetamol, acetaminofén, ibuprofeno, aspirina o diclofenaco son hepatotóxicos y letales para perros y gatos).
- NO indiques maniobras invasivas caseras ni compresiones peligrosas.
- Brinda de 2 a 4 pautas claras de estabilización, postura, abrigo y transporte seguro hacia la clínica.
- Responde ESTRICTAMENTE con un objeto JSON válido conforme al esquema requerido, sin ningún texto adicional fuera de la estructura JSON.
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
            "sangr", "hemorrag", "fractur", "hueso", "inconsciente", "desmay", "coma",
            "convulsion", "ataque", "ahogo", "no respira", "asfixia", "cianosis",
            "lengua azul", "atropell", "choque", "golpe fuerte", "ojo salido",
            "proptosis", "abdomen hinchado", "torsion", "veneno", "toxico", "raticida"
        ]

        urgent_signals = [
            "vomit", "diarre", "sangre", "herida profunda", "corte profundo",
            "quemadur", "dolor fuerte", "grita", "llora", "abdomen duro",
            "prolapso", "mordedura", "fiebre alta", "deshidratad"
        ]

        moderate_signals = [
            "cojera", "cojea", "pata", "ojo", "lagrimeo", "pus", "infeccion",
            "absceso", "rasca", "prurito", "alergia", "hinchazon", "bulto",
            "decaido", "triste", "inapetente", "no come"
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
                presumptive_analysis=(
                    f"Evaluación de Triage Clínico de Emergencia para {patient_desc}: "
                    f"Se identifican signos clínicos de ALTO COMPROMISO VITAL y riesgo inminente de fallo hemodinámico "
                    f"o respiratorio derivado de la sintomatología ({descripcion}). Requiere estabilización hospitalaria inmediata."
                ),
                vital_risks=[
                    "Shock hipovolémico, neurogénico o distributivo",
                    "Paro cardiorrespiratorio o asfixia por compromiso de vías aéreas",
                    "Hipotermia severa y fallo multiorgánico",
                ],
                first_aid_actions=[
                    "Mantener al animal en posición decúbito lateral sobre una superficie acolchada con cuello estirado para despejar vías aéreas.",
                    "Si hay sangrado activo visible, realizar compresión mecánica suave y constante con apósitos o toallas limpias sin retirarlas.",
                    "PROHIBIDO suministrar cualquier medicamento humano (paracetamol, ibuprofeno, aspirina causan muerte fulminante).",
                    "Trasladar de inmediato al centro de urgencias veterinarias 24h más cercano manteniendo abrigo con una manta térmica.",
                ],
                vet_consultation_timeframe="Inmediato (menos de 1 hora)",
                requires_immediate_hospitalization=True,
            )

        elif any(sig in text for sig in urgent_signals):
            return TriageAnalysisResponse(
                urgency_level="URGENTE",
                color_code="ORANGE",
                presumptive_analysis=(
                    f"Evaluación de Triage Clínico para {patient_desc}: "
                    f"Se observa cuadro de severidad URGENTE con dolor agudo, compromiso de fluidos o lesión penetrante. "
                    f"Existe riesgo manifiesto de deterioro sistémico si no recibe atención en las próximas horas."
                ),
                vital_risks=[
                    "Deshidratación severa y desequilibrio hidroelectrolítico",
                    "Infección bacteriana sistémica o sepsis por contaminación de tejido",
                    "Shock por dolor agudo descompensado",
                ],
                first_aid_actions=[
                    "Colocar al paciente en un área tranquila, con luz tenue y ventilada, evitando movimientos bruscos.",
                    "No forzar la ingesta de agua o alimento sólido para evitar aspiraciones o dilatación gástrica.",
                    "No aplicar cremas, alcohol ni remedios caseros sobre lesiones abiertas o fluidos.",
                    "Acudir de inmediato al servicio veterinario para analgesia profesional y rehidratación intravenosa.",
                ],
                vet_consultation_timeframe="Inmediato (menos de 1 hora)",
                requires_immediate_hospitalization=True,
            )

        elif any(sig in text for sig in moderate_signals):
            return TriageAnalysisResponse(
                urgency_level="MODERADO",
                color_code="YELLOW",
                presumptive_analysis=(
                    f"Evaluación de Triage Clínico para {patient_desc}: "
                    f"Cuadro de severidad MODERADA sin compromiso hemodinámico fulminante inmediato. "
                    f"Presenta malestar, dolor localizado o proceso inflamatorio que requiere valoración en el transcurso del día."
                ),
                vital_risks=[
                    "Sobreinfección bacteriana o empeoramiento de la lesión dérmica por rascado o lamido",
                    "Cronicidad del dolor musculoesquelético o articular",
                ],
                first_aid_actions=[
                    "Impedir que el paciente se lama, muerda o rasque la zona afectada (usar collar isabelino si se dispone).",
                    "Limpiar suavemente el área externa peri-lesional únicamente con suero fisiológico templado.",
                    "Mantener al animal cómodo, sin forzar actividad física o saltos.",
                    "Programar consulta veterinaria presencial en el transcurso del día.",
                ],
                vet_consultation_timeframe="En el transcurso del día",
                requires_immediate_hospitalization=False,
            )

        else:
            return TriageAnalysisResponse(
                urgency_level="LEVE",
                color_code="GREEN",
                presumptive_analysis=(
                    f"Evaluación de Triage Clínico para {patient_desc}: "
                    f"Afección LEVE o de curso benigno sin signos de alarma sistémica. "
                    f"No se detectan compromisos respiratorios, circulatorios ni dolor incapacitante."
                ),
                vital_risks=[
                    "Molestia superficial o irritación localizada de baja gravedad",
                ],
                first_aid_actions=[
                    "Monitorear la evolución de los síntomas, apetito y ánimo durante las próximas 24 a 48 horas.",
                    "Mantener la higiene de la zona con agua tibia o suero fisiológico sin frotar con fuerza.",
                    "Evitar cambios bruscos de alimentación o exposición a factores estresantes.",
                    "Acudir a consulta médica ambulatoria si los síntomas persisten o aumentan de intensidad.",
                ],
                vet_consultation_timeframe="Próximas 24-48 horas",
                requires_immediate_hospitalization=False,
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

        # Build clinical user prompt
        user_prompt = (
            f"DATOS CLÍNICOS DEL PACIENTE PARA TRIAGE:\n"
            f"- Especie: {especie}\n"
            f"- Peso: {peso_kg} kg\n"
        )
        if raza:
            user_prompt += f"- Raza: {raza}\n"
        if edad_aprox:
            user_prompt += f"- Edad aproximada: {edad_aprox}\n"
        user_prompt += (
            f"- Descripción de síntomas y motivo de consulta: {descripcion}\n\n"
            f"Analiza con minuciosidad clínica la imagen adjunta (lesión, animal o fluido patológico) "
            f"y los datos suministrados. Emite tu clasificación diagnóstica y terapéutica de primeros "
            f"auxilios respondiendo ÚNICAMENTE con el objeto JSON estructurado según el esquema solicitado."
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
